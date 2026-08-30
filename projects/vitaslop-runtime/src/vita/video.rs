//! SceMp4: demuxing a title's full-motion video.
//!
//! A title's movie is an MP4 the guest asks `SceMp4` to DEMUX for it. It decodes the
//! result itself, through `SceVideodec`/`SceAvcdec` for the picture and `SceAudiodec` for
//! the sound - so this module produces elementary stream and nothing else. The decoders
//! live in [`crate::vita::avcdec`] and [`crate::vita::audiodec`].
//!
//! # What is established, and how
//!
//! Most of `SceMp4` is undocumented - the vitasdk NID database has no entry for the
//! library. What is implemented here was read off the one title that uses it, with
//! `cargo run -p vitaslop-runtime --example call_sites|disasm`:
//!
//! - **`sceMp4OpenFile(path, ...)` returns a HANDLE, and zero is its failure.** The call
//!   site stores the result as the session handle and branches only on zero. `r0` is the
//!   path - MEASURED, it is `"app0:Data/Movie/TITLE.mp4"` on a real run.
//! - **`sceMp4StartFileStreaming(handle, out_word, out_struct)`** returns a status where
//!   `>= 0` is success, and the guest reads the word back immediately.
//! - **`0x8be0e3d3(handle, arg, out)`** fills a 0x158-byte struct describing one access
//!   unit, and is the fetch. Its first two words ARE established - the caller branches on
//!   them - and the rest is filled on the most probable reading and reported once per run.
//!   See [`mp4_get_next_unit`].
//!
//! # Why the units are Annex B
//!
//! What comes out of an MP4 sample table is length-prefixed NALs with the parameter sets
//! held once, out of band, in the `avcC` record. What a video-decoder API is given is an
//! ELEMENTARY STREAM: it knows nothing about containers, so it has to be self-describing.
//! The demuxer therefore rewrites each sample into Annex B and puts the parameter sets in
//! front of every sync sample - which is also what makes the stream joinable at a seek.


/// One movie being demuxed: the container, and where playback is in it.
pub struct MovieSession {
    /// The guest path this was opened from, for diagnostics.
    pub path: String,
    /// The descriptor the movie is STREAMED through. It stays open for the session.
    ///
    /// # This used to be the whole file, and that was a browser-only stall
    ///
    /// The first version read the entire container into a `Vec` at open - "a movie is tens
    /// of megabytes against a heap measured in gigabytes". True on a desktop, where the
    /// container is already resident and the read is a memcpy. In a BROWSER the file is not
    /// resident: it is served out of storage a range at a time, so the same loop became 124
    /// storage reads and a 31 MB allocation, all inside ONE host call that yields to
    /// nothing - no presentation, no decoder callbacks, no audio, for as long as it took.
    /// The desktop never showed it; a phone stopped dead at the frame the movie opens on.
    ///
    /// So the header is read at open and the sample data is read per access unit, which is
    /// what the device does and what the API is shaped for.
    pub fd: i32,
    /// The parsed container.
    pub mp4: crate::mp4::Mp4,
    /// The video track's parameter sets, so a sample can be rewritten into a
    /// self-describing elementary stream. See the module docs.
    pub avcc: crate::mp4::AvcC,
    /// Next sample to hand over for each SERVED track, in decode order, as
    /// `(index into `mp4.tracks`, next sample)`. See [`next_unit_track`]: a movie is one
    /// interleaved stream of units and the title routes them by the stream id each unit
    /// carries, so the demuxer has to keep a cursor per track rather than per movie.
    pub cursors: Vec<(usize, usize)>,
    /// Next video sample (access unit) to hand over, in decode order.
    pub next_sample: usize,
    /// The handle handed to the guest. Never zero - zero is this API's failure value.
    pub handle: i32,
    /// The buffer the GUEST supplied at open, and its size: `r3` and the first stack
    /// argument of `sceMp4OpenFile` (MEASURED as `0x8328fcc0` and `0xc0000` on a real run).
    ///
    /// Access units are delivered INTO this rather than into memory of our own. The guest
    /// hands a library a working buffer precisely so the library uses it, and the fetch
    /// consumer stamps this base and size onto every unit descriptor it builds - which is
    /// what a null there was faulting on.
    pub unit_buffer: Option<(u32, u32)>,
    /// The movie's own AAC decoder and the frames it has produced ahead of the title
    /// asking for them - see [`crate::vita::audiodec`] for why the decoding happens here,
    /// at DELIVERY time, rather than inside `sceAudiodecDecode`.
    pub audio: Option<MovieAudio>,
    /// Which stream ids the title has enabled, as it named them.
    pub enabled_streams: std::collections::BTreeSet<u32>,
    /// Guest-clock microsecond at which this movie's timeline is taken to have started, set
    /// when the first unit is handed over. `None` until then. See [`movie_unit_wait_us`]: it
    /// is what makes "this unit's PTS" comparable with "now".
    pub timeline_origin_us: Option<u64>,
    /// Pictures handed to the guest so far, for the one line a run says about playback.
    pub delivered: u64,
    /// Consecutive times the pacing gate has told the demuxer "not yet" without a unit
    /// having been taken in between, and whether the stall it implies has been reported.
    ///
    /// A refusal is normal - the gate exists to hand units out at the movie's own rate - but a
    /// refusal that never ends is a STOPPED MOVIE, and the two look identical from outside.
    /// MEASURED on the retail golf title's opening: the player's video thread spins 413 times a
    /// displayed frame waiting for a unit while its demuxer is refused 17 times a frame, and the
    /// run says nothing at all. See `report_unit_gate_stall`.
    pub gate_refusals: u64,
    pub reported_gate_stall: bool,
}

// The `#[hostcall]` macro rewrites these signatures and emits its own fully-qualified
// paths, so a module of nothing but host calls has no use for a plain `use` of them -
// hence the qualified types below rather than an import that reads as unused.
use crate::hostcall;

/// `SCE_ERROR_ERRNO_ENOENT`. The API's own error table is not documented anywhere
/// clean-room, so this reports the failure a missing movie file would produce - the
/// closest thing to "this stream is not available" that is a known-good Sce error value.
const SCE_ERROR_ERRNO_ENOENT: i32 = 0x8001_0002u32 as i32;

/// int sceMp4OpenFile(...)
/// Report that the movie cannot be opened, so the title skips it.
///
/// **This returns ZERO, not an error code, and that is not a detail.** MEASURED from the
/// one call site (`0x8117d8b8` in one title's eboot): the result is stored straight into
/// the session as a handle and only ZERO takes the error path -
/// `str r0,[r4,0x1c]; cmp r0,#0; bne`. The previous version returned
/// `SCE_ERROR_ERRNO_ENOENT`, which is non-zero, so the guest read the failure as a valid
/// handle and streamed on against it - which is exactly the "the title IGNORES the failed
/// open" behaviour recorded here before, and it was ours, not the title's.
#[hostcall]
pub(super) fn mp4_open_file(
    ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    path: crate::host::Ptr,
    _a1: crate::host::Ptr,
    _a2: crate::host::Ptr,
    buffer: crate::host::Ptr,
    buffer_size: u32,
) -> i32 {
    do_open_file(ctx, st, path.addr(), buffer.addr(), buffer_size)
}

/// The body of [`mp4_open_file`], as a plain function so it can use guard clauses.
fn do_open_file(
    ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    path_ptr: u32,
    buffer: u32,
    buffer_size: u32,
) -> i32 {
    let path = substitute_movie(ctx.read_cstr(path_ptr, 256));
    match open_movie(st, &path, buffer, buffer_size) {
        Ok(handle) => handle,
        Err(reason) => {
            report_no_video(st, &path, &reason);
            0
        }
    }
}

/// >>> OPEN A DIFFERENT MOVIE THAN THE TITLE ASKED FOR
/// (`VITASLOP_MOVIE_SUBSTITUTE=app0:Data/Movie/SOMETHING.mp4`).
///
/// A diagnostic, and it exists for one specific reason: the movie a title opens on its front
/// screen may have no AUDIO TRACK, while the ones that do are behind thousands of frames of
/// menu navigation. Without this, the whole guest audio path - `sceMp4GetNextUnit`'s stream
/// ids, the read-ahead decode, `sceAudiodecDecode` filling `pPcm` - can only be exercised by
/// a run that first plays most of a game.
///
/// It substitutes the PATH and nothing else, so everything downstream is what the title
/// would have done, and it says so once: a run that quietly played a different movie than
/// the title asked for would be a confusing thing to inherit.
fn substitute_movie(asked: String) -> String {
    let Ok(sub) = crate::knobs::var("VITASLOP_MOVIE_SUBSTITUTE") else { return asked };
    if sub.trim().is_empty() || sub == asked {
        return asked;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            target: "vitaslop::movie",
            asked = %asked, opening = %sub,
            "VITASLOP_MOVIE_SUBSTITUTE: opening a DIFFERENT movie than the title asked for"
        );
    });
    sub
}

/// Read the file and demux it.
fn open_movie(
    st: &mut crate::host::VitaState,
    path: &str,
    buffer: u32,
    buffer_size: u32,
) -> Result<i32, String> {
    // SCE_O_RDONLY is 1, not 0 - the Vita's flags are a value, not a bit position, and
    // opening with 0 yields a descriptor that is neither readable nor writable: the open
    // SUCCEEDS and every read then returns nothing, which is a much more confusing failure
    // than a refused open.
    const SCE_O_RDONLY: u32 = 0x0001;
    let fd = st.io_open(path, SCE_O_RDONLY);
    if fd < 0 {
        return Err(format!("{path} cannot be opened ({fd:#x})"));
    }
    // >>> ONLY THE HEADER IS READ. See `MovieSession::fd` for what reading the whole file
    // cost, and why a desktop could never show it.
    let size = st.io_size(path).unwrap_or(0);
    if size == 0 {
        st.io_close(fd);
        return Err(format!("{path} is empty or its size is unknown"));
    }
    let (moov_at, moov_len) = crate::mp4::find_moov(size, |at, len| read_at(st, fd, at, len))
        .map_err(|e| format!("{path}: no moov box ({e:?})"))?;
    // A header is hundreds of kilobytes at most; a cap turns a mis-parse into a refusal
    // rather than an allocation the size of the file.
    const MAX_MOOV: u64 = 8 * 1024 * 1024;
    if moov_len > MAX_MOOV {
        st.io_close(fd);
        return Err(format!("{path}: its moov box is {moov_len} bytes, past the {MAX_MOOV} cap"));
    }
    let moov = read_at(st, fd, moov_at, moov_len as usize);
    if moov.len() as u64 != moov_len {
        st.io_close(fd);
        return Err(format!(
            "{path}: read {} of the {moov_len}-byte moov box at {moov_at}",
            moov.len()
        ));
    }

    let mp4 = crate::mp4::Mp4::parse_moov(&moov).map_err(|e| format!("{path}: {e:?}"))?;
    let track = mp4
        .track(crate::mp4::TrackKind::Video)
        .ok_or_else(|| format!("{path} has no video track"))?;
    if &track.codec != b"avc1" && &track.codec != b"avc3" {
        return Err(format!(
            "{path}: the video track is {:?}, not H.264",
            String::from_utf8_lossy(&track.codec)
        ));
    }
    if track.codec_config.is_empty() {
        return Err(format!("{path}: the video track carries no avcC record"));
    }
    let avcc = crate::mp4::AvcC::parse(&track.codec_config)
        .map_err(|e| format!("{path}: the avcC record cannot be read ({e:?})"))?;
    if avcc.sps.is_empty() || avcc.pps.is_empty() {
        return Err(format!(
            "{path}: the avcC record carries {} SPS and {} PPS - a decoder cannot be \
             configured from it",
            avcc.sps.len(),
            avcc.pps.len()
        ));
    }
    let (width, height) = (track.width, track.height);
    let samples = track.samples.len();
    let audio = mp4
        .track(crate::mp4::TrackKind::Audio)
        .map(|t| format!("{} ({} samples)", String::from_utf8_lossy(&t.codec), t.samples.len()));

    // A handle, not a status: see the module docs. Any non-zero value will do; a small
    // counter keeps it recognisable in a log.
    let handle = 0x4d50_0001u32 as i32;
    // WARN so a device's default log level carries it - see the note on the first-picture
    // report in `vita::avcdec` for why the movie path says its landmarks out loud.
    tracing::warn!(
        target: "vitaslop::movie",
        %path, width, height, samples, ?audio,
        buffer = format_args!("{buffer:#010x}"), buffer_size,
        "SceMp4: demuxing a movie"
    );
    st.movie = Some(MovieSession {
        path: path.to_string(),
        fd,
        cursors: served_cursors(&mp4),
        audio: open_movie_audio(st, &mp4),
        mp4,
        avcc,
        next_sample: 0,
        handle,
        unit_buffer: if buffer != 0 && buffer_size != 0 {
            Some((buffer, buffer_size))
        } else {
            None
        },
        enabled_streams: std::collections::BTreeSet::new(),
        timeline_origin_us: None,
        delivered: 0,
        gate_refusals: 0,
        reported_gate_stall: false,
    });
    Ok(handle)
}

/// Read `len` bytes at an absolute file offset through an open descriptor.
///
/// Short reads are normal at the end of a file and are returned as-is; every caller here
/// checks the length it got rather than assuming the one it asked for.
fn read_at(st: &mut crate::host::VitaState, fd: i32, at: u64, len: usize) -> Vec<u8> {
    const SEEK_SET: i32 = 0;
    if st.io_lseek(fd, at as i64, SEEK_SET) < 0 {
        return Vec::new();
    }
    st.io_read(fd, len).unwrap_or_default()
}

/// Say once per run that this title wanted to play a movie and did not get one, and WHY.
/// Unconditional, not behind a debug flag: the picture the run produces is missing
/// whatever the movie would have shown, and nothing else in the output says so.
fn report_no_video(st: &mut crate::host::VitaState, path: &str, reason: &str) {
    if st.reported_no_video {
        return;
    }
    st.reported_no_video = true;
    tracing::warn!(
        target: "vitaslop::movie",
        %path, %reason,
        "SceMp4: this movie will NOT play. It is reported unavailable so the title skips          it and carries on; whatever it would have shown is missing from this run."
    );
}

/// int sceMp4StartFileStreaming(handle, out_word, out_struct)
///
/// MEASURED at the call site: `r0` is the handle from `sceMp4OpenFile`, `r1` and `r2` are
/// out-parameters, and the guest treats `>= 0` as success and reads the word at `r1` back
/// immediately.
///
/// **What the out-parameters MEAN is not established**, so this fills them with zero and
/// says so - once, unconditionally. That is the one approximation left in this path, and
/// it is the shape of thing that must report itself: a zero the guest reads as a stream
/// property would otherwise be indistinguishable from a real answer.
#[hostcall]
pub(super) fn mp4_start_file_streaming(
    ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    handle: i32,
    out_word: crate::host::Ptr,
    out_struct: crate::host::Ptr,
    _a3: crate::host::Ptr,
) -> i32 {
    do_start_file_streaming(ctx, st, handle, out_word.addr(), out_struct.addr())
}

fn do_start_file_streaming(
    ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    handle: i32,
    out_word: u32,
    out_struct: u32,
) -> i32 {
    let Some(movie) = st.movie.as_mut() else {
        // No session: the guest is streaming a handle it never received.
        return SCE_ERROR_ERRNO_ENOENT;
    };
    if handle != movie.handle {
        return SCE_ERROR_ERRNO_ENOENT;
    }

    // Nothing is submitted here: the guest owns the decoder, and the first thing it does
    // after this returns is ask for a unit.

    // >>> VIDEO STREAMS, NOT STREAMS. MEASURED, AND IT IS WHAT STALLS A RETAIL MOVIE.
    //
    // The title reads this word and then, ONCE PER COUNT, calls
    // `sceVideodecInitLibraryWithUnmapMem` + `sceAvcdecCreateDecoder` and
    // `sceMp4EnableStream(i)`. So what it is asking is "how many streams do I need an H.264
    // DECODER for", and answering with the served-track count offered it our AUDIO track as a
    // second video stream: it built decoder handle 2 for it, enabled stream 1, queued the AAC
    // units on it, and its player then stopped - `avPlayer VideoDec` parked on a condition
    // only it ever touches and `avPlayer Demux` on one only the video thread ever signals,
    // with **5 access units submitted on the browser and 7 natively** out of 2,555.
    //
    // The A/B that settles it, one variable, same build: `VITASLOP_MP4_AUDIO=0` (which drops
    // the audio cursor and so makes this word 1) plays the movie - **170+ units in the same
    // 800 frames**, at real sizes, and the parked pair is gone. Reporting the VIDEO count with
    // the audio track still served is the same fact without withholding anything: the title
    // enables the streams it enumerated, and a stream it never enables is never handed a unit.
    //
    // Whether an audio track can reach this title AT ALL is a separate open question - it has
    // never called `sceAudiodecDecode` on any run - and it is not answered by handing it one
    // it cannot recognise. See `served_cursors`.
    let streams = served_video_streams(&movie.mp4, &movie.cursors);
    if out_word != 0 {
        // EXPERIMENT, and the reasoning is on the record: the guest's wrapper around this
        // call ends `ldr r0,[r4,0x20]; pop {r4,pc}` - it RETURNS this word - so whatever it
        // means, zero is what a caller reads as "nothing came of starting the stream". A
        // stream id or a stream count are both shapes where 1 is the working value, and 1
        // was what went in while every movie had one served stream.
        //
        // Now that a movie can have TWO (video and audio), the count is the reading that can
        // be TESTED. IT WAS, AND IT CAME BACK NEGATIVE: with a two-stream movie substituted
        // into the title's front-screen player (`VITASLOP_MOVIE_SUBSTITUTE`) and this
        // reporting 2, the title created its VIDEO decoder and no audio decoder at all - so
        // whatever this word is, it is not what gates that. The value stays as the count
        // because it is the more defensible of two guesses and because the run that tried it
        // played the movie and walked on through the menus unharmed; the field's meaning is
        // still open.
        ctx.write_u32(out_word, streams);
    }
    if out_struct != 0 {
        for word in 0..16u32 {
            ctx.write_u32(out_struct + word * 4, 0);
        }
        // The fetch consumer reads the first two words of THIS struct back out of the
        // session and stamps them onto every unit descriptor (`ldrd r4,r5,[session+0x60]`).
        // Zeros there are what the guest was faulting on, and the buffer it supplied at
        // open is the only pair of values in scope that fits base/size.
        if let Some((buffer, size)) = st.movie.as_ref().and_then(|m| m.unit_buffer) {
            ctx.write_u32(out_struct, buffer);
            ctx.write_u32(out_struct + 4, size);
        }
    }
    // Where the two out-parameters live and WHO asked, because what the second one carries is
    // still open (see the report below) and the only way to settle it is to disassemble the
    // caller that reads it back. The link register is the join to `refs` and `VITASLOP_PEEK`.
    tracing::debug!(
        target: "vitaslop::movie",
        out_word = format_args!("{out_word:#010x}"),
        out_struct = format_args!("{out_struct:#010x}"),
        streams,
        lr = format_args!("{:#010x}", ctx.regs[14]),
        "sceMp4StartFileStreaming"
    );
    report_unknown_stream_properties(st);
    0
}

/// Say, once, that the stream properties handed back are placeholders.
fn report_unknown_stream_properties(st: &mut crate::host::VitaState) {
    if st.reported_stream_properties {
        return;
    }
    st.reported_stream_properties = true;
    tracing::warn!(
        target: "vitaslop::movie",
        "SceMp4: sceMp4StartFileStreaming succeeded but what its two out-parameters carry          is not established. Anything the title derives from them (a buffer size, a stream          count, a duration) is therefore an assumption, not a result."
    );
}

/// `0x8be0e3d3(handle, arg, out)` - hand back the next decoded picture.
///
/// RECOVERED from the one call site (`0x8117d96c`), which memsets a 0x158-byte struct,
/// passes it as `r2`, and then branches on what came back. The branch structure is what
/// establishes the first two fields:
///
/// ```text
/// out[0x00] <  0   -> the caller returns -1 (an error)
/// out[0x00] == 0   -> a UNIT was returned; out[0x04] must be 1 or the caller logs and
///                     fails, so 1 is the VIDEO unit type
/// out[0x00] == 1   -> no unit; out[0x04] must be 2 or 4, and the caller takes this path
///                     silently - which is what "nothing right now" is
/// out[0x00] >  1   -> the caller logs and fails
/// ```
///
/// In the unit case the caller copies `out[0x08]`/`out[0x0c]` into its own picture
/// descriptor as a pair, the 64-bit `out[0x30]` as a timestamp, `out[0x10]`/`out[0x14]` as
/// 16-bit values, and `out[0x18]`/`out[0x1c]` as bytes. **Which of the pair is the pointer,
/// and what the bytes mean, is NOT established by the disassembly** - those fields are
/// filled on the most probable reading and reported once per run, so a wrong guess reads as
/// a stated assumption rather than as a silently wrong picture.
#[hostcall]
pub(super) fn mp4_get_next_unit(
    ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    handle: i32,
    _arg: crate::host::Ptr,
    out: crate::host::Ptr,
) -> i32 {
    do_get_next_unit(ctx, st, handle, out.addr())
}

/// Field offsets in the 0x158-byte unit struct, as recovered above.
mod unit {
    /// Status: 0 = a unit follows, 1 = nothing right now, negative = error.
    pub const STATUS: u32 = 0x00;
    /// Unit type: 1 with status 0 is video; 2 or 4 with status 1 is "nothing".
    pub const KIND: u32 = 0x04;
    /// Picture WIDTH. **Established by arithmetic, not by shape**: the consumer copies
    /// this pair to its own `+0x60`/`+0x64`, and the code that allocates the movie's
    /// display memory computes `align16(+0x60) * align16(+0x64) * 3 / 2 * buffers` - a
    /// 4:2:0 frame, so the pair is width and height. The first reading of this field as a
    /// POINTER is what asked for a 210 MB physically-contiguous block and brought the
    /// title down: the request matched that formula over the buffer address to the byte.
    pub const WIDTH: u32 = 0x08;
    /// Picture HEIGHT - see [`WIDTH`].
    pub const HEIGHT: u32 = 0x0c;
    /// Copied as a 16-bit value to the consumer's `+0x6e`. Role unknown; left zero.
    pub const HALF_A: u32 = 0x10;
    /// Copied as a 16-bit value to the consumer's `+0x70`. Role unknown; left zero.
    pub const HALF_B: u32 = 0x14;
    /// Copied as a byte. Role unknown; left zero.
    pub const BYTE_A: u32 = 0x18;
    /// Copied as a byte. Role unknown; left zero.
    pub const BYTE_B: u32 = 0x1c;
    /// The one whole word copied to the consumer's own `+0x20`, and the only field left
    /// that can carry the ELEMENTARY STREAM LENGTH - which the guest must have, because
    /// it feeds the bytes to its own decoder. Filled with the access unit's byte count.
    pub const ES_SIZE: u32 = 0x20;
    /// 64-bit, copied as one - taken as the presentation timestamp.
    pub const TIMESTAMP: u32 = 0x30;
}

fn do_get_next_unit(
    ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    handle: i32,
    out: u32,
) -> i32 {
    if out == 0 {
        return -1;
    }
    match st.movie.as_ref() {
        Some(movie) if movie.handle == handle => {}
        _ => return -1,
    }

    // `none` never hands a unit back, which is the arm that separates "the playback path
    // itself faults" from "our inferred fields fault". It used to be read and then
    // ignored, so every run with it set was in fact the default arm.
    if unit_mode() == UnitMode::NoUnits {
        ctx.write_u32(out + unit::STATUS, 1);
        ctx.write_u32(out + unit::KIND, 2);
        return 0;
    }

    // This DESCRIBES the stream's current unit; it does not consume it. The consumer keeps
    // the pair at `+0x08`/`+0x0c` as the picture size and sizes the movie's display memory
    // from it, and the title's own DEMUX THREAD is what pulls the elementary stream, with
    // `sceMp4GetNextUnit` and `sceMp4GetNextUnitData`. Advancing here as well would decode
    // every other frame.
    //
    // It used to hand back a DECODED picture, on the reading that SceMp4 decodes for the
    // guest. MEASURED, that reading is wrong: with units suppressed the title's movie
    // thread goes on to call `SceVideodec`/`SceAvcdec`, which it imports - so the title
    // decodes for itself and SceMp4 only demuxes.
    // >>> A UNIT IS NOT OFFERED BEFORE ITS OWN PRESENTATION TIME. See `movie_unit_wait_us`.
    //
    // This is the call that decides whether the title commits to a unit at all: it takes the
    // size from here, allocates exactly that, and only then asks for the data. So this is where
    // "not yet" belongs, and the two other placements tried first are both recorded there.
    let sample = match peek_access_unit(st) {
        Ok(Some(sample)) => sample,
        Ok(None) => {
            // End of stream: the "nothing right now" outcome the caller takes silently.
            ctx.write_u32(out + unit::STATUS, 1);
            ctx.write_u32(out + unit::KIND, 4);
            return 0;
        }
        Err(reason) => {
            let path = st.movie.as_ref().map(|m| m.path.clone()).unwrap_or_default();
            report_no_video(st, &path, &reason);
            return -1;
        }
    };

    let needed = sample.bytes.len() as u32;

    ctx.write_u32(out + unit::STATUS, 0);
    ctx.write_u32(out + unit::KIND, 1);
    ctx.write_u32(out + unit::WIDTH, sample.width);
    ctx.write_u32(out + unit::HEIGHT, sample.height);
    ctx.write_u32(out + unit::HALF_A, 0);
    ctx.write_u32(out + unit::HALF_B, 0);
    ctx.write_u32(out + unit::BYTE_A, 0);
    ctx.write_u32(out + unit::BYTE_B, 0);
    ctx.write_u32(out + unit::ES_SIZE, needed);
    let pts = sample.pts;
    ctx.write_u32(out + unit::TIMESTAMP, pts as u32);
    ctx.write_u32(out + unit::TIMESTAMP + 4, (pts >> 32) as u32);

    if let Some(movie) = st.movie.as_mut() {
        movie.delivered += 1;
    }
    report_unit_assumptions(st, sample.width, sample.height, needed);
    0
}

/// The movie's sound: a host decoder, and the frames it has decoded AHEAD of the title
/// asking for them.
///
/// # Why the decoding happens here and not in `sceAudiodecDecode`
///
/// That call has to return the PCM, and in a browser a host decoder cannot answer during a
/// call. But the engine is the thing demultiplexing the movie - it knows every access unit
/// before the title does, because it is the one handing them over - so each unit is
/// submitted the moment it is delivered and the frames queue up here. By the time the
/// title's audio thread asks, the frame is already decoded, and the guest's synchronous API
/// is served synchronously. See [`crate::vita::audiodec`].
pub struct MovieAudio {
    decoder: Box<dyn vitaslop_platform::audio_dec::AudioDecode>,
    /// Decoded frames, oldest first, each with the access unit it came from.
    ready: std::collections::VecDeque<DecodedFrame>,
    /// Access units submitted and not yet answered, oldest first: `(byte length, first
    /// word)`. The first word is what `sceAudiodecDecode` checks the guest's elementary
    /// stream against, which is how a queue that went out of step says so.
    pending: std::collections::VecDeque<(u32, u32)>,
    /// Frames dropped because nothing collected them - a title that stopped decoding.
    dropped: u64,
}

/// One decoded audio frame, waiting for the guest to ask for it.
pub struct DecodedFrame {
    pub samples: Vec<i16>,
    /// Bytes of elementary stream this frame was decoded from, which is what
    /// `SceAudiodecCtrl::inputEsSize` is owed.
    pub es_size: u32,
    /// The first word of that access unit, for the ordering check.
    pub es_head: u32,
}

/// How many decoded frames to hold for a title that is not collecting them. One AAC frame
/// is about 21 ms of audio, so this is half a second - deep enough that no decoder pipeline
/// is ever the reason one is dropped, shallow enough that a title which stops asking does
/// not accumulate a movie's worth of PCM.
const AUDIO_BACKLOG: usize = 24;

/// Open a decoder for the movie's audio track, if it has one this engine can decode.
///
/// `None` is an ordinary outcome and covers three different things - no audio track, a
/// codec this engine does not decode, and a host with no decoder at all - each of which is
/// reported where it is found, because "the movie is silent" otherwise looks the same from
/// the outside in all three cases.
fn open_movie_audio(
    st: &mut crate::host::VitaState,
    mp4: &crate::mp4::Mp4,
) -> Option<MovieAudio> {
    let track = mp4.track(crate::mp4::TrackKind::Audio)?;
    if &track.codec != b"mp4a" {
        tracing::warn!(
            target: "vitaslop::movie",
            codec = %String::from_utf8_lossy(&track.codec),
            "SceMp4: this movie's audio track is not AAC, so it will be silent"
        );
        return None;
    }
    let stream = vitaslop_platform::audio_dec::AudioStream {
        // The `esds` DecoderSpecificInfo IS the `AudioSpecificConfig` - not the box's own
        // bytes, which is a nest of descriptors around it.
        asc: track.audio_specific_config(),
        channels: 0,
        sample_rate: track.timescale,
    };
    match st.audio_dec.open_aac(&stream) {
        Ok(decoder) => {
            tracing::warn!(
                target: "vitaslop::movie",
                backend = %decoder.describe(),
                samples = track.samples.len(),
                "SceMp4: decoding the movie's sound on the host's own AAC decoder"
            );
            Some(MovieAudio {
                decoder,
                ready: std::collections::VecDeque::new(),
                pending: std::collections::VecDeque::new(),
                dropped: 0,
            })
        }
        Err(e) => {
            tracing::warn!(
                target: "vitaslop::movie",
                error = %e,
                "SceMp4: this host cannot decode the movie's sound, so it will be silent. \
                 The picture is unaffected."
            );
            None
        }
    }
}

/// Submit one delivered audio access unit to the movie's decoder and collect whatever has
/// come back. Called where the unit is handed to the title, which is what makes the decode
/// run AHEAD of the title's own decode call.
fn pump_movie_audio(movie: &mut MovieSession, es: &[u8], pts: i64) {
    let Some(audio) = movie.audio.as_mut() else { return };
    let head = u32::from_le_bytes([
        es.first().copied().unwrap_or(0),
        es.get(1).copied().unwrap_or(0),
        es.get(2).copied().unwrap_or(0),
        es.get(3).copied().unwrap_or(0),
    ]);
    if let Err(e) = audio.decoder.submit(es, pts) {
        report_audio_decode_failed(&e.to_string());
        return;
    }
    audio.pending.push_back((es.len() as u32, head));
    loop {
        match audio.decoder.poll() {
            Ok(Some(pcm)) => {
                let (es_size, es_head) = audio.pending.pop_front().unwrap_or((0, 0));
                audio.ready.push_back(DecodedFrame { samples: pcm.samples, es_size, es_head });
                while audio.ready.len() > AUDIO_BACKLOG {
                    audio.ready.pop_front();
                    audio.dropped += 1;
                    // A movie quietly missing half a second of sound is exactly the kind of
                    // thing that gets noticed as "the audio is a bit odd" a week later.
                    report_audio_backlog_dropped(audio.dropped);
                }
            }
            Ok(None) => break,
            Err(e) => {
                report_audio_decode_failed(&e.to_string());
                break;
            }
        }
    }
}

/// Take the oldest decoded audio frame, for `sceAudiodecDecode`.
pub(crate) fn take_decoded_audio(st: &mut crate::host::VitaState) -> Option<DecodedFrame> {
    st.movie.as_mut()?.audio.as_mut()?.ready.pop_front()
}

/// Say, once, that decoded audio was thrown away because the title stopped collecting it.
/// Not necessarily a defect - a title that stops playing a movie stops decoding its sound -
/// but it is the difference between "the sound is late" and "the sound is gone", and only
/// this says which.
fn report_audio_backlog_dropped(dropped: u64) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            target: "vitaslop::movie",
            dropped,
            backlog = AUDIO_BACKLOG,
            "decoded audio frames were dropped because the title stopped collecting them -              the movie is short by that much sound from here"
        );
    });
}

/// Say, once, that the movie's audio decoder failed. A movie that loses its sound part way
/// through is exactly the kind of thing that otherwise gets noticed as "the audio is a bit
/// odd" a week later.
fn report_audio_decode_failed(reason: &str) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::error!(
            target: "vitaslop::movie",
            reason,
            "the movie's audio decoder failed; the rest of this movie is silent"
        );
    });
}

/// One access unit, as the title's demux thread receives it.
struct AccessUnit {
    bytes: Vec<u8>,
    width: u32,
    height: u32,
    pts: u64,
    dts: u64,
    sync: bool,
    /// >>> THE STREAM THE UNIT BELONGS TO, AS THE TITLE NUMBERS STREAMS: the 0-BASED INDEX
    /// of the track in the container, NOT its `track_ID`.
    ///
    /// This is what the title routes on - its demux thread looks the id up in its own table
    /// and queues the record on that stream's decoder - so the numbering has to be the one
    /// it uses. MEASURED: on a movie whose tracks are `track_ID` 1 and 2, the title enables
    /// stream **0**. It is counting from zero over the container's streams, and a `track_ID`
    /// would have missed its table entirely.
    stream: u32,
    /// Whether this unit is video. Only a video unit gets the picture-shaped fields.
    video: bool,
}

/// >>> WHICH TRACK OWES THE NEXT UNIT: the one whose next sample has the earliest
/// DECODE time, across every track this engine can serve.
///
/// A movie file is one interleaved stream and its consumer is one demux thread pulling
/// units in decode order; splitting that into "the video track" was fine while nothing else
/// was served, and it is exactly what left a movie silent. Timestamps are compared in
/// MICROSECONDS because the two tracks have different timescales (a 48 kHz audio track and a
/// video track counted in frames), so their raw tick values are not comparable at all.
///
/// `None` when every served track is exhausted, which is the end of the movie.
fn next_unit_track(movie: &MovieSession) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize, u64)> = None;
    for &(track, at) in &movie.cursors {
        // >>> ONLY THE STREAMS THE TITLE ENABLED, once it has enabled any.
        //
        // `sceMp4EnableStream` is how a player says which streams it is prepared to receive,
        // and handing it units for one it did not ask for puts records on a queue nothing
        // drains. MEASURED: this title's front-screen movie player enables stream 0 and
        // nothing else - it is a video-only player - so a movie substituted in with a sound
        // track must not start delivering audio units to it.
        //
        // An empty set means the title has not said, which is the state every unit fetch
        // before the first `EnableStream` is in; everything served is offered there.
        if !movie.enabled_streams.is_empty() && !movie.enabled_streams.contains(&(track as u32)) {
            continue;
        }
        let t = movie.mp4.tracks.get(track)?;
        let Some(sample) = t.samples.get(at) else { continue };
        let us = match t.timescale {
            0 => sample.dts,
            scale => sample.dts.saturating_mul(1_000_000) / scale as u64,
        };
        if best.is_none_or(|(_, _, b)| us < b) {
            best = Some((track, at, us));
        }
    }
    best.map(|(track, at, _)| (track, at))
}

/// The tracks this engine will hand units for, as cursors, in the order they appear in the
/// container.
///
/// Video always. Audio only when it is a codec the engine can decode - a track handed over
/// and then never decoded is worse than one that was never offered, because the title's own
/// demux thread will queue its units and wait for a decoder that produces nothing.
/// `VITASLOP_MP4_AUDIO=0` withholds audio, which is the A/B arm for any title whose demux
/// behaves differently once a second stream appears.
fn served_cursors(mp4: &crate::mp4::Mp4) -> Vec<(usize, usize)> {
    let audio = !matches!(crate::knobs::var("VITASLOP_MP4_AUDIO").as_deref(), Ok("0"));
    mp4.tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| match t.kind {
            crate::mp4::TrackKind::Video => true,
            crate::mp4::TrackKind::Audio => audio && &t.codec == b"mp4a",
            crate::mp4::TrackKind::Other => false,
        })
        .map(|(i, _)| (i, 0))
        .collect()
}

/// How many of the served cursors are VIDEO tracks.
///
/// This is what `sceMp4StartFileStreaming`'s first out-parameter reports - see the call site
/// for the retail measurement that fixed the reading. Never zero: a caller that reads zero
/// streams has been told the movie cannot be played at all.
fn served_video_streams(mp4: &crate::mp4::Mp4, cursors: &[(usize, usize)]) -> u32 {
    cursors
        .iter()
        .filter(|(t, _)| mp4.tracks.get(*t).is_some_and(|t| t.kind == crate::mp4::TrackKind::Video))
        .count()
        .max(1) as u32
}

/// Describe the next access unit WITHOUT consuming it.
fn peek_access_unit(st: &mut crate::host::VitaState) -> Result<Option<AccessUnit>, String> {
    let Some((track, at)) = st.movie.as_ref().and_then(next_unit_track) else {
        return Ok(None);
    };
    access_unit_at(st, track, at)
}

/// Take the next access unit out of the container, and move that track's cursor on.
///
/// The parameter sets go in front of every SYNC sample of a video track rather than only
/// the first. That costs a few dozen bytes a keyframe and is what makes the stream decodable
/// from any of them - which is what a seek, and a decoder reset, both need.
fn next_access_unit(st: &mut crate::host::VitaState) -> Result<Option<AccessUnit>, String> {
    let Some((track, at)) = st.movie.as_ref().and_then(next_unit_track) else {
        return Ok(None);
    };
    let unit = access_unit_at(st, track, at)?;
    if unit.is_some() {
        if let Some(movie) = st.movie.as_mut() {
            // The gate opened, so the refusal run ends here - see `movie_unit_wait_us`.
            movie.gate_refusals = 0;
            if let Some(c) = movie.cursors.iter_mut().find(|(t, _)| *t == track) {
                c.1 += 1;
            }
            // Kept in step for the video track, because everything that reports progress
            // still counts video samples.
            if movie.mp4.tracks[track].kind == crate::mp4::TrackKind::Video {
                movie.next_sample = at + 1;
            }
        }
    }
    Ok(unit)
}

/// Build the access unit for one sample of one track.
fn access_unit_at(
    st: &mut crate::host::VitaState,
    track: usize,
    index: usize,
) -> Result<Option<AccessUnit>, String> {
    let movie = st.movie.as_mut().ok_or("no movie session")?;
    let t = movie.mp4.tracks.get(track).ok_or("no such track")?;
    let Some(sample) = t.samples.get(index) else {
        return Ok(None);
    };
    let (offset, size) = (sample.offset, sample.size as usize);
    let (width, height, sync) = (t.width, t.height, sample.sync);
    let (pts, dts) = (sample.pts, sample.dts);
    // The track's position in the container, which is how the title numbers streams.
    let stream = track as u32;
    let video = t.kind == crate::mp4::TrackKind::Video;
    let fd = movie.fd;
    // Only a video sample needs rewriting into a self-describing elementary stream; an AAC
    // sample IS the elementary stream the decoder is given.
    let sets = if video && sync { movie.avcc.annex_b_parameter_sets() } else { Vec::new() };
    // One read per access unit - tens of kilobytes - which is what the device's own
    // streaming demuxer does.
    let raw = read_at(st, fd, offset, size);
    if raw.len() != size {
        return Err(format!(
            "sample {index} of track {stream} is {size} bytes at {offset} but only {} could \
             be read",
            raw.len()
        ));
    }
    let movie = st.movie.as_mut().ok_or("no movie session")?;
    let bytes = if video {
        let mut bytes = Vec::with_capacity(size + sets.len());
        bytes.extend_from_slice(&sets);
        movie
            .avcc
            .sample_to_annex_b(&raw, &mut bytes)
            .map_err(|e| format!("sample {index} is not framed as the avcC says ({e:?})"))?;
        bytes
    } else {
        raw
    };
    Ok(Some(AccessUnit { bytes, width, height, pts, dts, sync, stream, video }))
}

/// Which reading of the unit struct to hand back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnitMode {
    /// Hand units over, which is what plays the movie.
    Normal,
    /// Never hand back a unit. The arm that separates "the playback path itself faults"
    /// from "our inferred fields fault", one run each.
    NoUnits,
}

/// `VITASLOP_MP4_UNITS=none`: never return an access unit. A diagnostic, not a mode -
/// with it set the movie cannot play at all.
fn unit_mode() -> UnitMode {
    match std::env::var("VITASLOP_MP4_UNITS").ok().as_deref() {
        Some("none") => UnitMode::NoUnits,
        _ => UnitMode::Normal,
    }
}

/// Say, once, exactly which parts of the unit struct are inferred.
fn report_unit_assumptions(st: &mut crate::host::VitaState, width: u32, height: u32, size: u32) {
    if st.reported_unit_layout {
        return;
    }
    st.reported_unit_layout = true;
    tracing::info!(
        target: "vitaslop::movie",
        width, height, first_unit_bytes = size,
        "SceMp4: handing the title Annex B access units. Status, type and the          width/height pair are established from the caller's own code; +0x20 as the          elementary stream length and +0x30 as a timestamp are the most probable          reading, and +0x10/+0x14/+0x18/+0x1c are left zero because their roles are          unknown."
    );
}

/// Field offsets in the unit-information struct `sceMp4GetNextUnit` fills.
///
/// RECOVERED from the one caller, which is the title's own demux thread: it passes a
/// scratch area and then copies six fields out of it into a 0x30-byte unit record it hands
/// to its decoder. Those six copies are what fix the offsets; the ROLES follow from what
/// the decoder call site then does with the record - `es.size` comes from the record's
/// `+0x04`, which is this struct's `+0x08`, so that field is the unit's LENGTH and the
/// rest read off around it.
mod unit_info {
    /// Length of the access unit in bytes.
    pub const SIZE: u32 = 0x08;
    /// Presentation timestamp, 64-bit, in the track's own timescale.
    pub const PTS: u32 = 0x10;
    /// Decode timestamp, 64-bit.
    pub const DTS: u32 = 0x18;
    /// Copied as `> 0` into a byte of the caller's record - a sync/keyframe flag.
    pub const SYNC: u32 = 0x28;
    /// Copied as a byte. Taken as the stream id, which is the one the title enabled.
    pub const STREAM_ID: u32 = 0x2c;
    /// Copied as a byte. Role unknown; left zero.
    pub const BYTE_B: u32 = 0x30;
}

/// int sceMp4GetNextUnit(handle, info)
///
/// Describe the unit at the stream cursor without consuming it; `sceMp4GetNextUnitData`
/// is what copies the bytes and moves on.
///
/// **ZERO is the success return and a positive value is END OF STREAM**, which is the
/// opposite of the usual shape and is established from the caller: on zero it looks the
/// unit's stream id up against its own stream table and queues the record, and on a
/// POSITIVE value it logs "Looping back to start of the file" and calls `sceMp4Reset`.
/// Returning 1 for "here is a unit" made the demux thread rewind after every frame.
/// >>> AND IT PARKS BRIEFLY WHEN IT SAYS "NOT YET", BECAUSE THE CALLER SPINS ON IT.
///
/// The title's demux thread does not sleep between asks - it re-asks immediately. MEASURED
/// when the pacing gate first landed with a plain return: **148,000 host calls in a single
/// fast-forward frame**, all of them this call being refused and asked again. That is a spin,
/// and a spin inside a title's own thread is guest CPU on every device that runs it.
///
/// So a refusal parks the caller for the shorter of "until the unit is due" and
/// [`UNIT_WAIT_SLICE_US`]. Unlike the park inside `sceMp4GetNextUnitData` - which crashed the
/// run, see [`movie_unit_wait_us`] - the thread holds NOTHING here: it has not been promised a
/// unit, and if the movie is torn down while it sleeps it wakes, asks again and is told there
/// is no session. Short slices rather than one long sleep for the same reason: the less time a
/// guest thread spends inside a host call, the less of its owner's shutdown it can miss.
pub(super) fn mp4_get_next_unit_info(
    ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
) -> crate::SvcOutcome {
    let (handle, info) = (ctx.arg(0) as i32, ctx.arg(1));
    let wait = movie_unit_wait_us(st, handle);
    let got = do_get_next_unit_info(ctx, st, handle, info);
    ctx.ret(got as u32);
    match wait {
        Some(us) if st.is_preemptive() => {
            st.sleep_park(us.min(UNIT_WAIT_SLICE_US));
            crate::SvcOutcome::Block
        }
        _ => crate::SvcOutcome::Continue,
    }
}

/// The longest a demux thread is parked for one "not yet". Two milliseconds: short enough that
/// a movie being torn down is never waited through, long enough that a 33 ms frame gap costs a
/// dozen wake-ups rather than thousands of spins.
const UNIT_WAIT_SLICE_US: u64 = 2_000;

fn do_get_next_unit_info(
    ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    handle: i32,
    info: u32,
) -> i32 {
    if info == 0 {
        return SCE_ERROR_ERRNO_ENOENT;
    }
    match st.movie.as_ref() {
        Some(movie) if movie.handle == handle => {}
        _ => return SCE_ERROR_ERRNO_ENOENT,
    }
    let unit = match peek_access_unit(st) {
        Ok(Some(unit)) => unit,
        // End of stream - see above: POSITIVE, not zero.
        Ok(None) => return 1,
        Err(reason) => {
            let path = st.movie.as_ref().map(|m| m.path.clone()).unwrap_or_default();
            report_no_video(st, &path, &reason);
            return -1;
        }
    };
    // >>> A UNIT IS NOT OFFERED BEFORE ITS OWN PRESENTATION TIME. See `movie_unit_wait_us` for
    // >>> the rate this fixes and for the two placements that were measured and rejected.
    //
    // THIS is the call the title's demux loop asks with: MEASURED over one front-end run,
    // `sceMp4GetNextUnitInfo` 175 times against `sceMp4GetNextUnit` **twice**. A size of zero
    // with a SUCCESS return is "nothing this time" - the return code is deliberately left at 0,
    // because a POSITIVE return here is what makes this title log "Looping back to start of the
    // file" and call `sceMp4Reset`, i.e. restart the movie rather than wait a moment for it.
    if movie_unit_wait_us(st, handle).is_some() {
        ctx.write_u32(info + unit_info::SIZE, 0);
        return 0;
    }
    ctx.write_u32(info + unit_info::SIZE, unit.bytes.len() as u32);
    ctx.write_u32(info + unit_info::PTS, unit.pts as u32);
    ctx.write_u32(info + unit_info::PTS + 4, (unit.pts >> 32) as u32);
    ctx.write_u32(info + unit_info::DTS, unit.dts as u32);
    ctx.write_u32(info + unit_info::DTS + 4, (unit.dts >> 32) as u32);
    ctx.write_u32(info + unit_info::SYNC, u32::from(unit.sync));
    // >>> THE STREAM THE UNIT BELONGS TO, and this used to be a hard zero.
    //
    // The title's demux thread looks this up in its own stream table and queues the record
    // on that stream's decoder - so with every unit claiming stream 0, a movie could only
    // ever have one stream, and that is what left every movie silent.
    ctx.write_u32(info + unit_info::STREAM_ID, unit.stream);
    ctx.write_u32(info + unit_info::BYTE_B, 0);
    0
}

/// int sceMp4GetNextUnitData(handle, dest)
///
/// Copy the unit at the cursor into the caller's buffer and MOVE ON - this is the call
/// that consumes the stream. RECOVERED from the caller: it takes the size out of the
/// record `sceMp4GetNextUnit` filled, allocates exactly that many bytes, stores the block
/// as the record's data pointer, and passes it here; the record then travels to
/// `sceAvcdecDecode`, whose `es.pBuf`/`es.size` are those same two fields.
/// `sceMp4GetNextUnitData`, as the SCHEDULER sees it.
///
/// # Why this call is PACED
///
/// On hardware this reads the container off storage, and the demux thread is descheduled
/// for the transfer. Here the file is already open and a sample is a memory read, so the
/// call returned instantly - and a demuxer that never waits sprints ahead of the decoder
/// that consumes it. MEASURED on a device: the title's 512 KB unit pool filled, every
/// subsequent allocation returned NULL, and the thread spun retrying - thousands of times,
/// enough to bury the diagnostics panel and to show up as guest CPU.
///
/// So the read is charged the same modelled storage cost every other guest read is, through
/// the same accumulator, and parks the caller when the debt is worth a context switch. The
/// pool then drains at roughly the rate the device would fill it.
pub(super) fn mp4_get_next_unit_data(
    ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
) -> crate::SvcOutcome {
    let (handle, dest) = (ctx.arg(0) as i32, ctx.arg(1));
    // >>> A UNIT IS NOT READABLE BEFORE ITS OWN PRESENTATION TIME.
    //
    // Storage pacing alone does not bound a MOVIE, because storage is faster than any movie's
    // bitrate on every device this runs on. MEASURED on this title's front end: it pulls
    // exactly one unit per display flip, submits it, and shows the picture that comes back -
    // so the demuxer's delivery rate IS the playback rate, and at 60 flips a second a 29.97 fps
    // movie played at **2.0x**. 210 flips consumed 7.0 seconds of movie in 3.5 seconds of guest
    // time, on this desktop, reproducibly.
    //
    // The title is not doing anything unusual: a player that trusts the demuxer to hand it
    // units at the stream's rate is an ordinary shape, and on hardware SceMp4 is reading a
    // stream off storage into a ring whose drain the title paces against. What it cannot do is
    // run ahead of the movie's own clock, and here it could.
    //
    // So the unit's PTS is a floor on when it may be handed over, with `READ_AHEAD_US` of lead
    // so the decoder's pipeline still fills and a hitch does not immediately starve it. Parking
    // rather than refusing: `sceMp4GetNextUnitInfo` reports "no unit" with a POSITIVE return,
    // which this title reads as end of stream and answers with `sceMp4Reset` - so a demuxer
    // that says "not yet" makes the movie restart. A thread that waits is what the hardware
    // does anyway.
    // The pacing gate is on `sceMp4GetNextUnit`, NOT here - see [`movie_unit_wait_us`] and the
    // two failed placements recorded there. By the time the title reaches this call it has
    // already sized a buffer for a unit it was promised, and refusing at that point makes it
    // submit whatever was in that buffer.
    let got = do_get_next_unit_data(ctx, st, handle, dest);
    ctx.ret(got as u32);
    if got <= 0 {
        return crate::SvcOutcome::Continue;
    }
    super::iofilemgr::charge_read(st, got as usize)
}

/// How long the caller must wait before the unit at the cursor is due, or `None` if it is due
/// now (or there is no movie, no clock origin, or nothing left to hand over).
///
/// >>> WHERE THIS GATE GOES, AND THE TWO PLACES IT MUST NOT.
///
/// The title's demux loop is: ask `sceMp4GetNextUnit` for a descriptor, allocate exactly the
/// size it reports, call `sceMp4GetNextUnitData` for the bytes, submit to `sceAvcdecDecode`.
/// Both other points were tried and measured:
///
/// - **Parking the caller inside `GetNextUnitData`** paces the movie exactly and CRASHES. At
///   the frame this title leaves its front end it calls `sceAvcdecDeleteDecoder` with the movie
///   120 units into 1241 - it tears the player down mid-playback, which is what "skip the
///   intro" is - and the demux thread was asleep inside the library at the time. It woke
///   holding a unit for a session that no longer existed: `memory access out of bounds`.
/// - **Returning zero bytes from `GetNextUnitData`** does not crash and does not work: the
///   title has already committed to a unit by then, so it submits the buffer it allocated
///   whatever comes back. MEASURED: 8.33 access units a frame, every decode call empty, no
///   picture ever delivered.
///
/// So the answer has to come before the title commits, which is `sceMp4GetNextUnit` reporting
/// no unit this time - the same answer it already gives at end of stream, and the one the
/// caller takes silently. Not `sceMp4GetNextUnitInfo`: a POSITIVE return there is what makes
/// this title log "Looping back to start of the file" and call `sceMp4Reset`.
///
/// The origin is stamped on the first unit handed over rather than at open: a title that opens
/// a movie and then spends two seconds loading before it starts pulling would otherwise find
/// the first two seconds of the stream all due at once, which is exactly the burst this exists
/// to prevent.
fn movie_unit_wait_us(st: &mut crate::host::VitaState, handle: i32) -> Option<u64> {
    if !st.is_preemptive() {
        // A run-to-completion host has no scheduler to park against - see `read_blocking`.
        return None;
    }
    let now = st.guest_mono_us();
    let movie = st.movie.as_mut()?;
    if movie.handle != handle {
        return None;
    }
    // The SAMPLE TABLE, not `peek_access_unit`. The question here is "what time is this unit
    // for", and peeking answers it by reading the sample off storage and rewriting it into an
    // elementary stream - tens of kilobytes and an allocation, thrown away, on a call the title
    // makes 175 times a run and which usually says "not yet". The table has the timestamp
    // already.
    let (track_index, at) = next_unit_track(movie)?;
    let track = movie.mp4.tracks.get(track_index)?;
    if track.timescale == 0 {
        return None;
    }
    let pts = track.samples.get(at)?.pts;
    let pts_us = pts.saturating_mul(1_000_000) / track.timescale as u64;
    let origin = *movie.timeline_origin_us.get_or_insert(now.saturating_sub(pts_us));
    let due = origin.saturating_add(pts_us).saturating_sub(READ_AHEAD_US);
    if due <= now {
        return None;
    }
    // >>> A GATE THAT NEVER OPENS IS A STOPPED MOVIE, AND IT USED TO BE SILENT.
    //
    // Count consecutive refusals; `next_access_unit` clears this the moment a unit is taken.
    // The threshold is thousands rather than tens because the caller SPINS - the title's demux
    // thread re-asks immediately - so a handful of refusals is one frame of ordinary pacing.
    movie.gate_refusals += 1;
    let (n, said) = (movie.gate_refusals, movie.reported_gate_stall);
    if n >= GATE_STALL_REFUSALS && !said {
        movie.reported_gate_stall = true;
        let (origin, due, pts) = (origin, due, pts_us);
        tracing::warn!(
            target: "vitaslop::movie",
            refusals = n,
            due_us = due, now_us = now, origin_us = origin, pts_us = pts,
            ahead_us = due.saturating_sub(now),
            "SceMp4: the demuxer has been told NOT YET {n} times in a row without taking a              single unit, so this movie is not advancing. The unit at the cursor is due at              `origin + pts`, and that moment is still ahead of the guest clock - which means              either the timeline origin is wrong or the clock this gate reads is not the one              the title's own player runs on."
        );
    }
    Some(due - now)
}

/// Consecutive "not yet" answers that mean the movie has STOPPED rather than that it is being
/// paced. See [`movie_unit_wait_us`]: the caller re-asks immediately, so this is a couple of
/// seconds of a title spinning, not a few frames of ordinary rate limiting.
const GATE_STALL_REFUSALS: u64 = 2_000;

/// How far ahead of the movie's own clock the demuxer may run.
///
/// Two frames of a 30 fps movie. It has to be more than zero - a decoder pipeline is several
/// frames deep and a title that submits a unit only when it is exactly due would never fill
/// one - and it has to be small enough that "ahead" cannot become "the whole file", which is
/// the state this pacing exists to prevent.
const READ_AHEAD_US: u64 = 66_667;

fn do_get_next_unit_data(
    ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    handle: i32,
    dest: u32,
) -> i32 {
    if dest == 0 {
        return SCE_ERROR_ERRNO_ENOENT;
    }
    match st.movie.as_ref() {
        Some(movie) if movie.handle == handle => {}
        _ => return SCE_ERROR_ERRNO_ENOENT,
    }
    let unit = match next_access_unit(st) {
        Ok(Some(unit)) => unit,
        Ok(None) => return 0,
        Err(reason) => {
            let path = st.movie.as_ref().map(|m| m.path.clone()).unwrap_or_default();
            report_no_video(st, &path, &reason);
            return -1;
        }
    };
    ctx.write_bytes(dest, &unit.bytes);
    // An AUDIO unit is decoded the moment it is handed over, so its PCM is waiting when the
    // title's own decode call comes - see [`MovieAudio`].
    if !unit.video {
        let pts = unit.pts as i64;
        if let Some(movie) = st.movie.as_mut() {
            pump_movie_audio(movie, &unit.bytes, pts);
        }
    }
    if let Some(movie) = st.movie.as_mut() {
        movie.delivered += 1;
        if movie.delivered <= 4 || movie.delivered % 200 == 0 {
            tracing::debug!(
                target: "vitaslop::movie",
                units = movie.delivered, of = movie.mp4.track(crate::mp4::TrackKind::Video)
                    .map(|t| t.samples.len()).unwrap_or(0),
                bytes = unit.bytes.len(),
                "sceMp4GetNextUnitData"
            );
        }
    }
    unit.bytes.len() as i32
}

/// int <unnamed SceMp4 0x40351e1a>(handle)
///
/// Read as `sceMp4Reset`: the demux thread logs "Looping back to start of the file",
/// calls this, and immediately asks for the next unit again. Rewinding the stream cursor
/// is the only behaviour that makes that sequence produce a movie that loops.
#[hostcall]
pub(super) fn mp4_reset(
    _ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    handle: i32,
) -> i32 {
    do_reset(st, handle)
}

fn do_reset(st: &mut crate::host::VitaState, handle: i32) -> i32 {
    let Some(movie) = st.movie.as_mut() else {
        return SCE_ERROR_ERRNO_ENOENT;
    };
    if handle != movie.handle {
        return SCE_ERROR_ERRNO_ENOENT;
    }
    movie.next_sample = 0;
    for c in movie.cursors.iter_mut() {
        c.1 = 0;
    }
    // The stream restarts, so its timeline does too - see [`movie_unit_wait_us`]. Leaving the
    // old origin in place would make every unit of the second pass due the moment it is asked
    // for, i.e. the movie loops back and then plays at whatever rate the title can pull.
    movie.timeline_origin_us = None;
    if let Some(audio) = movie.audio.as_mut() {
        audio.ready.clear();
        audio.pending.clear();
        let _ = audio.decoder.reset();
    }
    tracing::debug!(target: "vitaslop::movie", "sceMp4Reset: back to the first unit");
    0
}

/// int <unnamed SceMp4 0x609e57ad>(handle, stream, enable)
///
/// Read as `sceMp4EnableStream`: the 0.945 name list carries that name with no 3.60 NID
/// beside it, and the two call sites are one-line thunks that differ ONLY in a trailing
/// `1` and `0` - which is what an enable/disable pair looks like and very little else does.
///
/// The demuxer here produces one stream's units at a time regardless, so enabling is
/// recorded and reported rather than acted on; a title that disabled the video stream and
/// still got pictures would be a defect, so the state is kept in order to say so.
#[hostcall]
pub(super) fn mp4_enable_stream(
    _ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    handle: i32,
    stream: u32,
    enable: u32,
) -> i32 {
    do_enable_stream(st, handle, stream, enable)
}

fn do_enable_stream(st: &mut crate::host::VitaState, handle: i32, stream: u32, enable: u32) -> i32 {
    let Some(movie) = st.movie.as_mut() else {
        return SCE_ERROR_ERRNO_ENOENT;
    };
    if handle != movie.handle {
        return SCE_ERROR_ERRNO_ENOENT;
    }
    if enable != 0 {
        movie.enabled_streams.insert(stream);
    } else {
        movie.enabled_streams.remove(&stream);
    }
    tracing::debug!(
        target: "vitaslop::movie", stream, enable,
        enabled = ?movie.enabled_streams,
        "sceMp4EnableStream"
    );
    0
}

/// int sceMp4CloseFile(handle)
///
/// Drops the session and CLOSES its descriptor. That matters now the movie is streamed
/// rather than held: the session owns an open file for as long as it lives, and a title
/// that plays one movie after another would otherwise leak one per movie.
///
/// Succeeds even for a handle that names no session: a close of something that does not
/// exist is the one call here that can honestly report success, and failing it would send
/// the title into a teardown error path over a resource this engine never created.
#[hostcall]
pub(super) fn mp4_close_file(
    _ctx: &mut crate::host::GuestCtx,
    st: &mut crate::host::VitaState,
    handle: i32,
) -> i32 {
    do_close_file(st, handle)
}

fn do_close_file(st: &mut crate::host::VitaState, handle: i32) -> i32 {
    let Some(movie) = st.movie.as_ref() else { return 0 };
    if movie.handle != handle {
        return 0;
    }
    let (fd, path, delivered) = (movie.fd, movie.path.clone(), movie.delivered);
    // >>> A MOVIE WITH SOUND THE TITLE NEVER ASKED FOR PLAYED SILENT, AND THAT IS SAID OUT
    // >>> LOUD RATHER THAN LEFT AS A QUIET PICTURE.
    //
    // The engine opens a host AAC decoder for any `mp4a` track (see `open_movie_audio`), but a
    // unit only reaches the title through a stream the title ENABLED. This one enables its
    // video stream and nothing else, so the sound is decoded by nobody and the movie is mute.
    // That is a real gap against the console, not a decision - and a run that does not report
    // it looks exactly like a run whose movie had no sound to begin with.
    let audio_track = movie.mp4.track(crate::mp4::TrackKind::Audio).is_some();
    let audio_enabled = movie
        .mp4
        .tracks
        .iter()
        .enumerate()
        .any(|(i, t)| t.kind == crate::mp4::TrackKind::Audio && movie.enabled_streams.contains(&(i as u32)));
    if audio_track && !audio_enabled {
        tracing::warn!(
            target: "vitaslop::movie", %path,
            "SceMp4: this movie has a sound track and the title never enabled that stream, so it \
             played SILENT. The engine decodes the audio only for a stream the title asks for, \
             and how this title's player is meant to reach a movie's sound is not established - \
             it has never called sceAudiodecDecode on any run."
        );
    }
    st.movie = None;
    st.io_close(fd);
    tracing::debug!(
        target: "vitaslop::movie", %path, units = delivered,
        "sceMp4CloseFile: the movie is closed"
    );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp4::{Mp4, Sample, Track, TrackKind};

    fn track(id: u32, kind: TrackKind, codec: &[u8; 4], timescale: u32, dts: &[u64]) -> Track {
        Track {
            id,
            kind,
            codec: *codec,
            timescale,
            duration: 0,
            width: 0,
            height: 0,
            codec_config: Vec::new(),
            samples: dts
                .iter()
                .map(|&d| Sample { offset: 0, size: 1, pts: d, dts: d, duration: 1, sync: true })
                .collect(),
        }
    }

    /// >>> THE UNITS COME OUT IN DECODE ORDER ACROSS THE TRACKS, in MICROSECONDS.
    ///
    /// The two tracks of a movie count time in different units - video in frames, audio in
    /// samples - so their raw timestamps are not comparable and a demuxer that compares them
    /// anyway hands over one track's units first and the other's at the end. This is one
    /// video track at 60 ticks/s and one audio track at 48000, interleaved.
    #[test]
    fn units_are_interleaved_by_decode_time_across_tracks() {
        let mp4 = Mp4 {
            timescale: 1000,
            duration: 0,
            tracks: vec![
                // 0, 1/60 s, 2/60 s
                track(1, TrackKind::Video, b"avc1", 60, &[0, 1, 2]),
                // 0, 1024/48000 s (~21 ms), 2048/48000 s (~43 ms)
                track(2, TrackKind::Audio, b"mp4a", 48_000, &[0, 1024, 2048]),
            ],
        };
        let mut cursors = super::served_cursors(&mp4);
        assert_eq!(cursors.len(), 2, "both tracks are served");
        let mut order = Vec::new();
        // The same choice `next_unit_track` makes, driven by hand so the test needs no
        // session, no file and no guest.
        for _ in 0..6 {
            let mut best: Option<(usize, u64)> = None;
            for (i, &(t, at)) in cursors.iter().enumerate() {
                let tr = &mp4.tracks[t];
                let Some(sample) = tr.samples.get(at) else { continue };
                let us = sample.dts * 1_000_000 / tr.timescale as u64;
                if best.is_none_or(|(_, b)| us < b) {
                    best = Some((i, us));
                }
            }
            let (i, _) = best.expect("a track still has units");
            order.push(mp4.tracks[cursors[i].0].id);
            cursors[i].1 += 1;
        }
        // video(0ms) audio(0ms) video(16.6) audio(21.3) video(33.3) audio(42.6)
        assert_eq!(order, vec![1, 2, 1, 2, 1, 2]);
    }

    /// >>> THE STREAM COUNT `sceMp4StartFileStreaming` REPORTS IS THE VIDEO COUNT.
    ///
    /// The retail title reads that word and creates one H.264 decoder per count, so a movie
    /// whose audio track was counted too had its AAC track handed to a video decoder - and its
    /// player then stalled seven access units into 2,555. See `do_start_file_streaming`.
    #[test]
    fn the_reported_stream_count_counts_only_video_tracks() {
        let mp4 = Mp4 {
            timescale: 1000,
            duration: 0,
            tracks: vec![
                track(1, TrackKind::Video, b"avc1", 60, &[0]),
                track(2, TrackKind::Audio, b"mp4a", 48_000, &[0]),
            ],
        };
        let cursors = super::served_cursors(&mp4);
        assert_eq!(cursors.len(), 2, "both tracks are still SERVED - only the count differs");
        assert_eq!(super::served_video_streams(&mp4, &cursors), 1);
    }

    /// A track whose codec this engine does not decode is not offered at all: the title's
    /// demux thread would queue its units against a decoder that never answers.
    #[test]
    fn only_decodable_tracks_are_served() {
        let mp4 = Mp4 {
            timescale: 1000,
            duration: 0,
            tracks: vec![
                track(1, TrackKind::Video, b"avc1", 60, &[0]),
                track(2, TrackKind::Audio, b"ac-3", 48_000, &[0]),
                track(3, TrackKind::Other, b"tx3g", 1000, &[0]),
            ],
        };
        let served = super::served_cursors(&mp4);
        assert_eq!(served.len(), 1);
        assert_eq!(mp4.tracks[served[0].0].id, 1);
    }
}

#[cfg(test)]
mod movie_audio_tests {
    //! >>> THE GUEST'S OWN PATH TO A MOVIE'S SOUND, DRIVEN WITHOUT THE GAME.
    //!
    //! Everything between the container and `pPcm` is exercised here: the demuxer's
    //! per-track cursors, the stream id each unit carries, the read-ahead that decodes an
    //! audio unit when it is HANDED OVER, and `sceAudiodecDecode` serving the queue into
    //! guest memory. The alternative was to reach a movie with sound by driving a title's
    //! menus, which is thousands of frames of replay and answers this question no better.
    //!
    //! Env-gated on a retail movie, which cannot be committed:
    //! `VITASLOP_MOVIE=<a .mp4 with an AAC track> cargo test -p vitaslop-runtime --release
    //! -- --nocapture movie_audio`

    use crate::host::{GuestCtx, SliceMemory, VitaState};
    use vitaslop_transpiler::abi::REG_COUNT;

    /// Guest memory for the test: one flat region with the movie's buffers in it.
    const BASE: u32 = 0x8000_0000;
    const MEM: usize = 8 << 20;

    /// Where the test puts things in that region.
    const PATH_AT: u32 = BASE + 0x1000;
    const UNIT_BUF: u32 = BASE + 0x2000;
    const INFO_AT: u32 = BASE + 0x1_0000;
    const CTRL_AT: u32 = BASE + 0x1_1000;
    const AAC_INFO_AT: u32 = BASE + 0x1_1100;
    // >>> A MEGABYTE APART, because a VIDEO access unit is not small: at 0x20000 the unit
    // buffer ran into the control block and the decode then read a clobbered handle. The
    // test's own memory map is the sort of thing that fails as "the engine is broken".
    const ES_AT: u32 = BASE + 0x10_0000;
    const PCM_AT: u32 = BASE + 0x30_0000;

    #[test]
    fn a_movies_audio_reaches_guest_memory_through_the_guests_own_calls() {
        let Ok(movie) = std::env::var("VITASLOP_MOVIE") else {
            eprintln!("VITASLOP_MOVIE is not set; skipping the guest movie-audio path");
            return;
        };
        let bytes = std::fs::read(&movie).expect("the movie file");

        let mut st = VitaState::new(BASE, MEM as u32, Box::new(crate::world::DeterministicWorld::default()));
        st.audio_dec = Box::new(vitaslop_platform::audio_dec::AacFactory);
        st.add_file("app0:movie.mp4", bytes);
        let mut mem = vec![0u8; MEM];
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; crate::host::VFP_ARG_COUNT];
        let mut slice = SliceMemory(&mut mem);
        let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut slice, BASE);

        // The path the guest would have passed, in guest memory.
        let path = b"app0:movie.mp4\0";
        ctx.write_bytes(PATH_AT, path);

        let handle = super::do_open_file(&mut ctx, &mut st, PATH_AT, UNIT_BUF, 0x40000);
        assert!(handle != 0, "the movie opens");
        assert!(
            st.movie.as_ref().is_some_and(|m| m.audio.is_some()),
            "the movie has an audio track this host can decode - without one this test \
             proves nothing, so it fails rather than passing vacuously"
        );

        // The title's own decoder, created the way it does: an info block describing the
        // stream, then `sceAudiodecCreateDecoderExternal`.
        ctx.write_u32(AAC_INFO_AT + 0x00, 0x14);
        ctx.write_u32(AAC_INFO_AT + 0x08, 2);
        ctx.write_u32(AAC_INFO_AT + 0x0c, 48_000);
        ctx.write_u32(CTRL_AT + 0x24, AAC_INFO_AT);
        ctx.write_u32(CTRL_AT + 0x10, 1536);
        ctx.write_u32(CTRL_AT + 0x1c, 8192);
        let size = crate::vita::audiodec::do_get_context_size(&mut st, 0x1005);
        assert!(size > 0, "a context size of zero is how the title decides not to decode");
        let rc = crate::vita::audiodec::do_create_decoder_external(
            &mut ctx,
            &mut st,
            crate::host::Ptr(CTRL_AT),
            0x1005,
            BASE + 0x50_0000,
        );
        assert_eq!(rc, 0, "the decoder is created");

        // Pull units the way the title's demux thread does, and decode the audio ones the
        // way its audio thread does.
        let mut audio_units = 0usize;
        let mut video_units = 0usize;
        let mut pcm_frames = 0usize;
        let mut loudest = 0i16;
        let mut streams = std::collections::BTreeSet::new();
        for _ in 0..400 {
            if super::do_get_next_unit_info(&mut ctx, &mut st, handle, INFO_AT) != 0 {
                break;
            }
            let size = ctx.read_u32(INFO_AT + 0x08);
            let stream = ctx.read_u32(INFO_AT + 0x2c);
            streams.insert(stream);
            let video = st
                .movie
                .as_ref()
                .and_then(|m| m.mp4.tracks.get(stream as usize))
                .is_some_and(|t| t.kind == crate::mp4::TrackKind::Video);
            let got = super::do_get_next_unit_data(&mut ctx, &mut st, handle, ES_AT);
            assert_eq!(got, size as i32, "the unit's bytes are the size it described");
            if video {
                video_units += 1;
                continue;
            }
            audio_units += 1;
            // `sceAudiodecDecode` as the title makes it: `pEs` and `pPcm` set, nothing else.
            ctx.write_u32(CTRL_AT + 0x08, ES_AT);
            ctx.write_u32(CTRL_AT + 0x14, PCM_AT);
            let rc = crate::vita::audiodec::do_decode(&mut ctx, &mut st, crate::host::Ptr(CTRL_AT));
            assert_eq!(rc, 0, "the decode succeeds");
            let out = ctx.read_u32(CTRL_AT + 0x18);
            if out == 0 {
                continue;
            }
            pcm_frames += 1;
            let pcm = ctx.read_bytes(PCM_AT, out as usize);
            for s in pcm.chunks_exact(2) {
                let v = i16::from_le_bytes([s[0], s[1]]);
                loudest = loudest.max(v.saturating_abs());
            }
        }

        eprintln!(
            "streams {streams:?}: {video_units} video units, {audio_units} audio units, \
             {pcm_frames} PCM frames, loudest sample {loudest}"
        );
        // The ids are the title's numbering: 0-based INDICES into the container's streams.
        assert!(streams.len() >= 2, "the units carry TWO stream ids, not one - {streams:?}");
        assert!(streams.contains(&0), "the first stream is 0, not a track_ID - {streams:?}");
        assert!(video_units > 0 && audio_units > 0, "both tracks are served");
        assert!(pcm_frames > 10, "the guest's decode calls produced PCM");
        // >>> AND IT IS SOUND, NOT SILENCE. A frame count cannot tell those apart
        // ([[vitaslop-count-frames-cannot-tell-silence-from-music]]), and every failure this
        // path can have - the wrong stream, the wrong config, a queue serving nothing -
        // produces silence rather than an error.
        assert!(loudest > 1000, "the PCM written into guest memory is silent (peak {loudest})");
    }
}
