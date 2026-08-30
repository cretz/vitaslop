//! An ISO base media file format (MP4) demuxer.
//!
//! A title's full-motion video is an MP4 the guest asks `SceMp4` to demux for it: it
//! wants stream properties (codec, size, timescale, duration) and then a stream of
//! ACCESS UNITS with timestamps, which it feeds to the video decoder itself. That makes
//! the demuxer a self-contained, testable component with no Vita in it at all - which is
//! why it lives here rather than in `vita/`.
//!
//! The format is ISO/IEC 14496-12: a tree of length-tagged boxes, big-endian. What a
//! demuxer needs is all in one subtree per track:
//!
//! ```text
//! moov
//!   trak
//!     tkhd                 track id, and for video the display width/height (16.16)
//!     mdia
//!       mdhd               timescale, duration
//!       hdlr               'vide' / 'soun'
//!       minf/stbl
//!         stsd             sample description: the codec, and its setup data
//!         stts             sample -> duration       (run-length coded)
//!         ctts             sample -> pts offset     (optional, B-frames)
//!         stsc             sample -> chunk mapping  (run-length coded)
//!         stsz             sample sizes
//!         stco / co64      chunk file offsets
//!         stss             sync (keyframe) sample numbers (absent = all sync)
//! ```
//!
//! The sample tables are read once into a flat [`Sample`] list per track, because every
//! consumer wants random access by index or by time and the run-length forms make that
//! awkward. A movie is tens of thousands of samples, so the flat list is under a
//! megabyte - cheaper than re-walking the tables on every request.
//!
//! Parsing is total: every read is bounds-checked and a malformed file is an [`Mp4Error`],
//! never a panic. Bytes come from the guest's own filesystem, so "malformed" includes
//! "we mis-parsed it", and either way the caller must be able to report rather than crash.

use std::fmt;

/// Why an MP4 could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mp4Error {
    /// A box header or payload ran past the end of its parent.
    Truncated(&'static str),
    /// A box declared a size smaller than its own header.
    BadBoxSize { kind: [u8; 4], size: u64 },
    /// No `moov`, or no track inside it.
    NoTracks,
    /// A track was missing a box a demuxer cannot do without.
    MissingBox(&'static str),
    /// A sample table referenced a chunk or sample that does not exist.
    InconsistentTables(&'static str),
}

impl fmt::Display for Mp4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mp4Error::Truncated(what) => write!(f, "truncated MP4: {what}"),
            Mp4Error::BadBoxSize { kind, size } => {
                write!(f, "box '{}' declares an impossible size {size}", fourcc(*kind))
            }
            Mp4Error::NoTracks => write!(f, "MP4 has no tracks"),
            Mp4Error::MissingBox(b) => write!(f, "MP4 track has no {b} box"),
            Mp4Error::InconsistentTables(w) => write!(f, "MP4 sample tables disagree: {w}"),
        }
    }
}

fn fourcc(k: [u8; 4]) -> String {
    String::from_utf8_lossy(&k).into_owned()
}

/// What a track carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackKind {
    Video,
    Audio,
    Other,
}

/// One sample (an access unit): where it is in the file, when it is shown, and whether
/// decoding can start from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    /// Byte offset in the file.
    pub offset: u64,
    pub size: u32,
    /// Decode timestamp, in the track's timescale.
    pub dts: u64,
    /// Presentation timestamp, in the track's timescale (`dts` plus the `ctts` offset).
    pub pts: u64,
    /// Duration, in the track's timescale.
    pub duration: u32,
    /// Whether this sample is a sync point (a keyframe) - where a seek may land.
    pub sync: bool,
}

/// One track of a movie, with its sample table already flattened.
#[derive(Clone, Debug)]
pub struct Track {
    pub id: u32,
    pub kind: TrackKind,
    /// The `stsd` entry's format, e.g. `avc1` for H.264 or `mp4a` for AAC.
    pub codec: [u8; 4],
    /// Ticks per second for this track's timestamps.
    pub timescale: u32,
    /// Track duration in its own timescale.
    pub duration: u64,
    /// Display size from `tkhd`, in whole pixels (the field is 16.16 fixed point).
    /// Zero for a non-visual track.
    pub width: u32,
    pub height: u32,
    /// The codec setup bytes from the `stsd` entry - for `avc1` this is the `avcC`
    /// record (SPS/PPS), which a decoder needs before the first frame.
    pub codec_config: Vec<u8>,
    pub samples: Vec<Sample>,
}

impl Track {
    /// >>> THE `AudioSpecificConfig` INSIDE AN `esds`, which is what a decoder is
    /// configured from and is NOT the box's own bytes.
    ///
    /// [`Track::codec_config`] holds the whole `esds` payload for an audio track, and that
    /// is a nest of MPEG-4 descriptors: an `ES_Descriptor` (tag 3) holding a
    /// `DecoderConfigDescriptor` (tag 4) holding a `DecoderSpecificInfo` (tag 5), whose
    /// payload IS the config. Each descriptor's length is a variable-length integer with a
    /// continuation bit, which is the part that makes this worth writing once.
    ///
    /// Empty when the track carries no such descriptor - a caller then has the channel
    /// count and sample rate from the container and nothing else, which is enough to
    /// synthesise one for plain AAC-LC.
    pub fn audio_specific_config(&self) -> Vec<u8> {
        // version + flags, then the ES_Descriptor.
        let mut at = 4usize;
        let d = &self.codec_config;
        // ES_Descriptor
        let Some((body, next)) = descriptor(d, at, 0x03) else { return Vec::new() };
        let _ = next;
        // ES_ID (2) + a flags byte, whose bits say which optional fields follow.
        let mut inner = body.0 + 3;
        let flags = d.get(body.0 + 2).copied().unwrap_or(0);
        if flags & 0x80 != 0 {
            // streamDependenceFlag: a 16-bit dependsOn_ES_ID.
            inner += 2;
        }
        if flags & 0x40 != 0 {
            // URL_Flag: a length-prefixed URL.
            let len = d.get(inner).copied().unwrap_or(0) as usize;
            inner += 1 + len;
        }
        if flags & 0x20 != 0 {
            // OCRstreamFlag: a 16-bit OCR_ES_ID.
            inner += 2;
        }
        at = inner;
        // DecoderConfigDescriptor: objectTypeIndication, streamType, bufferSize (3),
        // maxBitrate (4), avgBitrate (4) - 13 bytes - then the specific info.
        let Some((cfg, _)) = descriptor(d, at, 0x04) else { return Vec::new() };
        let Some((asc, _)) = descriptor(d, cfg.0 + 13, 0x05) else { return Vec::new() };
        d.get(asc.0..asc.1).map(|b| b.to_vec()).unwrap_or_default()
    }

    /// Duration in microseconds.
    pub fn duration_us(&self) -> u64 {
        if self.timescale == 0 {
            return 0;
        }
        self.duration.saturating_mul(1_000_000) / self.timescale as u64
    }

    /// The index of the last sync sample at or before `index`, for a seek that must land
    /// on something decodable. `None` if the track has no sync sample at or before it.
    pub fn sync_at_or_before(&self, index: usize) -> Option<usize> {
        self.samples[..=index.min(self.samples.len().saturating_sub(1))]
            .iter()
            .rposition(|s| s.sync)
    }

    /// The first sample whose presentation time is at or after `pts`.
    pub fn sample_at_pts(&self, pts: u64) -> Option<usize> {
        self.samples.iter().position(|s| s.pts >= pts)
    }
}

/// A parsed movie.
#[derive(Clone, Debug)]
pub struct Mp4 {
    /// Ticks per second for the movie timeline (`mvhd`).
    pub timescale: u32,
    pub duration: u64,
    pub tracks: Vec<Track>,
}

impl Mp4 {
    /// Parse the movie header out of `data`. The sample DATA (`mdat`) is not touched -
    /// only the tables - so this is cheap even for a file that is mostly video.
    pub fn parse(data: &[u8]) -> Result<Mp4, Mp4Error> {
        let moov = find_box(data, *b"moov")?.ok_or(Mp4Error::NoTracks)?;
        Mp4::parse_moov(moov)
    }

    /// Parse from the `moov` box's payload alone.
    ///
    /// This is the entry point for a caller that is STREAMING the file rather than
    /// holding it: the header is a few hundred kilobytes at the front or the back, and
    /// everything else is sample data to be read on demand. It matters most where reads
    /// are not free - a browser serving the container out of storage a range at a time -
    /// because the alternative is pulling tens of megabytes through that path before the
    /// first frame can be shown.
    pub fn parse_moov(moov: &[u8]) -> Result<Mp4, Mp4Error> {
        let mvhd = find_box(moov, *b"mvhd")?.ok_or(Mp4Error::MissingBox("mvhd"))?;
        let (timescale, duration) = parse_mvhd(mvhd)?;

        let mut tracks = Vec::new();
        for trak in boxes(moov, *b"trak")? {
            tracks.push(parse_trak(trak)?);
        }
        if tracks.is_empty() {
            return Err(Mp4Error::NoTracks);
        }
        Ok(Mp4 { timescale, duration, tracks })
    }

    /// The first track of a kind, which is what a player wants ("the video track").
    pub fn track(&self, kind: TrackKind) -> Option<&Track> {
        self.tracks.iter().find(|t| t.kind == kind)
    }

    /// Movie duration in microseconds.
    pub fn duration_us(&self) -> u64 {
        if self.timescale == 0 {
            return 0;
        }
        self.duration.saturating_mul(1_000_000) / self.timescale as u64
    }
}

// ---------------------------------------------------------------------------
// Box walking
// ---------------------------------------------------------------------------

/// The payload of each child box of `data` with this type, in order.
fn boxes(data: &[u8], want: [u8; 4]) -> Result<Vec<&[u8]>, Mp4Error> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 8 <= data.len() {
        let (kind, body, next) = read_box(data, off)?;
        if kind == want {
            out.push(body);
        }
        off = next;
    }
    Ok(out)
}

/// The payload of the first child box of `data` with this type.
fn find_box(data: &[u8], want: [u8; 4]) -> Result<Option<&[u8]>, Mp4Error> {
    let mut off = 0usize;
    while off + 8 <= data.len() {
        let (kind, body, next) = read_box(data, off)?;
        if kind == want {
            return Ok(Some(body));
        }
        off = next;
    }
    Ok(None)
}

/// Where the `moov` box lives in a file, as `(offset of its PAYLOAD, payload length)`,
/// found by walking only the top-level box headers.
///
/// A caller that cannot hold the whole file uses this to read the header and nothing else.
/// `read_at(offset, len)` supplies bytes; it may return fewer than asked for at the end of
/// the file, and returning nothing ends the walk.
///
/// `moov` is at the FRONT of some files and the BACK of others - both appear among one
/// title's own movies - so neither end can be assumed and the walk is the only answer that
/// works for both.
pub fn find_moov(
    file_len: u64,
    mut read_at: impl FnMut(u64, usize) -> Vec<u8>,
) -> Result<(u64, u64), Mp4Error> {
    let mut off = 0u64;
    while off + 8 <= file_len {
        let header = read_at(off, 16);
        if header.len() < 8 {
            break;
        }
        let size = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as u64;
        let kind = [header[4], header[5], header[6], header[7]];
        // 1 means the real size is a 64-bit field after the type; 0 means "to end of file".
        let (size, header_len) = match size {
            1 => {
                if header.len() < 16 {
                    break;
                }
                (be_u64(&header, 8)?, 16u64)
            }
            0 => (file_len - off, 8),
            n if n < 8 => return Err(Mp4Error::BadBoxSize { kind, size: n }),
            n => (n, 8),
        };
        if kind == *b"moov" {
            return Ok((off + header_len, size.saturating_sub(header_len)));
        }
        off = off.checked_add(size).ok_or(Mp4Error::Truncated("box chain"))?;
    }
    Err(Mp4Error::NoTracks)
}

/// Follow a path of nested box types, e.g. `["mdia", "minf", "stbl"]`.
fn find_path<'a>(mut data: &'a [u8], path: &[[u8; 4]]) -> Result<Option<&'a [u8]>, Mp4Error> {
    for step in path {
        match find_box(data, *step)? {
            Some(inner) => data = inner,
            None => return Ok(None),
        }
    }
    Ok(Some(data))
}

/// `(type, payload, offset of the next box)` for the box starting at `off`.
fn read_box(data: &[u8], off: usize) -> Result<([u8; 4], &[u8], usize), Mp4Error> {
    let head = data.get(off..off + 8).ok_or(Mp4Error::Truncated("box header"))?;
    let mut kind = [0u8; 4];
    kind.copy_from_slice(&head[4..8]);
    let size32 = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as u64;
    let (size, header) = match size32 {
        // 0 means "to the end of the enclosing box".
        0 => ((data.len() - off) as u64, 8usize),
        1 => {
            let ext = data.get(off + 8..off + 16).ok_or(Mp4Error::Truncated("64-bit box size"))?;
            let mut b = [0u8; 8];
            b.copy_from_slice(ext);
            (u64::from_be_bytes(b), 16usize)
        }
        n => (n, 8usize),
    };
    if size < header as u64 {
        return Err(Mp4Error::BadBoxSize { kind, size });
    }
    let end = off
        .checked_add(size as usize)
        .filter(|&e| e <= data.len())
        .ok_or(Mp4Error::Truncated("box payload"))?;
    Ok((kind, &data[off + header..end], end))
}

fn be_u32(d: &[u8], at: usize) -> Result<u32, Mp4Error> {
    let b = d.get(at..at + 4).ok_or(Mp4Error::Truncated("u32"))?;
    Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

fn be_u16(d: &[u8], at: usize) -> Result<u16, Mp4Error> {
    let b = d.get(at..at + 2).ok_or(Mp4Error::Truncated("u16"))?;
    Ok(u16::from_be_bytes([b[0], b[1]]))
}

fn be_u64(d: &[u8], at: usize) -> Result<u64, Mp4Error> {
    let b = d.get(at..at + 8).ok_or(Mp4Error::Truncated("u64"))?;
    let mut v = [0u8; 8];
    v.copy_from_slice(b);
    Ok(u64::from_be_bytes(v))
}

/// A full box's `version` byte (the payload starts `version:u8, flags:u24`).
fn version(d: &[u8]) -> Result<u8, Mp4Error> {
    d.first().copied().ok_or(Mp4Error::Truncated("full box version"))
}

// ---------------------------------------------------------------------------
// Header boxes
// ---------------------------------------------------------------------------

/// `(timescale, duration)` from `mvhd`. Version 1 widens the times to 64 bits.
fn parse_mvhd(d: &[u8]) -> Result<(u32, u64), Mp4Error> {
    match version(d)? {
        1 => Ok((be_u32(d, 20)?, be_u64(d, 24)?)),
        _ => Ok((be_u32(d, 12)?, be_u32(d, 16)? as u64)),
    }
}

/// `(track id, width, height)` from `tkhd`. The dimensions are 16.16 fixed point at the
/// end of the box, and are the DISPLAY size (after any aspect correction), which is what
/// a title asks a movie for.
fn parse_tkhd(d: &[u8]) -> Result<(u32, u32, u32), Mp4Error> {
    // Payload layout (after the 8-byte box header). Version 1 widens creation,
    // modification and duration from 32 to 64 bits, moving everything after them by 12.
    //   v0: flags 4, created 4, modified 4, track_id 4, reserved 4, duration 4,
    //       reserved 8, layer 2, alt_group 2, volume 2, reserved 2, matrix 36 = 76,
    //       then width 4, height 4.
    let (id, dims_at) = match version(d)? {
        1 => (be_u32(d, 20)?, 88),
        _ => (be_u32(d, 12)?, 76),
    };
    let w = be_u32(d, dims_at)? >> 16;
    let h = be_u32(d, dims_at + 4)? >> 16;
    Ok((id, w, h))
}

/// `(timescale, duration)` from `mdhd`.
fn parse_mdhd(d: &[u8]) -> Result<(u32, u64), Mp4Error> {
    match version(d)? {
        1 => Ok((be_u32(d, 20)?, be_u64(d, 24)?)),
        _ => Ok((be_u32(d, 12)?, be_u32(d, 16)? as u64)),
    }
}

fn parse_hdlr(d: &[u8]) -> TrackKind {
    match d.get(8..12) {
        Some(b"vide") => TrackKind::Video,
        Some(b"soun") => TrackKind::Audio,
        _ => TrackKind::Other,
    }
}

/// `(codec fourcc, setup bytes)` from the first `stsd` entry.
///
/// The entry layout differs per handler, but each begins with the 8-byte `SampleEntry`
/// fields (6 reserved + data_reference_index) and the codec-specific setup box (`avcC`,
/// `esds`, ...) is a child box after a fixed-size prefix of the ENTRY PAYLOAD: 78 bytes
/// for a visual entry (8 + 70 of `VisualSampleEntry`), 28 for an audio one (8 + 20 of
/// `AudioSampleEntry`). Anything else yields no setup bytes rather than a guess.
fn parse_stsd(d: &[u8], kind: TrackKind) -> Result<([u8; 4], Vec<u8>), Mp4Error> {
    // version/flags (4) + entry count (4), then the entries as boxes.
    let entries = d.get(8..).ok_or(Mp4Error::Truncated("stsd"))?;
    let (codec, body, _) = read_box(entries, 0)?;
    let prefix = match kind {
        TrackKind::Video => 78,
        TrackKind::Audio => 28,
        TrackKind::Other => return Ok((codec, Vec::new())),
    };
    let config = match body.get(prefix..) {
        Some(rest) => match read_box(rest, 0) {
            Ok((_, setup, _)) => setup.to_vec(),
            Err(_) => Vec::new(),
        },
        None => Vec::new(),
    };
    Ok((codec, config))
}


/// One MPEG-4 descriptor at `at`, if it carries `tag`: its payload range and the offset
/// just past it. The length is 1 to 4 bytes, each carrying 7 bits with the top bit meaning
/// "another byte follows".
fn descriptor(d: &[u8], at: usize, tag: u8) -> Option<((usize, usize), usize)> {
    if d.get(at).copied()? != tag {
        return None;
    }
    let mut len = 0usize;
    let mut cursor = at + 1;
    for _ in 0..4 {
        let byte = d.get(cursor).copied()?;
        cursor += 1;
        len = (len << 7) | usize::from(byte & 0x7F);
        if byte & 0x80 == 0 {
            break;
        }
    }
    let end = cursor.checked_add(len)?;
    if end > d.len() {
        return None;
    }
    Some(((cursor, end), end))
}

// ---------------------------------------------------------------------------
// Sample tables
// ---------------------------------------------------------------------------

fn parse_trak(trak: &[u8]) -> Result<Track, Mp4Error> {
    let tkhd = find_box(trak, *b"tkhd")?.ok_or(Mp4Error::MissingBox("tkhd"))?;
    let (id, width, height) = parse_tkhd(tkhd)?;
    let mdia = find_box(trak, *b"mdia")?.ok_or(Mp4Error::MissingBox("mdia"))?;
    let mdhd = find_box(mdia, *b"mdhd")?.ok_or(Mp4Error::MissingBox("mdhd"))?;
    let (timescale, duration) = parse_mdhd(mdhd)?;
    let kind = find_box(mdia, *b"hdlr")?.map(parse_hdlr).unwrap_or(TrackKind::Other);
    let stbl = find_path(mdia, &[*b"minf", *b"stbl"])?.ok_or(Mp4Error::MissingBox("stbl"))?;
    let stsd = find_box(stbl, *b"stsd")?.ok_or(Mp4Error::MissingBox("stsd"))?;
    let (codec, codec_config) = parse_stsd(stsd, kind)?;
    let samples = build_samples(stbl)?;
    Ok(Track { id, kind, codec, timescale, duration, width, height, codec_config, samples })
}

/// Flatten `stts`/`ctts`/`stsc`/`stsz`/`stco`/`stss` into one sample list.
fn build_samples(stbl: &[u8]) -> Result<Vec<Sample>, Mp4Error> {
    let sizes = parse_stsz(find_box(stbl, *b"stsz")?.ok_or(Mp4Error::MissingBox("stsz"))?)?;
    let chunk_offsets = match find_box(stbl, *b"stco")? {
        Some(b) => parse_stco(b, false)?,
        None => parse_stco(
            find_box(stbl, *b"co64")?.ok_or(Mp4Error::MissingBox("stco/co64"))?,
            true,
        )?,
    };
    let stsc = parse_stsc(find_box(stbl, *b"stsc")?.ok_or(Mp4Error::MissingBox("stsc"))?)?;
    let durations = parse_stts(find_box(stbl, *b"stts")?.ok_or(Mp4Error::MissingBox("stts"))?)?;
    let ctts = match find_box(stbl, *b"ctts")? {
        Some(b) => parse_ctts(b)?,
        None => Vec::new(),
    };
    // No `stss` means every sample is a sync point, which is what the spec says and what
    // an all-keyframe stream relies on.
    let sync: Option<Vec<u32>> = match find_box(stbl, *b"stss")? {
        Some(b) => Some(parse_u32_table(b)?),
        None => None,
    };

    let n = sizes.len();
    let mut samples = Vec::with_capacity(n);
    // Walk chunks in order, handing each its run of samples at increasing offsets.
    let mut sample_index = 0usize;
    let mut dts = 0u64;
    for (chunk_index, &chunk_off) in chunk_offsets.iter().enumerate() {
        let per_chunk = samples_in_chunk(&stsc, chunk_index as u32 + 1);
        let mut offset = chunk_off;
        for _ in 0..per_chunk {
            if sample_index >= n {
                break;
            }
            let size = sizes[sample_index];
            let duration = *durations
                .get(sample_index)
                .ok_or(Mp4Error::InconsistentTables("fewer durations than samples"))?;
            let shift = ctts.get(sample_index).copied().unwrap_or(0);
            samples.push(Sample {
                offset,
                size,
                dts,
                pts: dts.saturating_add_signed(shift as i64),
                duration,
                sync: true, // corrected below when an stss is present
            });
            offset += size as u64;
            dts += duration as u64;
            sample_index += 1;
        }
    }
    if sample_index != n {
        return Err(Mp4Error::InconsistentTables("chunks do not cover every sample"));
    }
    if let Some(sync) = sync {
        for s in samples.iter_mut() {
            s.sync = false;
        }
        for one_based in sync {
            let Some(s) = samples.get_mut(one_based.saturating_sub(1) as usize) else {
                return Err(Mp4Error::InconsistentTables("stss names a sample that does not exist"));
            };
            s.sync = true;
        }
    }
    Ok(samples)
}

/// `stsz`: either one shared size or one per sample.
fn parse_stsz(d: &[u8]) -> Result<Vec<u32>, Mp4Error> {
    let uniform = be_u32(d, 4)?;
    let count = be_u32(d, 8)? as usize;
    if uniform != 0 {
        return Ok(vec![uniform; count]);
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(be_u32(d, 12 + 4 * i)?);
    }
    Ok(out)
}

/// `stco` (32-bit) or `co64` (64-bit) chunk offsets.
fn parse_stco(d: &[u8], wide: bool) -> Result<Vec<u64>, Mp4Error> {
    let count = be_u32(d, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(if wide { be_u64(d, 8 + 8 * i)? } else { be_u32(d, 8 + 4 * i)? as u64 });
    }
    Ok(out)
}

/// One `stsc` run: from `first_chunk` (1-based) onward, each chunk holds this many
/// samples, until the next run's `first_chunk`.
#[derive(Clone, Copy, Debug)]
struct StscRun {
    first_chunk: u32,
    samples_per_chunk: u32,
}

fn parse_stsc(d: &[u8]) -> Result<Vec<StscRun>, Mp4Error> {
    let count = be_u32(d, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = 8 + 12 * i;
        out.push(StscRun {
            first_chunk: be_u32(d, at)?,
            samples_per_chunk: be_u32(d, at + 4)?,
        });
    }
    Ok(out)
}

/// How many samples chunk `chunk` (1-based) holds, per the run-length `stsc`.
fn samples_in_chunk(stsc: &[StscRun], chunk: u32) -> u32 {
    let mut n = 0;
    for run in stsc {
        if run.first_chunk > chunk {
            break;
        }
        n = run.samples_per_chunk;
    }
    n
}

/// `stts`: run-length coded per-sample durations, expanded.
fn parse_stts(d: &[u8]) -> Result<Vec<u32>, Mp4Error> {
    let count = be_u32(d, 4)? as usize;
    let mut out = Vec::new();
    for i in 0..count {
        let at = 8 + 8 * i;
        let n = be_u32(d, at)?;
        let delta = be_u32(d, at + 4)?;
        // A corrupt count could otherwise ask for gigabytes; the cap is far above any
        // real movie's sample count.
        if out.len().saturating_add(n as usize) > 16_000_000 {
            return Err(Mp4Error::InconsistentTables("stts run count is implausible"));
        }
        out.extend(std::iter::repeat_n(delta, n as usize));
    }
    Ok(out)
}

/// `ctts`: run-length coded pts-minus-dts offsets, expanded. Version 1 makes them signed.
fn parse_ctts(d: &[u8]) -> Result<Vec<i32>, Mp4Error> {
    let signed = version(d)? >= 1;
    let count = be_u32(d, 4)? as usize;
    let mut out = Vec::new();
    for i in 0..count {
        let at = 8 + 8 * i;
        let n = be_u32(d, at)?;
        let raw = be_u32(d, at + 4)?;
        let offset = if signed { raw as i32 } else { raw as i32 };
        if out.len().saturating_add(n as usize) > 16_000_000 {
            return Err(Mp4Error::InconsistentTables("ctts run count is implausible"));
        }
        out.extend(std::iter::repeat_n(offset, n as usize));
    }
    Ok(out)
}

/// A plain `count` then `count` u32 entries (`stss`).
fn parse_u32_table(d: &[u8]) -> Result<Vec<u32>, Mp4Error> {
    let count = be_u32(d, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(be_u32(d, 8 + 4 * i)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// avcC -> Annex B
// ---------------------------------------------------------------------------

/// The parameter sets and NAL length size an `avcC` record carries.
///
/// An MP4's H.264 samples are length-prefixed and carry no SPS/PPS: those live once, in
/// the sample entry. A decoder handed only the samples has nothing to configure itself
/// with - so a caller that must produce a self-describing ELEMENTARY STREAM (which is what
/// a video-decode API is given; it knows nothing about containers) needs both halves.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AvcC {
    /// 1, 2 or 4 - the width of each sample's NAL length prefixes.
    pub length_size: usize,
    /// Sequence parameter sets, each a bare NAL with no prefix.
    pub sps: Vec<Vec<u8>>,
    /// Picture parameter sets, same form.
    pub pps: Vec<Vec<u8>>,
}

impl AvcC {
    /// Parse an `avcC` record (ISO/IEC 14496-15 5.2.4.1).
    pub fn parse(d: &[u8]) -> Result<AvcC, Mp4Error> {
        if d.len() < 7 {
            return Err(Mp4Error::Truncated("avcC record"));
        }
        let mut out = AvcC { length_size: (d[4] & 0x03) as usize + 1, ..AvcC::default() };
        let mut off = 5usize;
        let read_set = |off: &mut usize, into: &mut Vec<Vec<u8>>, count: usize| {
            for _ in 0..count {
                let len = be_u16(d, *off)? as usize;
                *off += 2;
                let end = off.checked_add(len).ok_or(Mp4Error::Truncated("avcC parameter set"))?;
                if end > d.len() {
                    return Err(Mp4Error::Truncated("avcC parameter set"));
                }
                into.push(d[*off..end].to_vec());
                *off = end;
            }
            Ok::<(), Mp4Error>(())
        };
        let sps_count = (d[off] & 0x1f) as usize;
        off += 1;
        let mut sps = Vec::new();
        read_set(&mut off, &mut sps, sps_count)?;
        if off >= d.len() {
            return Err(Mp4Error::Truncated("avcC picture parameter set count"));
        }
        let pps_count = d[off] as usize;
        off += 1;
        let mut pps = Vec::new();
        read_set(&mut off, &mut pps, pps_count)?;
        out.sps = sps;
        out.pps = pps;
        Ok(out)
    }

    /// The parameter sets as Annex B NALs, ready to precede a keyframe.
    pub fn annex_b_parameter_sets(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for nal in self.sps.iter().chain(self.pps.iter()) {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
        }
        out
    }

    /// Rewrite one length-prefixed sample into Annex B, appending to `out`.
    ///
    /// Fails rather than guessing on a prefix that runs past the end: a sample table and a
    /// length prefix disagreeing means the file was mis-parsed, and a decoder handed the
    /// remains of that would fail somewhere much less informative.
    pub fn sample_to_annex_b(&self, sample: &[u8], out: &mut Vec<u8>) -> Result<(), Mp4Error> {
        let n = self.length_size;
        let mut off = 0usize;
        while off < sample.len() {
            if off + n > sample.len() {
                return Err(Mp4Error::Truncated("NAL length prefix"));
            }
            let mut len = 0usize;
            for i in 0..n {
                len = (len << 8) | sample[off + i] as usize;
            }
            off += n;
            let end = off.checked_add(len).ok_or(Mp4Error::Truncated("NAL payload"))?;
            if end > sample.len() {
                return Err(Mp4Error::Truncated("NAL payload"));
            }
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(&sample[off..end]);
            off = end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a box: 4-byte size, 4-byte type, payload.
    fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((body.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    fn cat(parts: &[Vec<u8>]) -> Vec<u8> {
        parts.concat()
    }

    fn u32s(vals: &[u32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_be_bytes()).collect()
    }

    /// A one-video-track movie: 4 samples in 2 chunks, keyframe every other sample,
    /// 600-tick durations at timescale 600 (so 1 fps, which makes the arithmetic
    /// checkable by eye).
    fn sample_movie() -> Vec<u8> {
        let mut tkhd = vec![0u8; 84];
        tkhd[12..16].copy_from_slice(&1u32.to_be_bytes()); // track id
        tkhd[76..80].copy_from_slice(&(640u32 << 16).to_be_bytes());
        tkhd[80..84].copy_from_slice(&(360u32 << 16).to_be_bytes());

        let mut mdhd = vec![0u8; 24];
        mdhd[12..16].copy_from_slice(&600u32.to_be_bytes()); // timescale
        mdhd[16..20].copy_from_slice(&2400u32.to_be_bytes()); // duration

        let mut hdlr = vec![0u8; 12];
        hdlr[8..12].copy_from_slice(b"vide");

        // stsd: one avc1 visual entry with an avcC child after the 78-byte prefix.
        let mut avc1 = vec![0u8; 78];
        avc1.extend(bx(b"avcC", &[1, 2, 3, 4]));
        let mut stsd = vec![0u8; 4];
        stsd.extend(u32s(&[1]));
        stsd.extend(bx(b"avc1", &avc1));

        let stts = cat(&[vec![0u8; 4], u32s(&[1, 4, 600])]);
        let stsc = cat(&[vec![0u8; 4], u32s(&[1, 1, 2, 1])]);
        let stsz = cat(&[vec![0u8; 4], u32s(&[0, 4, 10, 20, 30, 40])]);
        let stco = cat(&[vec![0u8; 4], u32s(&[2, 1000, 2000])]);
        let stss = cat(&[vec![0u8; 4], u32s(&[2, 1, 3])]);

        let stbl = cat(&[
            bx(b"stsd", &stsd),
            bx(b"stts", &stts),
            bx(b"stsc", &stsc),
            bx(b"stsz", &stsz),
            bx(b"stco", &stco),
            bx(b"stss", &stss),
        ]);
        let minf = bx(b"stbl", &stbl);
        let mdia = cat(&[bx(b"mdhd", &mdhd), bx(b"hdlr", &hdlr), bx(b"minf", &minf)]);
        let trak = cat(&[bx(b"tkhd", &tkhd), bx(b"mdia", &mdia)]);

        let mut mvhd = vec![0u8; 24];
        mvhd[12..16].copy_from_slice(&600u32.to_be_bytes());
        mvhd[16..20].copy_from_slice(&2400u32.to_be_bytes());

        let moov = cat(&[bx(b"mvhd", &mvhd), bx(b"trak", &trak)]);
        cat(&[bx(b"ftyp", b"mp42\0\0\0\0mp42mp41"), bx(b"moov", &moov), bx(b"mdat", &[0u8; 16])])
    }

    #[test]
    fn reads_the_movie_and_track_headers() {
        let m = Mp4::parse(&sample_movie()).unwrap();
        assert_eq!(m.timescale, 600);
        assert_eq!(m.duration_us(), 4_000_000);
        assert_eq!(m.tracks.len(), 1);
        let t = m.track(TrackKind::Video).unwrap();
        assert_eq!(t.id, 1);
        assert_eq!((t.width, t.height), (640, 360));
        assert_eq!(&t.codec, b"avc1");
        assert_eq!(t.codec_config, vec![1, 2, 3, 4], "avcC setup bytes not recovered");
        assert_eq!(t.duration_us(), 4_000_000);
    }

    #[test]
    fn flattens_the_sample_tables() {
        let m = Mp4::parse(&sample_movie()).unwrap();
        let t = m.track(TrackKind::Video).unwrap();
        // 2 chunks: the stsc says chunk 1 holds 2 samples, chunk 2 onward holds 1 - but
        // there are 4 samples and 2 chunks, so chunk 2 must take the remaining 2. The
        // spec's run-length form says chunk 2 holds 1, which would leave a sample
        // uncovered, so this movie declares 2/2 via the second run's value.
        assert_eq!(t.samples.len(), 4, "not every sample was placed in a chunk");
        assert_eq!(t.samples[0], Sample {
            offset: 1000,
            size: 10,
            dts: 0,
            pts: 0,
            duration: 600,
            sync: true
        });
        // Samples run consecutively inside a chunk.
        assert_eq!(t.samples[1].offset, 1010);
        assert_eq!(t.samples[1].dts, 600);
        // ...and the next chunk restarts at its own offset.
        assert_eq!(t.samples[2].offset, 2000);
        assert_eq!(t.samples[3].offset, 2030);
        // stss named samples 1 and 3 (1-based).
        assert_eq!(
            t.samples.iter().map(|s| s.sync).collect::<Vec<_>>(),
            vec![true, false, true, false]
        );
    }

    #[test]
    fn seeks_to_a_sync_sample_and_by_time() {
        let m = Mp4::parse(&sample_movie()).unwrap();
        let t = m.track(TrackKind::Video).unwrap();
        assert_eq!(t.sync_at_or_before(3), Some(2));
        assert_eq!(t.sync_at_or_before(1), Some(0));
        assert_eq!(t.sample_at_pts(0), Some(0));
        assert_eq!(t.sample_at_pts(1200), Some(2));
        assert_eq!(t.sample_at_pts(99_999), None);
    }

    #[test]
    fn no_stss_means_every_sample_is_a_sync_point() {
        // Drop the stss by rebuilding without it: the spec says its absence means all
        // samples are sync, which an all-intra stream depends on.
        let movie = sample_movie();
        let at = movie.windows(4).position(|w| w == b"stss").unwrap() - 4;
        let size = u32::from_be_bytes([movie[at], movie[at + 1], movie[at + 2], movie[at + 3]]);
        let mut trimmed = movie.clone();
        trimmed.drain(at..at + size as usize);
        // Fix up every enclosing box size by the amount removed.
        let shrink = |b: &mut Vec<u8>, kind: &[u8; 4]| {
            let p = b.windows(4).position(|w| w == kind).unwrap() - 4;
            let old = u32::from_be_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]);
            b[p..p + 4].copy_from_slice(&(old - size).to_be_bytes());
        };
        for kind in [b"stbl", b"minf", b"mdia", b"trak", b"moov"] {
            shrink(&mut trimmed, kind);
        }
        let t = Mp4::parse(&trimmed).unwrap();
        let t = t.track(TrackKind::Video).unwrap();
        assert!(t.samples.iter().all(|s| s.sync), "absent stss must mean all-sync");
    }

    #[test]
    fn a_malformed_file_is_an_error_not_a_panic() {
        assert_eq!(Mp4::parse(&[]).unwrap_err(), Mp4Error::NoTracks);
        // Arbitrary bytes read as a box header claim a size the buffer does not have.
        assert!(matches!(Mp4::parse(b"not an mp4 at all").unwrap_err(), Mp4Error::Truncated(_)));
        // A box claiming a size smaller than its own header.
        let mut bad = 4u32.to_be_bytes().to_vec();
        bad.extend_from_slice(b"moov");
        assert!(matches!(Mp4::parse(&bad).unwrap_err(), Mp4Error::BadBoxSize { .. }));
        // A box claiming more bytes than the file holds.
        let mut over = 9999u32.to_be_bytes().to_vec();
        over.extend_from_slice(b"moov");
        assert!(matches!(Mp4::parse(&over).unwrap_err(), Mp4Error::Truncated(_)));
        // Truncating the real movie anywhere must never panic.
        let movie = sample_movie();
        for cut in 0..movie.len() {
            let _ = Mp4::parse(&movie[..cut]);
        }
    }

    /// The synthetic movies above test the parser against the layout it documents. This
    /// one tests it against a REAL retail MP4, which is the only thing that catches the
    /// layout itself being wrong. Content-free assertions; skips without the fixture.
    #[test]
    fn reads_a_real_retail_movie() {
        let Some(dir) = crate::ingest::testfix::game_dir() else { return };
        // The fixture may be an app root or a dump root; look for movies under both.
        let mut found = None;
        for sub in ["files/Data/Movie", "Data/Movie", "movie", "Movie"] {
            let p = dir.join(sub);
            if let Ok(entries) = std::fs::read_dir(&p) {
                for e in entries.flatten() {
                    if e.path().extension().is_some_and(|x| x.eq_ignore_ascii_case("mp4")) {
                        found = Some(e.path());
                        break;
                    }
                }
            }
            if found.is_some() {
                break;
            }
        }
        let Some(path) = found else { return };
        let bytes = std::fs::read(&path).expect("read the movie");
        let m = Mp4::parse(&bytes).expect("parse a real retail MP4");

        assert!(m.duration_us() > 0, "movie has no duration");
        let v = m.track(TrackKind::Video).expect("no video track");
        assert_eq!(&v.codec, b"avc1", "expected H.264 video");
        assert!(v.width > 0 && v.height > 0, "video track has no display size");
        assert!(!v.codec_config.is_empty(), "no avcC setup bytes");
        assert!(!v.samples.is_empty(), "video track has no samples");
        assert!(v.samples.iter().any(|s| s.sync), "no keyframe to start decoding from");
        // Every sample must lie inside the file, and timestamps must not go backwards.
        let mut last_dts = 0;
        for s in &v.samples {
            assert!(
                s.offset + s.size as u64 <= bytes.len() as u64,
                "sample runs past the end of the file"
            );
            assert!(s.dts >= last_dts, "decode timestamps went backwards");
            last_dts = s.dts;
        }
        // The flattened table must agree with the track header on duration, to within one
        // sample: a real file's `mdhd` duration is a rounded movie-timeline value, so the
        // last sample routinely overhangs it by a tick or two. A disagreement bigger than
        // one sample would mean the tables were misread.
        let table_duration: u64 = v.samples.iter().map(|s| s.duration as u64).sum();
        let slack = v.samples.last().map(|s| s.duration as u64).unwrap_or(0);
        assert!(
            table_duration.abs_diff(v.duration) <= slack,
            "sample durations disagree with mdhd: {table_duration} vs {}",
            v.duration
        );
    }
}
