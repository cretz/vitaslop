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
