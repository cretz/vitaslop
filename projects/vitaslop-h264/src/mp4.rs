//! A minimal ISO base media file (MP4/MOV) reader - enough to feed the decoder.
//!
//! Not a demuxer library: it reads ONE H.264 video track out of a non-fragmented file, and
//! reports its parameter sets, its sample table, and its timestamps. That is exactly what
//! [`crate::Decoder`] needs and nothing more - audio, subtitles, edit lists, fragmented
//! files and progressive download are all somebody else's job, and each is refused by name
//! rather than half-handled.
//!
//! It exists because the alternative - "bring your own demuxer" - makes the simplest
//! possible use of this crate (decode this .mp4) depend on a much larger dependency than
//! the decoder itself.

use crate::bitstream::avcc::AvcC;
use crate::error::{Error, Result};

/// One sample (one access unit) of a video track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sample {
    /// Byte offset of the sample within the file.
    pub offset: usize,
    /// Size of the sample in bytes.
    pub size: usize,
    /// Decode timestamp, in the track's own timescale.
    pub dts: i64,
    /// Presentation timestamp, in the track's own timescale (`dts` plus the composition
    /// offset from `ctts`, when the file has one).
    pub pts: i64,
    /// True when the sample is a sync sample (an IDR) - i.e. a seek point.
    pub sync: bool,
}

/// An H.264 video track.
#[derive(Debug, Clone)]
pub struct VideoTrack {
    /// The track's parameter sets, from the `avc1` sample entry's `avcC` box.
    pub avcc: AvcC,
    /// The raw `avcC` record, for handing straight to a platform decoder.
    pub avcc_bytes: Vec<u8>,
    /// Ticks per second for this track's timestamps.
    pub timescale: u32,
    /// Width in pixels, as the sample entry declares it.
    pub width: u32,
    /// Height in pixels, as the sample entry declares it.
    pub height: u32,
    /// The samples, in decode order.
    pub samples: Vec<Sample>,
}

impl VideoTrack {
    /// The bytes of sample `index` within `file`.
    pub fn sample_data<'a>(&self, file: &'a [u8], index: usize) -> Result<&'a [u8]> {
        let s = self
            .samples
            .get(index)
            .ok_or_else(|| Error::bitstream(format!("no sample {index} in the track")))?;
        file.get(s.offset..s.offset + s.size).ok_or_else(|| {
            Error::bitstream(format!(
                "sample {index} runs past the end of the file ({} + {} > {})",
                s.offset,
                s.size,
                file.len()
            ))
        })
    }

    /// Convert a track timestamp to microseconds.
    pub fn to_micros(&self, ticks: i64) -> i64 {
        if self.timescale == 0 {
            return ticks;
        }
        ticks.saturating_mul(1_000_000) / self.timescale as i64
    }
}

/// Find the first H.264 video track in an MP4 file held in memory.
pub fn read_h264_track(file: &[u8]) -> Result<VideoTrack> {
    let moov = find_box(file, 0, file.len(), b"moov")
        .ok_or_else(|| Error::bitstream("no moov box: not an MP4, or a fragmented one"))?;
    if find_box(file, 0, file.len(), b"moof").is_some() {
        return Err(Error::unsupported("fragmented MP4 (moof)"));
    }

    let mut offset = moov.start;
    while let Some(trak) = find_box(file, offset, moov.end, b"trak") {
        offset = trak.end;
        if let Some(track) = read_track(file, trak)? {
            return Ok(track);
        }
    }
    Err(Error::unsupported("the file has no H.264 (avc1) video track"))
}

/// Where a box's payload starts and ends.
#[derive(Debug, Clone, Copy)]
struct BoxRange {
    start: usize,
    end: usize,
}

/// Scan the boxes between `from` and `until` for one with this type.
fn find_box(file: &[u8], from: usize, until: usize, kind: &[u8; 4]) -> Option<BoxRange> {
    let mut at = from;
    while at + 8 <= until {
        let size = u32::from_be_bytes(file[at..at + 4].try_into().ok()?) as usize;
        let name = &file[at + 4..at + 8];
        let (payload, end) = match size {
            // Size 0 means "to the end of the file".
            0 => (at + 8, until),
            // Size 1 means a 64-bit size follows the type.
            1 => {
                if at + 16 > until {
                    return None;
                }
                let large = u64::from_be_bytes(file[at + 8..at + 16].try_into().ok()?) as usize;
                (at + 16, (at + large).min(until))
            }
            _ => (at + 8, (at + size).min(until)),
        };
        if end <= at {
            return None; // a zero-length box would loop forever
        }
        if name == kind {
            return Some(BoxRange { start: payload, end });
        }
        at = end;
    }
    None
}

/// Read one `trak`, returning `None` when it is not an H.264 video track.
fn read_track(file: &[u8], trak: BoxRange) -> Result<Option<VideoTrack>> {
    let Some(mdia) = find_box(file, trak.start, trak.end, b"mdia") else {
        return Ok(None);
    };
    let Some(hdlr) = find_box(file, mdia.start, mdia.end, b"hdlr") else {
        return Ok(None);
    };
    // hdlr: version+flags (4), pre_defined (4), handler_type (4).
    if hdlr.start + 12 > hdlr.end || &file[hdlr.start + 8..hdlr.start + 12] != b"vide" {
        return Ok(None);
    }
    let Some(mdhd) = find_box(file, mdia.start, mdia.end, b"mdhd") else {
        return Ok(None);
    };
    let timescale = read_mdhd_timescale(file, mdhd)?;

    let Some(minf) = find_box(file, mdia.start, mdia.end, b"minf") else {
        return Ok(None);
    };
    let Some(stbl) = find_box(file, minf.start, minf.end, b"stbl") else {
        return Ok(None);
    };
    let Some(stsd) = find_box(file, stbl.start, stbl.end, b"stsd") else {
        return Ok(None);
    };

    // stsd: version+flags (4), entry_count (4), then the sample entries.
    let entries_at = stsd.start + 8;
    let Some(avc1) = find_avc_sample_entry(file, entries_at, stsd.end) else {
        return Ok(None);
    };
    // A visual sample entry is 78 bytes before its extension boxes; width and height sit at
    // offsets 24 and 26 within it.
    if avc1.start + 78 > avc1.end {
        return Err(Error::bitstream("avc1 sample entry is truncated"));
    }
    let width = u16::from_be_bytes([file[avc1.start + 24], file[avc1.start + 25]]) as u32;
    let height = u16::from_be_bytes([file[avc1.start + 26], file[avc1.start + 27]]) as u32;
    let avcc_box = find_box(file, avc1.start + 78, avc1.end, b"avcC")
        .ok_or_else(|| Error::bitstream("avc1 sample entry with no avcC record"))?;
    let avcc_bytes = file[avcc_box.start..avcc_box.end].to_vec();
    let avcc = AvcC::parse(&avcc_bytes)?;

    let samples = read_sample_table(file, stbl)?;
    Ok(Some(VideoTrack { avcc, avcc_bytes, timescale, width, height, samples }))
}

/// The `avc1`/`avc3` entry in an `stsd`, if there is one.
fn find_avc_sample_entry(file: &[u8], from: usize, until: usize) -> Option<BoxRange> {
    // `avc3` differs from `avc1` only in that parameter sets may also appear in-band, which
    // this crate handles anyway.
    find_box(file, from, until, b"avc1").or_else(|| find_box(file, from, until, b"avc3"))
}

fn read_mdhd_timescale(file: &[u8], mdhd: BoxRange) -> Result<u32> {
    if mdhd.start + 4 > mdhd.end {
        return Err(Error::bitstream("mdhd is truncated"));
    }
    let version = file[mdhd.start];
    // version 0: creation (4), modification (4), timescale (4), duration (4)
    // version 1: creation (8), modification (8), timescale (4), duration (8)
    let at = mdhd.start + 4 + if version == 1 { 16 } else { 8 };
    if at + 4 > mdhd.end {
        return Err(Error::bitstream("mdhd is truncated"));
    }
    Ok(u32::from_be_bytes(file[at..at + 4].try_into().unwrap()))
}

/// Build the sample list from `stts`, `stsc`, `stsz`, `stco`/`co64`, `stss` and `ctts`.
fn read_sample_table(file: &[u8], stbl: BoxRange) -> Result<Vec<Sample>> {
    let sizes = read_stsz(file, stbl)?;
    let count = sizes.len();
    let mut samples: Vec<Sample> = Vec::with_capacity(count);

    // Chunk offsets, and how many samples each chunk holds.
    let chunk_offsets = read_chunk_offsets(file, stbl)?;
    let stsc = read_table(file, stbl, b"stsc", 12)?;

    let mut index = 0usize;
    for (chunk, &chunk_offset) in chunk_offsets.iter().enumerate() {
        let per_chunk = samples_per_chunk(&stsc, chunk + 1);
        let mut at = chunk_offset;
        for _ in 0..per_chunk {
            if index >= count {
                break;
            }
            samples.push(Sample {
                offset: at,
                size: sizes[index],
                dts: 0,
                pts: 0,
                sync: false,
            });
            at += sizes[index];
            index += 1;
        }
    }
    if samples.len() != count {
        return Err(Error::bitstream(format!(
            "the sample table describes {count} samples but the chunk map places {}",
            samples.len()
        )));
    }

    // Decode timestamps from stts (run-length coded deltas).
    let stts = read_table(file, stbl, b"stts", 8)?;
    let mut time = 0i64;
    let mut index = 0usize;
    for entry in &stts {
        let run = u32::from_be_bytes(entry[0..4].try_into().unwrap()) as usize;
        let delta = u32::from_be_bytes(entry[4..8].try_into().unwrap()) as i64;
        for _ in 0..run {
            if index >= samples.len() {
                break;
            }
            samples[index].dts = time;
            samples[index].pts = time;
            time += delta;
            index += 1;
        }
    }

    // Composition offsets, when the file reorders (i.e. when it has B-frames).
    if let Ok(ctts) = read_table(file, stbl, b"ctts", 8) {
        let mut index = 0usize;
        for entry in &ctts {
            let run = u32::from_be_bytes(entry[0..4].try_into().unwrap()) as usize;
            // Version 1 makes this signed; reading it as signed is correct for both,
            // because version 0's values are small positive numbers.
            let offset = i32::from_be_bytes(entry[4..8].try_into().unwrap()) as i64;
            for _ in 0..run {
                if index >= samples.len() {
                    break;
                }
                samples[index].pts = samples[index].dts + offset;
                index += 1;
            }
        }
    }

    // Sync samples. A file with no stss says every sample is a sync sample.
    match read_table(file, stbl, b"stss", 4) {
        Ok(stss) => {
            for entry in &stss {
                let number = u32::from_be_bytes(entry[0..4].try_into().unwrap()) as usize;
                if number >= 1 && number <= samples.len() {
                    samples[number - 1].sync = true;
                }
            }
        }
        Err(_) => {
            for s in &mut samples {
                s.sync = true;
            }
        }
    }
    Ok(samples)
}

/// Sample sizes, from `stsz` (or `stz2`'s common case of a fixed size).
fn read_stsz(file: &[u8], stbl: BoxRange) -> Result<Vec<usize>> {
    let stsz = find_box(file, stbl.start, stbl.end, b"stsz")
        .ok_or_else(|| Error::bitstream("no stsz box in the sample table"))?;
    if stsz.start + 12 > stsz.end {
        return Err(Error::bitstream("stsz is truncated"));
    }
    let uniform = u32::from_be_bytes(file[stsz.start + 4..stsz.start + 8].try_into().unwrap());
    let count = u32::from_be_bytes(file[stsz.start + 8..stsz.start + 12].try_into().unwrap()) as usize;
    if uniform != 0 {
        return Ok(vec![uniform as usize; count]);
    }
    let mut sizes = Vec::with_capacity(count);
    let mut at = stsz.start + 12;
    for _ in 0..count {
        if at + 4 > stsz.end {
            return Err(Error::bitstream("stsz ends before its sample count"));
        }
        sizes.push(u32::from_be_bytes(file[at..at + 4].try_into().unwrap()) as usize);
        at += 4;
    }
    Ok(sizes)
}

/// Chunk offsets, from `stco` (32-bit) or `co64` (64-bit).
fn read_chunk_offsets(file: &[u8], stbl: BoxRange) -> Result<Vec<usize>> {
    if let Ok(entries) = read_table(file, stbl, b"stco", 4) {
        return Ok(entries
            .iter()
            .map(|e| u32::from_be_bytes(e[0..4].try_into().unwrap()) as usize)
            .collect());
    }
    let entries = read_table(file, stbl, b"co64", 8)?;
    Ok(entries
        .iter()
        .map(|e| u64::from_be_bytes(e[0..8].try_into().unwrap()) as usize)
        .collect())
}

/// Read a full-box table of fixed-size entries: version+flags, count, then the entries.
fn read_table<'a>(
    file: &'a [u8],
    stbl: BoxRange,
    kind: &[u8; 4],
    entry_size: usize,
) -> Result<Vec<&'a [u8]>> {
    let b = find_box(file, stbl.start, stbl.end, kind)
        .ok_or_else(|| Error::bitstream(format!("no {} box", String::from_utf8_lossy(kind))))?;
    if b.start + 8 > b.end {
        return Err(Error::bitstream(format!(
            "{} is truncated",
            String::from_utf8_lossy(kind)
        )));
    }
    let count = u32::from_be_bytes(file[b.start + 4..b.start + 8].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count);
    let mut at = b.start + 8;
    for _ in 0..count {
        if at + entry_size > b.end {
            return Err(Error::bitstream(format!(
                "{} ends before its entry count",
                String::from_utf8_lossy(kind)
            )));
        }
        out.push(&file[at..at + entry_size]);
        at += entry_size;
    }
    Ok(out)
}

/// How many samples chunk `number` (1-based) holds, from the run-length `stsc` table.
fn samples_per_chunk(stsc: &[&[u8]], number: usize) -> usize {
    let mut answer = 0usize;
    for entry in stsc {
        let first = u32::from_be_bytes(entry[0..4].try_into().unwrap()) as usize;
        let per = u32::from_be_bytes(entry[4..8].try_into().unwrap()) as usize;
        if first <= number {
            answer = per;
        } else {
            break;
        }
    }
    answer
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a box: size, type, payload.
    fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    fn full_box(kind: &[u8; 4], entries: &[&[u8]]) -> Vec<u8> {
        let mut payload = vec![0u8; 4];
        payload.extend_from_slice(&(entries.len() as u32).to_be_bytes());
        for e in entries {
            payload.extend_from_slice(e);
        }
        mp4_box(kind, &payload)
    }

    /// A one-track file with two samples, hand-built so the reader is tested against a
    /// layout this crate did not also generate with the same assumptions.
    fn tiny_file() -> (Vec<u8>, Vec<u8>) {
        let sample_a = [0xaa_u8; 10];
        let sample_b = [0xbb_u8; 6];

        let avcc = AvcC {
            profile_idc: 66,
            profile_compat: 0xc0,
            level_idc: 30,
            length_size: 4,
            sps: vec![vec![0x67, 0x42, 0xc0, 0x1e, 0x11]],
            pps: vec![vec![0x68, 0xce, 0x3c, 0x80]],
        };
        let mut avc1_payload = vec![0u8; 78];
        avc1_payload[24] = 0; // width high byte
        avc1_payload[25] = 64;
        avc1_payload[26] = 0;
        avc1_payload[27] = 48;
        avc1_payload.extend_from_slice(&mp4_box(b"avcC", &avcc.to_bytes()));
        let avc1 = mp4_box(b"avc1", &avc1_payload);

        let mut stsd_payload = vec![0u8; 4];
        stsd_payload.extend_from_slice(&1u32.to_be_bytes());
        stsd_payload.extend_from_slice(&avc1);
        let stsd = mp4_box(b"stsd", &stsd_payload);

        let stts = full_box(b"stts", &[&[0, 0, 0, 2, 0, 0, 0x03, 0xe8]]); // 2 samples, 1000 ticks
        let stsc = full_box(b"stsc", &[&[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1]]); // chunk 1: 2 samples
        let mut stsz_payload = vec![0u8; 4];
        stsz_payload.extend_from_slice(&0u32.to_be_bytes()); // not uniform
        stsz_payload.extend_from_slice(&2u32.to_be_bytes());
        stsz_payload.extend_from_slice(&(sample_a.len() as u32).to_be_bytes());
        stsz_payload.extend_from_slice(&(sample_b.len() as u32).to_be_bytes());
        let stsz = mp4_box(b"stsz", &stsz_payload);
        let stss = full_box(b"stss", &[&[0, 0, 0, 1]]); // sample 1 is a sync sample

        // The media data goes first so the chunk offset is known before the header is built.
        let mut file = Vec::new();
        let mdat_payload: Vec<u8> = sample_a.iter().chain(sample_b.iter()).copied().collect();
        file.extend_from_slice(&mp4_box(b"mdat", &mdat_payload));
        let chunk_offset = 8u32; // the mdat payload starts right after its header

        let stco = full_box(b"stco", &[&chunk_offset.to_be_bytes()]);
        let mut stbl_payload = Vec::new();
        for part in [&stsd, &stts, &stsc, &stsz, &stco, &stss] {
            stbl_payload.extend_from_slice(part);
        }
        let stbl = mp4_box(b"stbl", &stbl_payload);
        let minf = mp4_box(b"minf", &stbl);

        let mut mdhd_payload = vec![0u8; 4 + 8];
        mdhd_payload.extend_from_slice(&90_000u32.to_be_bytes()); // timescale
        mdhd_payload.extend_from_slice(&0u32.to_be_bytes()); // duration
        let mdhd = mp4_box(b"mdhd", &mdhd_payload);

        let mut hdlr_payload = vec![0u8; 8];
        hdlr_payload.extend_from_slice(b"vide");
        hdlr_payload.extend_from_slice(&[0u8; 12]);
        let hdlr = mp4_box(b"hdlr", &hdlr_payload);

        let mut mdia_payload = Vec::new();
        for part in [&mdhd, &hdlr, &minf] {
            mdia_payload.extend_from_slice(part);
        }
        let mdia = mp4_box(b"mdia", &mdia_payload);
        let trak = mp4_box(b"trak", &mdia);
        let moov = mp4_box(b"moov", &trak);
        file.extend_from_slice(&moov);
        (file, mdat_payload)
    }

    #[test]
    fn reads_a_tracks_samples_and_parameter_sets() {
        let (file, media) = tiny_file();
        let track = read_h264_track(&file).expect("the file has an H.264 track");
        assert_eq!(track.timescale, 90_000);
        assert_eq!((track.width, track.height), (64, 48));
        assert_eq!(track.avcc.profile_idc, 66);
        assert_eq!(track.avcc.length_size, 4);
        assert_eq!(track.samples.len(), 2);
        assert_eq!(track.samples[0].dts, 0);
        assert_eq!(track.samples[1].dts, 1000);
        assert!(track.samples[0].sync && !track.samples[1].sync);
        assert_eq!(track.sample_data(&file, 0).unwrap(), &media[..10]);
        assert_eq!(track.sample_data(&file, 1).unwrap(), &media[10..]);
    }

    #[test]
    fn timestamps_convert_to_microseconds() {
        let (file, _) = tiny_file();
        let track = read_h264_track(&file).unwrap();
        // 1000 ticks at 90 kHz is 11.111 ms.
        assert_eq!(track.to_micros(1000), 11_111);
    }

    #[test]
    fn a_file_with_no_video_track_is_reported_not_guessed() {
        let moov = mp4_box(b"moov", &mp4_box(b"trak", &[]));
        let err = read_h264_track(&moov).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err}");
    }
}
