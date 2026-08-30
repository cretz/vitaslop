//! Decode a real AAC stream and check it against a reference decoder's PCM.
//!
//! # Why the oracle is outside the repo
//!
//! The only AAC streams to hand are a retail game's, which cannot be committed, so this
//! test is env-gated: it runs when `VITASLOP_AAC_DIR` names a directory holding
//!
//! - `ref.aac` - an ADTS stream (`ffmpeg -i movie.mp4 -vn -c:a copy ref.aac`), and
//! - `ref.pcm` - the same stream decoded to interleaved signed-16
//!   (`ffmpeg -i movie.mp4 -vn -f s16le -acodec pcm_s16le ref.pcm`),
//!
//! and reports that it was skipped otherwise. ffmpeg is a BLACK-BOX oracle here: it is run
//! as a program and its output bytes are compared against, which is the same relationship
//! this project has with a real console.
//!
//! # Why the comparison has a tolerance and an alignment search
//!
//! AAC decoding is specified in floating point and two conforming decoders are not required
//! to produce identical samples; the standard's own conformance criterion is an error bound,
//! not equality. And a decoder is free to emit the codec's start-up delay as leading samples
//! or to swallow it, which shifts one output against the other by whole frames. So this
//! finds the alignment that fits best, reports it, and then holds the two to an RMS bound -
//! and to a MAXIMUM bound, because RMS alone would pass a decode that is right on average
//! and wrong in one loud place.

use vitaslop_aac::{Decoder, DecoderConfig};

/// One ADTS frame: where its payload starts and ends, and what its header says the stream
/// is. The payload is what a raw-AAC decoder is given; the header is not part of it.
struct Adts {
    payload: std::ops::Range<usize>,
    object_type: u8,
    rate_index: u8,
    channels: u8,
}

/// Walk an ADTS stream. Returns `None` at the first thing that is not a frame, which for a
/// file produced by a demuxer means the end.
fn next_adts(data: &[u8], at: usize) -> Option<Adts> {
    let h = data.get(at..at + 7)?;
    // Syncword: 12 bits of ones.
    if h[0] != 0xFF || h[1] & 0xF0 != 0xF0 {
        return None;
    }
    let protection_absent = h[1] & 1 == 1;
    let object_type = ((h[2] >> 6) & 0x03) + 1;
    let rate_index = (h[2] >> 2) & 0x0F;
    let channels = ((h[2] & 0x01) << 2) | ((h[3] >> 6) & 0x03);
    let length = (usize::from(h[3] & 0x03) << 11) | (usize::from(h[4]) << 3) | (usize::from(h[5]) >> 5);
    // A CRC, when present, sits between the header and the payload.
    let header = if protection_absent { 7 } else { 9 };
    if length < header || at + length > data.len() {
        return None;
    }
    Some(Adts { payload: at + header..at + length, object_type, rate_index, channels })
}

/// The `AudioSpecificConfig` an ADTS header describes: 5 bits of object type, 4 of
/// sample-rate index, 4 of channel configuration, then zero padding.
fn asc_from_adts(a: &Adts) -> Vec<u8> {
    let bits = (u16::from(a.object_type) << 11)
        | (u16::from(a.rate_index) << 7)
        | (u16::from(a.channels) << 3);
    bits.to_be_bytes().to_vec()
}

#[test]
fn a_real_stream_decodes_to_what_the_reference_decoder_produces() {
    let Ok(dir) = std::env::var("VITASLOP_AAC_DIR") else {
        eprintln!("VITASLOP_AAC_DIR is not set; skipping the AAC oracle");
        return;
    };
    let dir = std::path::Path::new(&dir);
    let stream = std::fs::read(dir.join("ref.aac")).expect("ref.aac");
    let reference = std::fs::read(dir.join("ref.pcm")).expect("ref.pcm");
    let reference: Vec<i16> =
        reference.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();

    let first = next_adts(&stream, 0).expect("ref.aac starts with an ADTS frame");
    let asc = asc_from_adts(&first);
    let channels = u32::from(first.channels).max(1);
    let mut decoder = match Decoder::new(DecoderConfig {
        asc,
        channels,
        sample_rate: 0,
    }) {
        Ok(d) => d,
        Err(e) if e.is_missing_decoder() => {
            eprintln!("no AAC decoder on this host ({e}); skipping");
            return;
        }
        Err(e) => panic!("opening the decoder: {e}"),
    };
    eprintln!("decoder: {}", decoder.describe());

    let mut got: Vec<i16> = Vec::with_capacity(reference.len());
    let mut at = 0usize;
    let mut frames = 0usize;
    while let Some(frame) = next_adts(&stream, at) {
        at = frame.payload.end;
        decoder.submit(&stream[frame.payload.clone()], frames as i64 * 1024).expect("submit");
        while let Some(pcm) = decoder.poll().expect("poll") {
            got.extend_from_slice(&pcm.samples);
        }
        frames += 1;
    }
    assert!(frames > 100, "the stream should hold more than a moment of audio");
    assert!(!got.is_empty(), "the decoder produced no PCM at all from {frames} frames");

    // >>> THE ALIGNMENT, SEARCHED AND REPORTED. Whole frames only: a decoder either emits
    // the start-up delay or it does not, and anything other than a multiple of 1024 samples
    // per channel would be a bug in one of them rather than a convention.
    let step = 1024 * channels as usize;
    let compare = got.len().min(reference.len()).saturating_sub(4 * step);
    assert!(compare > step, "not enough overlapping audio to compare");
    let (mut best_shift, mut best_err) = (0usize, f64::INFINITY);
    for shift in 0..=4 {
        let off = shift * step;
        if off + compare > got.len() {
            break;
        }
        let err = rms(&got[off..off + compare], &reference[..compare]);
        if err < best_err {
            best_err = err;
            best_shift = shift;
        }
    }
    let off = best_shift * step;
    let mine = &got[off..off + compare];
    let theirs = &reference[..compare];
    let peak = mine
        .iter()
        .zip(theirs)
        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).unsigned_abs())
        .max()
        .unwrap_or(0);
    eprintln!(
        "{frames} frames, {} samples compared, alignment {best_shift} frame(s), RMS {best_err:.1}, \
         peak |diff| {peak}",
        compare
    );
    // Full scale is 32768. An RMS of 64 is about -54 dB against it, which is far below any
    // decoder disagreement that could be heard and far above float rounding.
    assert!(best_err < 64.0, "RMS error {best_err:.1} against the reference decoder");
    // And no single sample may be wildly wrong: an RMS bound alone would pass a decode with
    // one loud click in it.
    assert!(peak < 4096, "a sample differs by {peak} from the reference");
}

fn rms(a: &[i16], b: &[i16]) -> f64 {
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(i32::from(*x) - i32::from(*y));
            d * d
        })
        .sum();
    (sum / a.len().max(1) as f64).sqrt()
}
