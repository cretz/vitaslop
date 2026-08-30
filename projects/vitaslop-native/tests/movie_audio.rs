//! A movie's sound, end to end: this engine's demuxer, this engine's audio seam, and a
//! reference decoder's PCM to check the result against.
//!
//! # What this covers that the crate-level oracle does not
//!
//! `vitaslop-aac`'s own test feeds an ADTS stream it parses itself. This one goes the way a
//! title's audio really travels: the container is read by `vitaslop_runtime::mp4`, the
//! `AudioSpecificConfig` comes out of the track's `esds` descriptors, the samples come from
//! the sample table, and the decoder is opened through the platform seam the engine
//! installs. A defect in any of those - a mis-read descriptor, an off-by-one sample table,
//! the wrong track - shows up here and in no other test.
//!
//! Env-gated on a retail movie, which cannot be committed:
//!
//! ```text
//! VITASLOP_MOVIE=<path to a .mp4 with an AAC track>
//! VITASLOP_AAC_DIR=<dir holding ref.pcm for that movie>
//! cargo test --release -p vitaslop-native --test movie_audio -- --nocapture
//! ```
//!
//! `ref.pcm` is `ffmpeg -i <movie> -vn -f s16le -acodec pcm_s16le ref.pcm`.

use vitaslop_platform::audio_dec::{AacFactory, AudioDecodeFactory, AudioStream};
use vitaslop_runtime::mp4::{Mp4, TrackKind};

#[test]
fn a_movies_sound_track_decodes_to_what_the_reference_decoder_produces() {
    let (Ok(movie), Ok(dir)) =
        (std::env::var("VITASLOP_MOVIE"), std::env::var("VITASLOP_AAC_DIR"))
    else {
        eprintln!("VITASLOP_MOVIE / VITASLOP_AAC_DIR are not set; skipping");
        return;
    };
    let data = std::fs::read(&movie).expect("the movie file");
    let reference = std::fs::read(std::path::Path::new(&dir).join("ref.pcm")).expect("ref.pcm");
    let reference: Vec<i16> =
        reference.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();

    let mp4 = Mp4::parse(&data).expect("the container parses");
    let track = mp4.track(TrackKind::Audio).expect("the movie has an audio track");
    assert_eq!(&track.codec, b"mp4a", "the track is AAC");
    let asc = track.audio_specific_config();
    assert!(!asc.is_empty(), "the esds carries an AudioSpecificConfig");
    eprintln!(
        "track {}: {} samples, timescale {}, asc {asc:02x?}",
        track.id,
        track.samples.len(),
        track.timescale
    );

    let mut decoder = match AacFactory.open_aac(&AudioStream {
        asc,
        channels: 0,
        sample_rate: track.timescale,
    }) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no AAC decoder on this host ({e}); skipping");
            return;
        }
    };
    eprintln!("decoder: {}", decoder.describe());

    let mut got: Vec<i16> = Vec::with_capacity(reference.len());
    let mut channels = 0u32;
    for (i, sample) in track.samples.iter().enumerate() {
        let at = sample.offset as usize;
        let es = &data[at..at + sample.size as usize];
        decoder.submit(es, i as i64 * 1024).expect("submit");
        while let Some(pcm) = decoder.poll().expect("poll") {
            channels = pcm.channels;
            got.extend_from_slice(&pcm.samples);
        }
    }
    assert!(channels > 0, "the decoder produced no audio at all");
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
    let mine = &got[best_shift * step..best_shift * step + compare];
    let peak = mine
        .iter()
        .zip(&reference[..compare])
        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).unsigned_abs())
        .max()
        .unwrap_or(0);
    eprintln!(
        "{} samples compared, alignment {best_shift} frame(s), RMS {best_err:.1}, peak |diff| {peak}",
        compare
    );
    // The same bounds the crate's own oracle uses, and for the same reason: AAC is
    // specified in floating point, so two conforming decoders agree closely rather than
    // exactly, and an RMS bound alone would pass a decode with one loud click in it.
    assert!(best_err < 64.0, "RMS error {best_err:.1} against the reference decoder");
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
