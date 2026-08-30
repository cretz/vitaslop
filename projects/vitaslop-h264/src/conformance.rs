//! The conformance suite, as a function - so it can be run somewhere other than `cargo test`.
//!
//! # Why this is not just a test file
//!
//! The checks here were a `tests/` binary, which means they only ever ran on a machine with
//! a Rust toolchain and a native target. The WebCodecs backend therefore shipped having
//! never executed at all, and its first three runs in a browser found three separate
//! defects - a `RefCell` re-entered from a decoder callback, a `copyTo` that refuses an
//! explicit format, and a frame layout that has to be read back rather than requested.
//! Every one of them would have been a red line here in seconds.
//!
//! So the body lives in the crate, is `async` (which costs a native caller nothing - see
//! [`crate::Decoder::receive_async`]), and takes no assets: the stream comes from
//! [`crate::synth`], built out of `I_PCM` macroblocks whose decoded pixels are knowable
//! from the bitstream itself. That is what makes byte equality legitimate against four
//! different platform decoders with no reference implementation in the picture.
//!
//! A machine with NO decoder is not a failure. "This container has no video hardware" is a
//! normal outcome, reported as such - see [`Report::decoder`].

use crate::bitstream::{AuSplitter, avcc};
use crate::error::Result;
use crate::{Decoder, DecoderConfig, Frame, InputFormat, Packet, synth};

/// What one named check did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The check ran and the decoder was right.
    Passed,
    /// The check ran and the decoder was wrong, with what was wrong about it.
    Failed(String),
}

/// The result of a whole run.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// The backend that ran, or `None` when this machine has no decoder at all.
    pub decoder: Option<String>,
    /// Every check, in the order it ran.
    pub cases: Vec<(String, Outcome)>,
}

impl Report {
    /// True when every check that ran passed. A run with no decoder passes vacuously - it
    /// is the caller's business whether that is acceptable where it is running.
    pub fn ok(&self) -> bool {
        self.cases.iter().all(|(_, o)| *o == Outcome::Passed)
    }

    /// The checks that failed, one line each.
    pub fn failures(&self) -> Vec<String> {
        self.cases
            .iter()
            .filter_map(|(name, o)| match o {
                Outcome::Failed(why) => Some(format!("{name}: {why}")),
                Outcome::Passed => None,
            })
            .collect()
    }

    /// One line per case plus a summary, for a log or a page.
    pub fn text(&self) -> String {
        let mut out = match &self.decoder {
            Some(name) => format!("decoder: {name}\n"),
            None => "decoder: NONE on this machine - nothing was checked\n".to_string(),
        };
        for (name, outcome) in &self.cases {
            match outcome {
                Outcome::Passed => out.push_str(&format!("  ok    {name}\n")),
                Outcome::Failed(why) => out.push_str(&format!("  FAIL  {name}: {why}\n")),
            }
        }
        let failed = self.failures().len();
        out.push_str(&format!("{} of {} checks passed\n", self.cases.len() - failed, self.cases.len()));
        out
    }

    fn record(&mut self, name: &str, r: std::result::Result<(), String>) {
        let outcome = match r {
            Ok(()) => Outcome::Passed,
            Err(why) => Outcome::Failed(why),
        };
        self.cases.push((name.to_string(), outcome));
    }
}

/// Run every check. `Err` only for something that stopped the suite itself; a decoder that
/// is merely WRONG comes back as a failed case in the report.
pub async fn run() -> Result<Report> {
    let mut report = Report::default();
    // One decoder first, purely to find out whether this machine has one and what it is.
    match Decoder::new(DecoderConfig::default()) {
        Ok(d) => report.decoder = Some(d.backend_name().to_string()),
        Err(e) if e.is_missing_decoder() => return Ok(report),
        Err(e) => return Err(e),
    }

    let r = byte_exact_pictures().await;
    report.record("pictures are byte-exact and in presentation order", r);
    let r = timestamps_survive().await;
    report.record("caller timestamps survive the round trip", r);
    let r = survives_a_reset().await;
    report.record("a stream survives a reset", r);
    let r = cropping_is_applied().await;
    report.record("cropping is applied to what the caller sees", r);
    let r = length_prefixed().await;
    report.record("length-prefixed samples with out-of-band parameter sets", r);
    Ok(report)
}

/// Decode a whole Annex B stream, waiting for asynchronous backends.
async fn decode_all(stream: &[u8], config: DecoderConfig) -> Result<Vec<Frame>> {
    let mut decoder = Decoder::new(config)?;
    let mut frames = Vec::new();
    decoder.send(Packet::new(stream))?;
    while let Some(frame) = decoder.receive_async().await? {
        frames.push(frame);
    }
    decoder.finish()?;
    while let Some(frame) = decoder.receive_async().await? {
        frames.push(frame);
    }
    Ok(frames)
}

/// Say WHERE two pictures differ rather than dumping both.
fn same_picture(actual: &[u8], expected: &[u8], what: &str) -> std::result::Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("{what}: {} bytes, expected {}", actual.len(), expected.len()));
    }
    let Some(first) = actual.iter().zip(expected).position(|(a, b)| a != b) else {
        return Ok(());
    };
    let differing = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
    Err(format!(
        "{what}: {differing} of {} samples differ, first at offset {first} (got {}, expected {})",
        expected.len(),
        actual[first],
        expected[first]
    ))
}

async fn byte_exact_pictures() -> std::result::Result<(), String> {
    // 128x96, five pictures: big enough that no decoder refuses the size, small enough that
    // the whole stream is a few hundred kilobytes of raw samples.
    let stream = synth::synthesize(8, 6, 5);
    let frames = decode_all(&stream.annex_b, DecoderConfig::default())
        .await
        .map_err(|e| e.to_string())?;
    if frames.len() != stream.frames.len() {
        return Err(format!("{} frames for a {}-picture stream", frames.len(), stream.frames.len()));
    }
    let mut actual = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        if (frame.width, frame.height) != (stream.width, stream.height) {
            return Err(format!(
                "frame {index} is {}x{}, expected {}x{}",
                frame.width, frame.height, stream.width, stream.height
            ));
        }
        frame.copy_to_i420(&mut actual);
        same_picture(&actual, &stream.frames[index], &format!("frame {index}"))?;
    }
    let order: Vec<i64> = frames.iter().map(|f| f.pts).collect();
    let mut sorted = order.clone();
    sorted.sort_unstable();
    if order != sorted {
        return Err(format!("frames came out of presentation order: {order:?}"));
    }
    Ok(())
}

async fn timestamps_survive() -> std::result::Result<(), String> {
    let stream = synth::synthesize(4, 4, 3);
    let mut decoder = Decoder::new(DecoderConfig {
        packets_are_access_units: Some(true),
        ..DecoderConfig::default()
    })
    .map_err(|e| e.to_string())?;

    let mut splitter = AuSplitter::new();
    let mut units = Vec::new();
    splitter.push_annex_b(&stream.annex_b, &mut units).map_err(|e| e.to_string())?;
    splitter.finish(&mut units).map_err(|e| e.to_string())?;

    let sent: Vec<i64> = (0..units.len() as i64).map(|i| 1000 + i * 37).collect();
    let mut got = Vec::new();
    for (unit, pts) in units.iter().zip(&sent) {
        decoder.send(Packet::with_pts(&unit.data, *pts)).map_err(|e| e.to_string())?;
        while let Some(f) = decoder.receive_async().await.map_err(|e| e.to_string())? {
            got.push(f.pts);
        }
    }
    decoder.finish().map_err(|e| e.to_string())?;
    while let Some(f) = decoder.receive_async().await.map_err(|e| e.to_string())? {
        got.push(f.pts);
    }
    if got != sent {
        return Err(format!("timestamps came back as {got:?}, sent {sent:?}"));
    }
    Ok(())
}

async fn survives_a_reset() -> std::result::Result<(), String> {
    let stream = synth::synthesize(4, 4, 3);
    let mut decoder = Decoder::new(DecoderConfig::default()).map_err(|e| e.to_string())?;
    decoder.send(Packet::new(&stream.annex_b)).map_err(|e| e.to_string())?;
    while decoder.receive_async().await.map_err(|e| e.to_string())?.is_some() {}
    decoder.reset().map_err(|e| e.to_string())?;

    // After a reset the same stream must decode again, from the top, to the same pixels.
    decoder.send(Packet::new(&stream.annex_b)).map_err(|e| e.to_string())?;
    let mut frames = Vec::new();
    while let Some(f) = decoder.receive_async().await.map_err(|e| e.to_string())? {
        frames.push(f);
    }
    decoder.finish().map_err(|e| e.to_string())?;
    while let Some(f) = decoder.receive_async().await.map_err(|e| e.to_string())? {
        frames.push(f);
    }
    if frames.len() != stream.frames.len() {
        return Err(format!("{} frames after a reset, expected {}", frames.len(), stream.frames.len()));
    }
    let mut actual = Vec::new();
    frames[0].copy_to_i420(&mut actual);
    same_picture(&actual, &stream.frames[0], "frame 0 after a reset")
}

async fn cropping_is_applied() -> std::result::Result<(), String> {
    // Coded 320x240, visible 312x232 - the shape every 1080p stream has, where the coded
    // size is a whole number of macroblocks and the visible one is not.
    let stream = synth::synthesize_cropped(20, 15, 2, 4, 4);
    let frames = decode_all(&stream.annex_b, DecoderConfig::default())
        .await
        .map_err(|e| e.to_string())?;
    let Some(frame) = frames.first() else {
        return Err("no frames".to_string());
    };
    if (frame.width, frame.height) != (stream.width, stream.height) {
        return Err(format!(
            "a cropped frame came back {}x{}, expected the VISIBLE {}x{}",
            frame.width, frame.height, stream.width, stream.height
        ));
    }
    let mut actual = Vec::new();
    frame.copy_to_i420(&mut actual);
    same_picture(&actual, &stream.frames[0], "the cropped frame")
}

async fn length_prefixed() -> std::result::Result<(), String> {
    // The MP4 shape: parameter sets out of band in an avcC record, samples length-prefixed.
    let stream = synth::synthesize(4, 4, 3);
    let mut splitter = AuSplitter::new();
    let mut units = Vec::new();
    splitter.push_annex_b(&stream.annex_b, &mut units).map_err(|e| e.to_string())?;
    splitter.finish(&mut units).map_err(|e| e.to_string())?;
    let record = avcc::AvcC::from_parameter_sets(
        splitter.sets.sps_nals(),
        splitter.sets.pps_nals(),
        4,
    )
    .map_err(|e| e.to_string())?;

    let mut decoder = Decoder::new(DecoderConfig {
        input: InputFormat::LengthPrefixed { length_size: 4 },
        extradata: Some(record.to_bytes()),
        packets_are_access_units: Some(true),
        ..DecoderConfig::default()
    })
    .map_err(|e| e.to_string())?;

    let mut frames = Vec::new();
    let mut sample = Vec::new();
    for unit in &units {
        sample.clear();
        avcc::annex_b_to_length_prefixed(&unit.data, 4, &mut sample);
        decoder.send(Packet::new(&sample)).map_err(|e| e.to_string())?;
        while let Some(f) = decoder.receive_async().await.map_err(|e| e.to_string())? {
            frames.push(f);
        }
    }
    decoder.finish().map_err(|e| e.to_string())?;
    while let Some(f) = decoder.receive_async().await.map_err(|e| e.to_string())? {
        frames.push(f);
    }
    if frames.len() != stream.frames.len() {
        return Err(format!("{} frames, expected {}", frames.len(), stream.frames.len()));
    }
    let mut actual = Vec::new();
    frames[0].copy_to_i420(&mut actual);
    same_picture(&actual, &stream.frames[0], "frame 0 of a length-prefixed stream")
}
