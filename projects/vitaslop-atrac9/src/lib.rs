//! A faithful Rust port of the MIT-licensed LibAtrac9 decoder by Alex Barney
//! (<https://github.com/Thealexbarney/LibAtrac9>), covering the ATRAC9 decode path
//! the Vita's NGS synthesizer uses. It is pure compute with no dependencies, so it
//! builds identically for native and wasm.
//!
//! Usage: build a decoder from the 4-byte config word carried in an AT9 stream's
//! RIFF `fmt ` chunk, then decode one ATRAC9 frame at a time into interleaved
//! signed-16 PCM.
//!
//! ```ignore
//! let mut dec = Atrac9Decoder::new(config_data)?;
//! let mut pcm = vec![0i16; dec.frame_samples() * dec.channels()];
//! let used = dec.decode_frame(&frame_bytes, &mut pcm)?;
//! ```
//!
//! Original work Copyright (c) 2018 Alex Barney, MIT License. This port carries the
//! same terms.

mod bandext;
mod bit_reader;
mod config;
mod decoder;
mod generated_huffman;
mod huffman;
mod mdct;
mod tables;

use config::Config;
use decoder::{DecodeCtx, Frame, GradientCurves};
use huffman::Codebooks;
use tables::{Trig, Windows};

/// A decode error. Variants mirror the LibAtrac9 status codes that the decode path
/// can return; all indicate a malformed or unsupported bitstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    BadConfigData,
    UnpackSuperframeFlagInvalid,
    UnpackReuseBandParamsInvalid,
    UnpackBandParamsInvalid,
    UnpackGradBoundaryInvalid,
    UnpackGradStartUnitOob,
    UnpackGradEndUnitOob,
    UnpackGradEndUnitInvalid,
    UnpackGradStartValueOob,
    UnpackGradEndValueOob,
    UnpackScaleFactorModeInvalid,
    UnpackScaleFactorOob,
    UnpackExtensionDataInvalid,
    /// The output slice was smaller than `frame_samples() * channels()`.
    OutputTooSmall,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "atrac9 decode error: {self:?}")
    }
}

impl std::error::Error for Error {}

/// Codec geometry read from the config word (LibAtrac9 `CodecInfo`).
#[derive(Debug, Clone, Copy)]
pub struct CodecInfo {
    pub channels: i32,
    pub channel_config_index: i32,
    pub sample_rate: i32,
    pub superframe_size: i32,
    pub frames_in_superframe: i32,
    pub frame_samples: i32,
    pub config_data: [u8; 4],
}

/// A stateful ATRAC9 decoder for one stream configuration. Holds the per-channel
/// MDCT overlap and delta-coding history, so frames must be decoded in order.
pub struct Atrac9Decoder {
    config: Config,
    frame: Frame,
    /// Bytes consumed so far by the frames of the CURRENT superframe - what lets
    /// [`decode_frame`](Self::decode_frame) charge the superframe's tail padding to its
    /// last frame. See that method for the measurement that made this load-bearing.
    superframe_seen_bytes: usize,
    // >>> THE FOUR CONSTANT TABLES ARE SHARED, NOT REBUILT PER DECODER.
    //
    // None of these depends on the stream config - `Codebooks::new`, `Trig::generate`,
    // `Windows::generate` and `generate_gradient_curves` all take no arguments - yet a
    // decoder used to build its own set of all four. A decoder is created per VOICE START,
    // and one racing title starts **3,224 voices in a race**, so that is 3,224 constructions
    // of the same Huffman tables, the same sin/cos tables and the same window curves, each
    // one a burst of allocation on the audio thread. A V8 worker profile put 0.50% of the
    // whole thread inside `Atrac9Decoder::new`, with `dlmalloc` another 2.6% overall.
    //
    // `OnceLock` rather than `lazy_static`/`thread_local`: the crate is dependency-free by
    // design and builds identically for native and wasm, and these are immutable plain data
    // after construction, so one shared copy is sound on either.
    codebooks: &'static Codebooks,
    trig: &'static Trig,
    windows: &'static Windows,
    gradient_curves: &'static GradientCurves,
}

/// The shared codebooks - see the fields of [`Atrac9Decoder`].
fn codebooks() -> &'static Codebooks {
    static IT: std::sync::OnceLock<Codebooks> = std::sync::OnceLock::new();
    IT.get_or_init(Codebooks::new)
}

/// The shared MDCT trig tables.
fn trig() -> &'static Trig {
    static IT: std::sync::OnceLock<Box<Trig>> = std::sync::OnceLock::new();
    IT.get_or_init(Trig::generate)
}

/// The shared MDCT window curves.
fn windows() -> &'static Windows {
    static IT: std::sync::OnceLock<Box<Windows>> = std::sync::OnceLock::new();
    IT.get_or_init(Windows::generate)
}

/// The shared gradient curves.
fn gradient_curves() -> &'static GradientCurves {
    static IT: std::sync::OnceLock<Box<GradientCurves>> = std::sync::OnceLock::new();
    IT.get_or_init(decoder::generate_gradient_curves)
}

impl Atrac9Decoder {
    /// Build a decoder from the 4-byte ATRAC9 config word.
    pub fn new(config_data: [u8; 4]) -> Result<Atrac9Decoder, Error> {
        let config = Config::parse(config_data)?;
        let frame = Frame::new(&config);
        Ok(Atrac9Decoder {
            config,
            frame,
            superframe_seen_bytes: 0,
            codebooks: codebooks(),
            trig: trig(),
            windows: windows(),
            gradient_curves: gradient_curves(),
        })
    }

    /// Interleaved output channel count.
    pub fn channels(&self) -> usize {
        self.config.channel_count as usize
    }

    /// PCM samples per channel produced by one [`decode_frame`](Self::decode_frame).
    pub fn frame_samples(&self) -> usize {
        self.config.frame_samples as usize
    }

    /// Bytes in one superframe (the caller's outer stride over frames).
    pub fn superframe_bytes(&self) -> usize {
        self.config.superframe_bytes as usize
    }

    /// Frames packed into one superframe.
    pub fn frames_per_superframe(&self) -> usize {
        self.config.frames_per_superframe as usize
    }

    /// Codec geometry, for callers that need the RIFF-independent view.
    pub fn info(&self) -> CodecInfo {
        CodecInfo {
            channels: self.config.channel_count,
            channel_config_index: self.config.channel_config_index,
            sample_rate: self.config.sample_rate,
            superframe_size: self.config.superframe_bytes,
            frames_in_superframe: self.config.frames_per_superframe,
            frame_samples: self.config.frame_samples,
            config_data: self.config.config_data,
        }
    }

    /// Decode one ATRAC9 frame from `input` into `out` (interleaved signed-16,
    /// length must be at least `frame_samples() * channels()`). Returns the number
    /// of input bytes this frame accounts for in the stream.
    ///
    /// # The superframe's tail padding belongs to its LAST frame
    /// A superframe is a FIXED-size container (`superframe_bytes`) holding
    /// `frames_per_superframe` variable-length byte-aligned frames, with any leftover as
    /// padding at the end. A caller that advances its read cursor by this return value -
    /// which is exactly what a retail title does with `sceAudiodecDecode`'s
    /// `INPUT_ES_SIZE` - must land on the next superframe boundary after the last frame,
    /// or the next decode starts inside the padding. MEASURED on that title: four streams,
    /// each failing `UnpackBandParamsInvalid` exactly once, always at a superframe-first
    /// ordinal, always after a superframe whose frames' bit-consumed sizes summed one byte
    /// short of the container (e.g. 118+110+109+110 = 447 of 448) - and one stream whose
    /// "corrupt frame" was the padding bytes `01 01 01 01` themselves. So the raw
    /// `bits / 8` count is reported for frames WITHIN a superframe, and the distance to
    /// the container boundary for the frame that CLOSES one.
    pub fn decode_frame(&mut self, input: &[u8], out: &mut [i16]) -> Result<usize, Error> {
        let needed = self.frame_samples() * self.channels();
        if out.len() < needed {
            return Err(Error::OutputTooSmall);
        }

        let ctx = DecodeCtx {
            config: &self.config,
            codebooks: self.codebooks,
            trig: self.trig,
            windows: self.windows,
            gradient_curves: self.gradient_curves,
        };
        let mut br = bit_reader::BitReader::new(input);
        // Whether THIS frame closes its superframe: `decoder::decode_frame` advances the
        // index and wraps it to 0 on the last frame of the container.
        decoder::decode_frame(&ctx, &mut self.frame, &mut br)?;
        let closed_superframe = self.frame.index_in_superframe == 0;
        decoder::pcm_float_to_short(&self.config, &self.frame, out);
        let raw = br.position / 8;
        Ok(if closed_superframe {
            let seen = std::mem::take(&mut self.superframe_seen_bytes);
            // The container has `superframe_bytes - seen` left; this frame accounts for
            // all of it, its own bytes plus the tail padding. `max(raw)` is a guard for a
            // stream whose frames overrun their own container - malformed, but the honest
            // answer there is still the bytes actually read, never a negative-padding lie.
            (self.config.superframe_bytes as usize - seen).max(raw)
        } else {
            self.superframe_seen_bytes += raw;
            raw
        })
    }
}

#[cfg(test)]
mod tests;
