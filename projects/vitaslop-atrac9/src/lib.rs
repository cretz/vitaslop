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
    codebooks: Codebooks,
    trig: Box<Trig>,
    windows: Box<Windows>,
    gradient_curves: Box<GradientCurves>,
}

impl Atrac9Decoder {
    /// Build a decoder from the 4-byte ATRAC9 config word.
    pub fn new(config_data: [u8; 4]) -> Result<Atrac9Decoder, Error> {
        let config = Config::parse(config_data)?;
        let frame = Frame::new(&config);
        Ok(Atrac9Decoder {
            config,
            frame,
            codebooks: Codebooks::new(),
            trig: Trig::generate(),
            windows: Windows::generate(),
            gradient_curves: decoder::generate_gradient_curves(),
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
    /// of input bytes consumed by this frame.
    pub fn decode_frame(&mut self, input: &[u8], out: &mut [i16]) -> Result<usize, Error> {
        let needed = self.frame_samples() * self.channels();
        if out.len() < needed {
            return Err(Error::OutputTooSmall);
        }

        let ctx = DecodeCtx {
            config: &self.config,
            codebooks: &self.codebooks,
            trig: &self.trig,
            windows: &self.windows,
            gradient_curves: &self.gradient_curves,
        };
        let mut br = bit_reader::BitReader::new(input);
        decoder::decode_frame(&ctx, &mut self.frame, &mut br)?;
        decoder::pcm_float_to_short(&self.config, &self.frame, out);
        Ok(br.position / 8)
    }
}

#[cfg(test)]
mod tests;
