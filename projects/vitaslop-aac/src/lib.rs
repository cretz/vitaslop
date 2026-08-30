//! Platform-native AAC decoding, behind one API.
//!
//! A movie's sound track is AAC, the Vita decodes it in `SceAudiodec`, and every host this
//! engine runs on already has an AAC decoder - Media Foundation on Windows, WebCodecs in a
//! browser. This crate reaches whichever one is there and hands back interleaved signed-16
//! PCM, which is what the guest's own audio path wants.
//!
//! # Submit and poll, not decode
//!
//! In a browser the decoded audio arrives on a callback that only runs when the worker
//! returns to the event loop, so no call here can promise "the PCM for this frame" on
//! return. The seam is therefore the same one the video path uses: SUBMIT an access unit,
//! POLL for whatever has come back. A caller that needs the PCM synchronously - and the
//! guest's `sceAudiodecDecode` does - gets it by running AHEAD, which it can, because the
//! stream comes out of a container it is already demultiplexing.
//!
//! ```ignore
//! let mut dec = Decoder::new(DecoderConfig { asc, channels: 2, sample_rate: 48_000 })?;
//! dec.submit(access_unit, pts)?;
//! while let Some(pcm) = dec.poll()? { /* pcm.samples is interleaved i16 */ }
//! ```

mod backend;
mod error;

pub use error::{Error, Result};

/// What the stream is, as the container describes it.
#[derive(Clone, Debug)]
pub struct DecoderConfig {
    /// The `AudioSpecificConfig` from the track's `esds` - two bytes for plain AAC-LC.
    /// Every backend needs it: it carries the object type, the sample-rate index and the
    /// channel configuration, and a decoder configured without it has to guess all three.
    pub asc: Vec<u8>,
    /// Channels the container declares. A hint: what the DECODER reports wins, and
    /// [`Pcm`] carries that.
    pub channels: u32,
    /// Sample rate the container declares, in Hz. Also a hint, for the same reason.
    pub sample_rate: u32,
}

/// One decoded frame of audio: interleaved signed-16, native endianness.
#[derive(Clone, Debug, Default)]
pub struct Pcm {
    pub channels: u32,
    pub sample_rate: u32,
    /// The timestamp the access unit was submitted with, handed back unchanged.
    pub pts: i64,
    /// Interleaved samples: `frames * channels` of them.
    pub samples: Vec<i16>,
}

impl Pcm {
    /// Frames (per-channel samples) in this buffer.
    pub fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }
}

/// A decoder for one AAC stream.
pub struct Decoder {
    inner: Box<dyn backend::Backend>,
    /// Access units submitted and not yet answered. A decoder is pipelined, so this is
    /// normally non-zero while playing - it is how [`Decoder::owes_frames`] tells "still
    /// working" from "nothing to wait for".
    outstanding: usize,
}

impl Decoder {
    /// Open a decoder for `config`, or report why this host cannot.
    pub fn new(config: DecoderConfig) -> Result<Decoder> {
        Ok(Decoder { inner: backend::open(&config)?, outstanding: 0 })
    }

    /// Hand over one access unit - one AAC frame, exactly as the container stores it.
    pub fn submit(&mut self, es: &[u8], pts: i64) -> Result<()> {
        self.inner.submit(es, pts)?;
        self.outstanding += 1;
        Ok(())
    }

    /// Take the next decoded frame, if one has arrived.
    pub fn poll(&mut self) -> Result<Option<Pcm>> {
        let out = self.inner.poll()?;
        if out.is_some() {
            self.outstanding = self.outstanding.saturating_sub(1);
        }
        Ok(out)
    }

    /// True while frames are still owed for input already given. Always a "may yet
    /// arrive", never a promise: a decoder is entitled to answer one input with none.
    pub fn owes_frames(&self) -> bool {
        self.outstanding > 0
    }

    /// Discard everything held, for a seek or a loop.
    pub fn reset(&mut self) -> Result<()> {
        self.outstanding = 0;
        self.inner.reset()
    }

    /// Which decoder this turned out to be, for the one line a run says about its audio.
    pub fn describe(&self) -> String {
        self.inner.describe()
    }
}

/// The `AudioSpecificConfig` fields this crate reads, for a caller that has the bytes and
/// wants to know what they say before opening anything.
///
/// Two things make this worth parsing here rather than trusting the container: a track
/// header can disagree with the config the decoder is actually given, and a browser needs
/// the sample rate and channel count in its `configure()` call as numbers, not as bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    /// MPEG-4 Audio Object Type. 2 is AAC-LC, which is what a Vita movie carries.
    pub object_type: u8,
    pub sample_rate: u32,
    pub channels: u32,
}

/// The sampling frequencies an `AudioSpecificConfig` index can name. Index 15 means the
/// rate follows as a 24-bit literal instead.
const ASC_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025,
    8_000, 7_350,
];

impl AudioSpecificConfig {
    /// Read the first bits of an `AudioSpecificConfig`: 5 bits of object type, 4 of
    /// sample-rate index (with an escape), 4 of channel configuration.
    pub fn parse(asc: &[u8]) -> Option<AudioSpecificConfig> {
        let mut bits = Bits { data: asc, at: 0 };
        let mut object_type = bits.take(5)? as u8;
        if object_type == 31 {
            object_type = 32 + bits.take(6)? as u8;
        }
        let index = bits.take(4)? as usize;
        let sample_rate = match index {
            15 => bits.take(24)?,
            i => *ASC_RATES.get(i)?,
        };
        let channels = match bits.take(4)? {
            // 0 means the channel setup is described by a program config element the
            // decoder has to read for itself; the caller's container hint stands.
            0 => 0,
            // 7 is 7.1 in the tables, which no Vita movie carries and which this does not
            // pretend to know the layout of.
            7 => 8,
            c => c,
        };
        Some(AudioSpecificConfig { object_type, sample_rate, channels })
    }
}

/// A most-significant-bit-first reader over the config bytes.
struct Bits<'a> {
    data: &'a [u8],
    at: usize,
}

impl Bits<'_> {
    fn take(&mut self, n: usize) -> Option<u32> {
        let mut out = 0u32;
        for _ in 0..n {
            let byte = *self.data.get(self.at >> 3)?;
            out = (out << 1) | u32::from((byte >> (7 - (self.at & 7))) & 1);
            self.at += 1;
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The config a Vita movie's AAC track actually carries, and the two escapes.
    #[test]
    fn an_audio_specific_config_reads_as_the_standard_lays_it_out() {
        // 0x11 0x90: object type 2 (AAC-LC), rate index 3 (48000), 2 channels.
        assert_eq!(
            AudioSpecificConfig::parse(&[0x11, 0x90]),
            Some(AudioSpecificConfig { object_type: 2, sample_rate: 48_000, channels: 2 })
        );
        // 0x12 0x08: object type 2, rate index 4 (44100), 1 channel.
        assert_eq!(
            AudioSpecificConfig::parse(&[0x12, 0x08]),
            Some(AudioSpecificConfig { object_type: 2, sample_rate: 44_100, channels: 1 })
        );
        // Truncated is None rather than a guess.
        assert_eq!(AudioSpecificConfig::parse(&[0x11]), None);
    }
}
