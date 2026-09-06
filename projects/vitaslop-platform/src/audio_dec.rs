//! Audio decoding, as the engine sees it.
//!
//! The counterpart of [`crate::video`], and it exists for the same reason: the engine
//! compiles to wasm and links no decoder, so it holds a factory supplied by whichever
//! front-end started it. What crosses this seam is PCM, never a decoder type.
//!
//! # Submit and poll, again
//!
//! In a browser the decoded audio arrives on a callback, so nothing here can promise the
//! PCM for an access unit on return. `sceAudiodecDecode` DOES have to return it, which is
//! why the caller runs ahead: it submits each access unit as the demuxer hands it to the
//! title, and by the time the title asks for that frame's PCM it is already here. See
//! `vitaslop_runtime::vita::audiodec`.

/// One decoded frame of audio: interleaved signed-16.
#[derive(Clone, Debug, Default)]
pub struct DecodedAudio {
    pub channels: u32,
    pub sample_rate: u32,
    /// Interleaved samples: `frames * channels` of them.
    pub samples: Vec<i16>,
}

/// What the stream is. The ENGINE describes it, because the guest's own decoder API tells
/// it (a channel count, a sample rate, whether the frames carry ADTS headers) rather than
/// handing over a config blob.
#[derive(Clone, Debug)]
pub struct AudioStream {
    /// `AudioSpecificConfig` bytes when the container had them; empty when the description
    /// is only the fields below, in which case the implementation synthesises one.
    pub asc: Vec<u8>,
    pub channels: u32,
    pub sample_rate: u32,
}

/// Why a decoder could not be created.
#[derive(Debug)]
pub enum AudioDecodeError {
    /// This machine has no AAC decoder. The caller carries on without sound rather than
    /// failing: a silent movie is still a movie.
    NoDecoder(String),
    /// The stream, or something in it, is not decodable here.
    Stream(String),
}

impl core::fmt::Display for AudioDecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AudioDecodeError::NoDecoder(m) => write!(f, "no audio decoder on this host: {m}"),
            AudioDecodeError::Stream(m) => write!(f, "audio stream cannot be decoded: {m}"),
        }
    }
}

/// Opens decoders. The engine holds one of these rather than a decoder.
pub trait AudioDecodeFactory: Send {
    /// Open a decoder for an AAC stream.
    fn open_aac(&mut self, stream: &AudioStream) -> Result<Box<dyn AudioDecode>, AudioDecodeError>;
}

/// The default: this host decodes no audio.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoAudioDecode;

impl AudioDecodeFactory for NoAudioDecode {
    fn open_aac(&mut self, _stream: &AudioStream) -> Result<Box<dyn AudioDecode>, AudioDecodeError> {
        Err(AudioDecodeError::NoDecoder(
            "this host was started without an audio decoder".to_string(),
        ))
    }
}

/// One stream's decoder.
pub trait AudioDecode: Send {
    /// Submit one access unit, with the sample index it starts at.
    fn submit(&mut self, es: &[u8], pts: i64) -> Result<(), AudioDecodeError>;
    /// Take the next decoded frame, if one has arrived.
    fn poll(&mut self) -> Result<Option<DecodedAudio>, AudioDecodeError>;
    /// Discard everything held, for a loop or a seek.
    fn reset(&mut self) -> Result<(), AudioDecodeError>;
    /// Which decoder this is, for the one line a run says about its audio.
    fn describe(&self) -> String;
}

/// A factory backed by [`vitaslop_aac`], i.e. by whatever decoder the platform has.
#[cfg(feature = "aac")]
#[derive(Debug, Default, Clone, Copy)]
pub struct AacFactory;

#[cfg(feature = "aac")]
impl AudioDecodeFactory for AacFactory {
    fn open_aac(&mut self, stream: &AudioStream) -> Result<Box<dyn AudioDecode>, AudioDecodeError> {
        let asc = if stream.asc.is_empty() {
            synth_asc(stream.channels, stream.sample_rate)
        } else {
            stream.asc.clone()
        };
        let decoder = vitaslop_aac::Decoder::new(vitaslop_aac::DecoderConfig {
            asc,
            channels: stream.channels,
            sample_rate: stream.sample_rate,
        })
        .map_err(|e| {
            if e.is_missing_decoder() {
                AudioDecodeError::NoDecoder(e.to_string())
            } else {
                AudioDecodeError::Stream(e.to_string())
            }
        })?;
        Ok(Box::new(AacDecoder(decoder)))
    }
}

/// >>> THE CONFIG THE GUEST'S API DOES NOT CARRY.
///
/// `SceAudiodecInfoAac` describes a stream as a channel count, a sample rate and two flags -
/// there is no `AudioSpecificConfig` in it - but every host decoder wants one, because that
/// is what says "AAC-LC, this rate, this channel layout". Those three fields are exactly
/// what an `AudioSpecificConfig` holds, so it is built rather than guessed: 5 bits of object
/// type (2, AAC-LC), 4 of sample-rate index, 4 of channel configuration.
#[cfg(feature = "aac")]
fn synth_asc(channels: u32, sample_rate: u32) -> Vec<u8> {
    const RATES: [u32; 13] = [
        96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000,
        11_025, 8_000, 7_350,
    ];
    let index = RATES.iter().position(|&r| r == sample_rate).unwrap_or(3) as u16;
    let bits = (2u16 << 11) | (index << 7) | ((channels.min(7) as u16) << 3);
    bits.to_be_bytes().to_vec()
}

#[cfg(feature = "aac")]
struct AacDecoder(vitaslop_aac::Decoder);

#[cfg(feature = "aac")]
impl AudioDecode for AacDecoder {
    fn submit(&mut self, es: &[u8], pts: i64) -> Result<(), AudioDecodeError> {
        self.0.submit(es, pts).map_err(|e| AudioDecodeError::Stream(e.to_string()))
    }

    fn poll(&mut self) -> Result<Option<DecodedAudio>, AudioDecodeError> {
        match self.0.poll().map_err(|e| AudioDecodeError::Stream(e.to_string()))? {
            Some(pcm) => Ok(Some(DecodedAudio {
                channels: pcm.channels,
                sample_rate: pcm.sample_rate,
                samples: pcm.samples,
            })),
            None => Ok(None),
        }
    }

    fn reset(&mut self) -> Result<(), AudioDecodeError> {
        self.0.reset().map_err(|e| AudioDecodeError::Stream(e.to_string()))
    }

    fn describe(&self) -> String {
        self.0.describe()
    }
}

#[cfg(test)]
mod tests {
    /// The synthesised config has to say what the stream is, or a decoder configured from
    /// it produces noise rather than an error.
    #[cfg(feature = "aac")]
    #[test]
    fn a_synthesised_config_says_aac_lc_at_the_right_rate() {
        let asc = super::synth_asc(2, 48_000);
        let parsed = vitaslop_aac::AudioSpecificConfig::parse(&asc).expect("it parses");
        assert_eq!(parsed.object_type, 2);
        assert_eq!(parsed.sample_rate, 48_000);
        assert_eq!(parsed.channels, 2);
        let mono = super::synth_asc(1, 44_100);
        let parsed = vitaslop_aac::AudioSpecificConfig::parse(&mono).expect("it parses");
        assert_eq!((parsed.sample_rate, parsed.channels), (44_100, 1));
    }
}
