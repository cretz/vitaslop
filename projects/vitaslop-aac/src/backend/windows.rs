//! Windows: the Media Foundation AAC decoder MFT.
//!
//! Media Foundation is COM and the decoder is a Transform: AAC access units go in as
//! `IMFSample`s, 16-bit PCM comes out. It is a SYNCHRONOUS transform - unlike the video
//! decoders, where the hardware ones are asynchronous - so `submit` can decode and this
//! backend never owes an answer for longer than one call.
//!
//! # The one thing the decoder cannot infer
//!
//! An AAC frame does not carry its own configuration; the `AudioSpecificConfig` from the
//! container does. Media Foundation takes it as the input type's USER DATA, laid out as
//! `HEAACWAVEINFO`'s tail (payload type, profile-level, struct type, two reserved fields)
//! followed by the config bytes. Getting that wrong does not fail loudly - the MFT accepts
//! the type and then produces silence or noise - so the layout is written out field by
//! field below rather than assembled from a struct definition.

use std::mem::ManuallyDrop;

use windows::Win32::Media::MediaFoundation::*;
use windows::core::Interface;

use super::Backend;
use crate::error::{Error, Result};
use crate::{DecoderConfig, Pcm};

/// Sample times are in 100ns units.
const TICKS_PER_SECOND: i64 = 10_000_000;

/// The Media Foundation backend.
pub struct MediaFoundationAac {
    transform: IMFTransform,
    channels: u32,
    sample_rate: u32,
    /// Decoded frames waiting for [`Backend::poll`], oldest first.
    ready: std::collections::VecDeque<Pcm>,
    /// Bytes the MFT wants an output sample to hold.
    output_sample_size: u32,
    streaming: bool,
    /// Scratch for the bytes of one access unit, so a submit does not allocate.
    input_scratch: Vec<u8>,
}

// SAFETY: an `IMFTransform` is a COM object living in the process's multithreaded
// apartment; what COM forbids is two threads using one interface AT ONCE, and this type is
// never shared between threads - it is MOVED to whichever thread drives the decoder, which
// is what `Send` means.
unsafe impl Send for MediaFoundationAac {}

impl MediaFoundationAac {
    pub fn new(config: &DecoderConfig) -> Result<MediaFoundationAac> {
        startup()?;
        // What the config SAYS beats what the container said: a track header can disagree
        // with the elementary stream, and the decoder is configured from the stream.
        let parsed = crate::AudioSpecificConfig::parse(&config.asc);
        let channels = parsed
            .map(|p| p.channels)
            .filter(|&c| c > 0)
            .unwrap_or(config.channels)
            .max(1);
        let sample_rate =
            parsed.map(|p| p.sample_rate).filter(|&r| r > 0).unwrap_or(config.sample_rate);
        if sample_rate == 0 {
            return Err(Error::Stream(
                "the stream declares no sample rate and the container gave none".to_string(),
            ));
        }
        let transform = enumerate_decoder()?;
        let input = input_type(&config.asc, channels, sample_rate)?;
        // SAFETY: a media type built above, on a transform just created.
        unsafe { transform.SetInputType(0, &input, 0) }
            .map_err(|e| mf_err("IMFTransform::SetInputType", e))?;
        let output = output_type(channels, sample_rate)?;
        // SAFETY: as above.
        unsafe { transform.SetOutputType(0, &output, 0) }
            .map_err(|e| mf_err("IMFTransform::SetOutputType", e))?;
        // SAFETY: the transform has both types set, which is when this is legal.
        let info = unsafe {
            transform
                .GetOutputStreamInfo(0)
                .map_err(|e| mf_err("IMFTransform::GetOutputStreamInfo", e))?
        };
        let mut me = MediaFoundationAac {
            transform,
            channels,
            sample_rate,
            ready: std::collections::VecDeque::new(),
            // A floor as well as the MFT's own answer: one AAC frame is 1024 samples per
            // channel and a decoder is entitled to ask for nothing and then fill it.
            output_sample_size: info.cbSize.max(1024 * 2 * channels * 2),
            streaming: false,
            input_scratch: Vec::new(),
        };
        me.message(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)?;
        me.message(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)?;
        me.streaming = true;
        Ok(me)
    }

    fn message(&self, message: MFT_MESSAGE_TYPE, param: usize) -> Result<()> {
        // SAFETY: a message on a transform this type owns.
        unsafe { self.transform.ProcessMessage(message, param) }
            .map_err(|e| mf_err("IMFTransform::ProcessMessage", e))
    }

    /// Drain everything the MFT will give up right now into `ready`.
    fn drain(&mut self) -> Result<()> {
        loop {
            let sample = alloc_sample(self.output_sample_size)?;
            let mut buffers = [MFT_OUTPUT_DATA_BUFFER {
                dwStreamID: 0,
                pSample: ManuallyDrop::new(Some(sample)),
                dwStatus: 0,
                pEvents: ManuallyDrop::new(None),
            }];
            let mut status = 0u32;
            // SAFETY: one output buffer, holding a sample allocated just above.
            let hr = unsafe { self.transform.ProcessOutput(0, &mut buffers, &mut status) };
            let produced = ManuallyDrop::take_option(&mut buffers[0].pSample);
            match hr {
                Ok(()) => {}
                Err(e) if e.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => return Ok(()),
                // The output type has to be renegotiated - which for this decoder means it
                // has read the stream and knows better than the container did.
                Err(e) if e.code() == MF_E_TRANSFORM_STREAM_CHANGE => {
                    self.renegotiate()?;
                    continue;
                }
                Err(e) => return Err(mf_err("IMFTransform::ProcessOutput", e)),
            }
            let Some(sample) = produced else { return Ok(()) };
            let pcm = self.read_sample(&sample)?;
            self.ready.push_back(pcm);
        }
    }

    /// Take the negotiated output type again after the MFT reports a stream change.
    fn renegotiate(&mut self) -> Result<()> {
        // SAFETY: called only on MF_E_TRANSFORM_STREAM_CHANGE, which is when a new
        // available type exists.
        let available = unsafe { self.transform.GetOutputAvailableType(0, 0) }
            .map_err(|e| mf_err("IMFTransform::GetOutputAvailableType", e))?;
        // SAFETY: the type just returned by the transform.
        unsafe { self.transform.SetOutputType(0, &available, 0) }
            .map_err(|e| mf_err("IMFTransform::SetOutputType", e))?;
        // SAFETY: reading two attributes of a media type.
        unsafe {
            if let Ok(c) = available.GetUINT32(&MF_MT_AUDIO_NUM_CHANNELS) {
                self.channels = c.max(1);
            }
            if let Ok(r) = available.GetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND) {
                self.sample_rate = r;
            }
        }
        Ok(())
    }

    /// Copy one decoded sample out as interleaved signed-16.
    fn read_sample(&self, sample: &IMFSample) -> Result<Pcm> {
        // SAFETY: a sample the transform just produced; the buffer stays locked until
        // Unlock, and nothing else touches it in between.
        unsafe {
            let buffer = sample
                .ConvertToContiguousBuffer()
                .map_err(|e| mf_err("IMFSample::ConvertToContiguousBuffer", e))?;
            let mut data: *mut u8 = std::ptr::null_mut();
            let mut len = 0u32;
            buffer
                .Lock(&mut data, None, Some(&mut len))
                .map_err(|e| mf_err("IMFMediaBuffer::Lock", e))?;
            let bytes = std::slice::from_raw_parts(data, len as usize);
            let mut samples = Vec::with_capacity(bytes.len() / 2);
            for pair in bytes.chunks_exact(2) {
                samples.push(i16::from_le_bytes([pair[0], pair[1]]));
            }
            let _ = buffer.Unlock();
            let pts = sample.GetSampleTime().unwrap_or(0);
            Ok(Pcm {
                channels: self.channels,
                sample_rate: self.sample_rate,
                pts: pts * self.sample_rate as i64 / TICKS_PER_SECOND,
                samples,
            })
        }
    }
}

impl Backend for MediaFoundationAac {
    fn submit(&mut self, es: &[u8], pts: i64) -> Result<()> {
        self.input_scratch.clear();
        self.input_scratch.extend_from_slice(es);
        let sample = alloc_sample(self.input_scratch.len() as u32)?;
        // SAFETY: a sample allocated just above with at least this capacity; the buffer is
        // locked for the copy and unlocked immediately after.
        unsafe {
            let buffer = sample
                .GetBufferByIndex(0)
                .map_err(|e| mf_err("IMFSample::GetBufferByIndex", e))?;
            let mut data: *mut u8 = std::ptr::null_mut();
            buffer.Lock(&mut data, None, None).map_err(|e| mf_err("IMFMediaBuffer::Lock", e))?;
            std::ptr::copy_nonoverlapping(
                self.input_scratch.as_ptr(),
                data,
                self.input_scratch.len(),
            );
            let _ = buffer.Unlock();
            buffer
                .SetCurrentLength(self.input_scratch.len() as u32)
                .map_err(|e| mf_err("IMFMediaBuffer::SetCurrentLength", e))?;
            let ticks = if self.sample_rate > 0 {
                pts * TICKS_PER_SECOND / self.sample_rate as i64
            } else {
                0
            };
            let _ = sample.SetSampleTime(ticks);
        }
        // SAFETY: a sample built above, on stream 0 of this transform.
        match unsafe { self.transform.ProcessInput(0, &sample, 0) } {
            Ok(()) => {}
            Err(e) if e.code() == MF_E_NOTACCEPTING => {
                // The MFT holds output it will not take more input past. Draining is the
                // whole remedy, and then the input goes in.
                self.drain()?;
                // SAFETY: as above.
                unsafe { self.transform.ProcessInput(0, &sample, 0) }
                    .map_err(|e| mf_err("IMFTransform::ProcessInput", e))?;
            }
            Err(e) => return Err(mf_err("IMFTransform::ProcessInput", e)),
        }
        self.drain()
    }

    fn poll(&mut self) -> Result<Option<Pcm>> {
        if self.ready.is_empty() {
            self.drain()?;
        }
        Ok(self.ready.pop_front())
    }

    fn reset(&mut self) -> Result<()> {
        self.ready.clear();
        if self.streaming {
            self.message(MFT_MESSAGE_COMMAND_FLUSH, 0)?;
            self.message(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)?;
        }
        Ok(())
    }

    fn describe(&self) -> String {
        format!("MediaFoundation AAC {} ch @ {} Hz", self.channels, self.sample_rate)
    }
}

impl Drop for MediaFoundationAac {
    fn drop(&mut self) {
        if self.streaming {
            let _ = self.message(MFT_MESSAGE_NOTIFY_END_STREAMING, 0);
        }
        // SAFETY: paired with this instance's MFStartup. Media Foundation reference counts
        // the pair, so other users in the process are unaffected.
        unsafe {
            let _ = MFShutdown();
        }
    }
}

/// `MFStartup`, once per backend instance (it is reference counted).
fn startup() -> Result<()> {
    // SAFETY: no preconditions beyond a version matching the headers compiled against.
    unsafe { MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET) }.map_err(|e| {
        Error::NoDecoder(format!("MFStartup failed ({e}): Media Foundation is not available"))
    })
}

/// Find a synchronous AAC decoder MFT and activate it.
fn enumerate_decoder() -> Result<IMFTransform> {
    let input = MFT_REGISTER_TYPE_INFO { guidMajorType: MFMediaType_Audio, guidSubtype: MFAudioFormat_AAC };
    let output =
        MFT_REGISTER_TYPE_INFO { guidMajorType: MFMediaType_Audio, guidSubtype: MFAudioFormat_PCM };
    // SAFETY: an enumeration call with two type descriptions and no attributes.
    let activates = unsafe {
        let mut list: *mut Option<IMFActivate> = std::ptr::null_mut();
        let mut count = 0u32;
        MFTEnumEx(
            MFT_CATEGORY_AUDIO_DECODER,
            MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_LOCALMFT | MFT_ENUM_FLAG_SORTANDFILTER,
            Some(&input),
            Some(&output),
            &mut list,
            &mut count,
        )
        .map_err(|e| mf_err("MFTEnumEx", e))?;
        if count == 0 || list.is_null() {
            return Err(Error::NoDecoder("no Media Foundation AAC decoder is registered".into()));
        }
        std::slice::from_raw_parts(list, count as usize).to_vec()
    };
    for activate in activates.into_iter().flatten() {
        // SAFETY: an activate object returned by the enumeration above.
        if let Ok(transform) = unsafe { activate.ActivateObject::<IMFTransform>() } {
            return Ok(transform);
        }
    }
    Err(Error::NoDecoder("every registered AAC decoder refused to activate".into()))
}

/// The input media type: AAC, with the stream's own config as user data.
fn input_type(asc: &[u8], channels: u32, sample_rate: u32) -> Result<IMFMediaType> {
    // SAFETY: creating a media type and setting attributes on it.
    unsafe {
        let t = MFCreateMediaType().map_err(|e| mf_err("MFCreateMediaType", e))?;
        t.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)
            .and_then(|()| t.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_AAC))
            .and_then(|()| t.SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, channels))
            .and_then(|()| t.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, sample_rate))
            .and_then(|()| t.SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16))
            .map_err(|e| mf_err("IMFMediaType::Set", e))?;
        // >>> THE USER DATA, LAID OUT BY HAND.
        //
        // This is `HEAACWAVEINFO` from `wPayloadType` on, and every field matters:
        //   payload type 0     = raw AAC access units (1 would be ADTS, 2 ADIF, 3 LOAS)
        //   profile-level 0x29 = AAC-LC level 2, which is what a Vita movie is
        //   struct type 0      = the `AudioSpecificConfig` follows
        // then the config bytes themselves. A decoder given the wrong payload type accepts
        // the media type and produces nothing useful, which is why this is not guessed.
        let mut user_data: Vec<u8> = Vec::with_capacity(12 + asc.len());
        user_data.extend_from_slice(&0u16.to_le_bytes()); // wPayloadType
        user_data.extend_from_slice(&0x29u16.to_le_bytes()); // wAudioProfileLevelIndication
        user_data.extend_from_slice(&0u16.to_le_bytes()); // wStructType
        user_data.extend_from_slice(&0u16.to_le_bytes()); // wReserved1
        user_data.extend_from_slice(&0u32.to_le_bytes()); // dwReserved2
        user_data.extend_from_slice(asc);
        t.SetBlob(&MF_MT_USER_DATA, &user_data)
            .map_err(|e| mf_err("IMFMediaType::SetBlob(MF_MT_USER_DATA)", e))?;
        Ok(t)
    }
}

/// The output media type: interleaved 16-bit PCM at the stream's rate.
fn output_type(channels: u32, sample_rate: u32) -> Result<IMFMediaType> {
    let block_align = channels * 2;
    // SAFETY: creating a media type and setting attributes on it.
    unsafe {
        let t = MFCreateMediaType().map_err(|e| mf_err("MFCreateMediaType", e))?;
        t.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Audio)
            .and_then(|()| t.SetGUID(&MF_MT_SUBTYPE, &MFAudioFormat_PCM))
            .and_then(|()| t.SetUINT32(&MF_MT_AUDIO_NUM_CHANNELS, channels))
            .and_then(|()| t.SetUINT32(&MF_MT_AUDIO_SAMPLES_PER_SECOND, sample_rate))
            .and_then(|()| t.SetUINT32(&MF_MT_AUDIO_BITS_PER_SAMPLE, 16))
            .and_then(|()| t.SetUINT32(&MF_MT_AUDIO_BLOCK_ALIGNMENT, block_align))
            .and_then(|()| {
                t.SetUINT32(&MF_MT_AUDIO_AVG_BYTES_PER_SECOND, block_align * sample_rate)
            })
            .and_then(|()| t.SetUINT32(&MF_MT_ALL_SAMPLES_INDEPENDENT, 1))
            .map_err(|e| mf_err("IMFMediaType::Set", e))?;
        Ok(t)
    }
}

/// One `IMFSample` holding one buffer of `size` bytes.
fn alloc_sample(size: u32) -> Result<IMFSample> {
    // SAFETY: two allocation calls with no preconditions.
    unsafe {
        let sample = MFCreateSample().map_err(|e| mf_err("MFCreateSample", e))?;
        let buffer =
            MFCreateMemoryBuffer(size.max(1)).map_err(|e| mf_err("MFCreateMemoryBuffer", e))?;
        sample.AddBuffer(&buffer).map_err(|e| mf_err("IMFSample::AddBuffer", e))?;
        Ok(sample)
    }
}

/// Turn a COM error into this crate's, keeping the HRESULT.
fn mf_err(what: &'static str, e: windows::core::Error) -> Error {
    Error::platform(what, e.code().0, e.message())
}

/// `ManuallyDrop<Option<T>>::take`, which the standard library does not offer directly.
trait TakeOption<T> {
    fn take_option(this: &mut ManuallyDrop<Option<T>>) -> Option<T>;
}

impl<T> TakeOption<T> for ManuallyDrop<Option<T>> {
    fn take_option(this: &mut ManuallyDrop<Option<T>>) -> Option<T> {
        // SAFETY: the value is not used again through the `ManuallyDrop`; it is replaced
        // with `None` so the wrapper still holds a valid value.
        unsafe { ManuallyDrop::take(this) }
    }
}
