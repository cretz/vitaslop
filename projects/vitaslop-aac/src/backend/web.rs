//! The browser: WebCodecs `AudioDecoder`.
//!
//! The decoder answers on a CALLBACK, which only runs when the worker returns to the event
//! loop - so this backend queues whatever has arrived and `poll` hands it over. That is the
//! whole reason the seam is submit/poll rather than decode.
//!
//! `AudioData.copyTo`, unlike `VideoFrame.copyTo`, is synchronous, so a frame is copied out
//! inside the output callback and nothing here has to await anything.

use std::cell::RefCell;
use std::rc::Rc;

use js_sys::{ArrayBuffer, Float32Array, Int16Array, Object, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use super::Backend;
use crate::error::{Error, Result};
use crate::{DecoderConfig, Pcm};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = AudioDecoder)]
    type JsAudioDecoder;

    #[wasm_bindgen(constructor, js_class = "AudioDecoder", catch)]
    fn new(init: &Object) -> std::result::Result<JsAudioDecoder, JsValue>;

    #[wasm_bindgen(method, js_class = "AudioDecoder", js_name = configure, catch)]
    fn configure(this: &JsAudioDecoder, config: &Object) -> std::result::Result<(), JsValue>;

    #[wasm_bindgen(method, js_class = "AudioDecoder", js_name = decode, catch)]
    fn decode(this: &JsAudioDecoder, chunk: &JsEncodedAudioChunk) -> std::result::Result<(), JsValue>;

    #[wasm_bindgen(method, js_class = "AudioDecoder", js_name = reset, catch)]
    fn reset(this: &JsAudioDecoder) -> std::result::Result<(), JsValue>;

    #[wasm_bindgen(method, js_class = "AudioDecoder", js_name = close, catch)]
    fn close(this: &JsAudioDecoder) -> std::result::Result<(), JsValue>;

    #[wasm_bindgen(js_name = EncodedAudioChunk)]
    type JsEncodedAudioChunk;

    #[wasm_bindgen(constructor, js_class = "EncodedAudioChunk", catch)]
    fn new(init: &Object) -> std::result::Result<JsEncodedAudioChunk, JsValue>;

    #[wasm_bindgen(js_name = AudioData)]
    type JsAudioData;

    #[wasm_bindgen(method, getter, js_class = "AudioData", js_name = format)]
    fn format(this: &JsAudioData) -> Option<String>;

    #[wasm_bindgen(method, getter, js_class = "AudioData", js_name = sampleRate)]
    fn sample_rate(this: &JsAudioData) -> f64;

    #[wasm_bindgen(method, getter, js_class = "AudioData", js_name = numberOfFrames)]
    fn number_of_frames(this: &JsAudioData) -> u32;

    #[wasm_bindgen(method, getter, js_class = "AudioData", js_name = numberOfChannels)]
    fn number_of_channels(this: &JsAudioData) -> u32;

    #[wasm_bindgen(method, getter, js_class = "AudioData", js_name = timestamp)]
    fn timestamp(this: &JsAudioData) -> f64;

    #[wasm_bindgen(method, js_class = "AudioData", js_name = allocationSize, catch)]
    fn allocation_size(this: &JsAudioData, options: &Object) -> std::result::Result<u32, JsValue>;

    #[wasm_bindgen(method, js_class = "AudioData", js_name = copyTo, catch)]
    fn copy_to(
        this: &JsAudioData,
        destination: &JsValue,
        options: &Object,
    ) -> std::result::Result<(), JsValue>;

    #[wasm_bindgen(method, js_class = "AudioData", js_name = close)]
    fn close(this: &JsAudioData);
}

/// Shared between the backend and the callbacks it installs in JS.
#[derive(Default)]
struct Shared {
    ready: std::collections::VecDeque<Pcm>,
    /// The first error the decoder reported. Sticky: WebCodecs closes itself on error.
    error: Option<String>,
    /// What the first decoded frame turned out to be - the format string it reported and
    /// the copy layout that worked. Not knowable before it runs, and worth having on the
    /// record when a device produces silence.
    layout: Option<String>,
}

/// The WebCodecs backend.
pub struct WebCodecsAac {
    decoder: JsAudioDecoder,
    shared: Rc<RefCell<Shared>>,
    /// Kept alive for as long as the decoder: dropping a closure detaches its JS function.
    _on_output: Closure<dyn FnMut(JsAudioData)>,
    _on_error: Closure<dyn FnMut(JsValue)>,
    channels: u32,
    sample_rate: u32,
}

// SAFETY: wasm here is single-threaded - the whole emulator, including its scheduler, runs
// on one worker - so nothing is ever sent anywhere. The bound exists because the trait is
// shared with the native backends, which really are moved between threads.
unsafe impl Send for WebCodecsAac {}

impl WebCodecsAac {
    pub fn new(config: &DecoderConfig) -> Result<WebCodecsAac> {
        let parsed = crate::AudioSpecificConfig::parse(&config.asc);
        let channels =
            parsed.map(|p| p.channels).filter(|&c| c > 0).unwrap_or(config.channels).max(1);
        let sample_rate =
            parsed.map(|p| p.sample_rate).filter(|&r| r > 0).unwrap_or(config.sample_rate);
        let object_type = parsed.map(|p| p.object_type).unwrap_or(2);
        if sample_rate == 0 {
            return Err(Error::Stream(
                "the stream declares no sample rate and the container gave none".to_string(),
            ));
        }

        let shared = Rc::new(RefCell::new(Shared::default()));
        let out_shared = shared.clone();
        let on_output = Closure::wrap(Box::new(move |data: JsAudioData| {
            let taken = read_audio_data(&data);
            data.close();
            let mut s = out_shared.borrow_mut();
            match taken {
                Ok((pcm, layout)) => {
                    if s.layout.is_none() {
                        s.layout = Some(layout);
                    }
                    s.ready.push_back(pcm);
                }
                Err(e) => {
                    s.error.get_or_insert(e.to_string());
                }
            }
        }) as Box<dyn FnMut(JsAudioData)>);
        let err_shared = shared.clone();
        let on_error = Closure::wrap(Box::new(move |e: JsValue| {
            err_shared.borrow_mut().error.get_or_insert(describe(&e));
        }) as Box<dyn FnMut(JsValue)>);

        let init = Object::new();
        set(&init, "output", on_output.as_ref())?;
        set(&init, "error", on_error.as_ref())?;
        let decoder = JsAudioDecoder::new(&init).map_err(|e| {
            Error::NoDecoder(format!("this browser has no WebCodecs AudioDecoder ({})", describe(&e)))
        })?;

        let cfg = Object::new();
        // `mp4a.40.<object type>` is how an MPEG-4 audio stream names itself; 2 is AAC-LC.
        set(&cfg, "codec", &JsValue::from_str(&format!("mp4a.40.{object_type}")))?;
        set(&cfg, "sampleRate", &JsValue::from_f64(sample_rate as f64))?;
        set(&cfg, "numberOfChannels", &JsValue::from_f64(channels as f64))?;
        // The `AudioSpecificConfig`, without which the decoder has to guess the stream's
        // shape - and Chrome refuses to configure a raw AAC stream at all.
        let description = Uint8Array::new_with_length(config.asc.len() as u32);
        description.copy_from(&config.asc);
        set(&cfg, "description", &description)?;
        decoder
            .configure(&cfg)
            .map_err(|e| Error::Stream(format!("AudioDecoder.configure: {}", describe(&e))))?;

        Ok(WebCodecsAac {
            decoder,
            shared,
            _on_output: on_output,
            _on_error: on_error,
            channels,
            sample_rate,
        })
    }
}

impl Backend for WebCodecsAac {
    fn submit(&mut self, es: &[u8], pts: i64) -> Result<()> {
        if let Some(e) = self.shared.borrow_mut().error.take() {
            return Err(Error::Stream(e));
        }
        let data = Uint8Array::new_with_length(es.len() as u32);
        data.copy_from(es);
        let init = Object::new();
        // Every AAC-LC frame is independently decodable, so every chunk is a key frame.
        set(&init, "type", &JsValue::from_str("key"))?;
        // Microseconds, which is what WebCodecs timestamps are.
        let micros = if self.sample_rate > 0 {
            pts.saturating_mul(1_000_000) / self.sample_rate as i64
        } else {
            0
        };
        set(&init, "timestamp", &JsValue::from_f64(micros as f64))?;
        set(&init, "data", &data)?;
        let chunk = JsEncodedAudioChunk::new(&init)
            .map_err(|e| Error::platform("new EncodedAudioChunk", 0, describe(&e)))?;
        self.decoder
            .decode(&chunk)
            .map_err(|e| Error::platform("AudioDecoder.decode", 0, describe(&e)))
    }

    fn poll(&mut self) -> Result<Option<Pcm>> {
        let mut s = self.shared.borrow_mut();
        if let Some(pcm) = s.ready.pop_front() {
            return Ok(Some(pcm));
        }
        match s.error.take() {
            Some(e) => Err(Error::Stream(e)),
            None => Ok(None),
        }
    }

    fn reset(&mut self) -> Result<()> {
        {
            let mut s = self.shared.borrow_mut();
            s.ready.clear();
            s.error = None;
        }
        // >>> `reset()` ALSO UNCONFIGURES A WebCodecs DECODER - the same trap the video
        // path hit, where a seek or a loop left the decoder unusable and every later
        // `decode` threw "Cannot call 'decode' on an unconfigured codec". There is nothing
        // to flush here that dropping the queued frames does not cover, so this does not
        // call it at all.
        Ok(())
    }

    fn describe(&self) -> String {
        let layout = self.shared.borrow().layout.clone().unwrap_or_else(|| "not yet decoded".into());
        format!("WebCodecs AAC {} ch @ {} Hz - {layout}", self.channels, self.sample_rate)
    }
}

impl Drop for WebCodecsAac {
    fn drop(&mut self) {
        let _ = self.decoder.close();
    }
}

/// Copy one `AudioData` out as interleaved signed-16.
///
/// # The format ladder, and why it is a ladder
///
/// `copyTo` may convert, and may refuse to: which conversions a browser implements is not
/// specified. Interleaved `s16` is exactly what the guest wants, so it is asked for first;
/// interleaved `f32` is the one every implementation has; and the last rung reads the
/// decoder's own PLANAR layout and interleaves here, which needs no conversion support at
/// all. Whichever rung worked is reported, because on a device this is the difference
/// between silence and sound and nothing else would say which.
fn read_audio_data(data: &JsAudioData) -> Result<(Pcm, String)> {
    let channels = data.number_of_channels().max(1);
    let frames = data.number_of_frames();
    let sample_rate = data.sample_rate() as u32;
    let reported = data.format().unwrap_or_default();
    let pts = (data.timestamp() * sample_rate as f64 / 1_000_000.0) as i64;
    let total = frames as usize * channels as usize;

    // Rung 1: interleaved signed-16, straight into the shape the guest wants.
    if let Ok(samples) = copy_interleaved_s16(data, total) {
        return Ok((
            Pcm { channels, sample_rate, pts, samples },
            format!("{reported} -> s16 interleaved"),
        ));
    }
    // Rung 2: interleaved float.
    if let Ok(floats) = copy_interleaved_f32(data, total) {
        return Ok((
            Pcm { channels, sample_rate, pts, samples: floats_to_i16(&floats) },
            format!("{reported} -> f32 interleaved"),
        ));
    }
    // Rung 3: the decoder's own planar float, interleaved here.
    let mut samples = vec![0i16; total];
    for plane in 0..channels {
        let options = Object::new();
        set(&options, "planeIndex", &JsValue::from_f64(plane as f64))
            .map_err(|e| Error::Stream(e.to_string()))?;
        set(&options, "format", &JsValue::from_str("f32-planar"))
            .map_err(|e| Error::Stream(e.to_string()))?;
        let dest = Float32Array::new_with_length(frames);
        data.copy_to(&dest, &options).map_err(|e| {
            Error::platform("AudioData.copyTo", 0, format!("planar f32: {}", describe(&e)))
        })?;
        let mut plane_samples = vec![0f32; frames as usize];
        dest.copy_to(&mut plane_samples);
        for (i, v) in plane_samples.iter().enumerate() {
            samples[i * channels as usize + plane as usize] = float_to_i16(*v);
        }
    }
    Ok((Pcm { channels, sample_rate, pts, samples }, format!("{reported} -> f32 planar, interleaved here")))
}

fn copy_interleaved_s16(data: &JsAudioData, total: usize) -> std::result::Result<Vec<i16>, JsValue> {
    let options = Object::new();
    Reflect::set(&options, &JsValue::from_str("planeIndex"), &JsValue::from_f64(0.0))?;
    Reflect::set(&options, &JsValue::from_str("format"), &JsValue::from_str("s16"))?;
    let dest = Int16Array::new_with_length(total as u32);
    data.copy_to(&dest, &options)?;
    let mut samples = vec![0i16; total];
    dest.copy_to(&mut samples);
    Ok(samples)
}

fn copy_interleaved_f32(data: &JsAudioData, total: usize) -> std::result::Result<Vec<f32>, JsValue> {
    let options = Object::new();
    Reflect::set(&options, &JsValue::from_str("planeIndex"), &JsValue::from_f64(0.0))?;
    Reflect::set(&options, &JsValue::from_str("format"), &JsValue::from_str("f32"))?;
    let dest = Float32Array::new_with_length(total as u32);
    data.copy_to(&dest, &options)?;
    let mut samples = vec![0f32; total];
    dest.copy_to(&mut samples);
    Ok(samples)
}

fn floats_to_i16(v: &[f32]) -> Vec<i16> {
    v.iter().map(|s| float_to_i16(*s)).collect()
}

/// Float to signed-16 the way every audio path does it: full scale is 32767, and the
/// clamp is not optional because a decoder may hand back samples slightly outside [-1, 1].
fn float_to_i16(v: f32) -> i16 {
    (v * 32767.0).clamp(-32768.0, 32767.0) as i16
}

fn set(target: &Object, key: &str, value: &JsValue) -> Result<()> {
    Reflect::set(target, &JsValue::from_str(key), value)
        .map(|_| ())
        .map_err(|e| Error::platform("Reflect::set", 0, describe(&e)))
}

/// A JS error as a string, whatever shape it came in.
fn describe(e: &JsValue) -> String {
    if let Some(s) = e.as_string() {
        return s;
    }
    if let Some(err) = e.dyn_ref::<js_sys::Error>() {
        return format!("{}: {}", String::from(err.name()), String::from(err.message()));
    }
    format!("{e:?}")
}

/// Unused on this target, but the trait's `ArrayBuffer` import keeps the js-sys feature set
/// honest about what this file touches.
#[allow(dead_code)]
fn _array_buffer_marker(_: &ArrayBuffer) {}
