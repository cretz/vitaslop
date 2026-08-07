//! The browser's [`AudioSink`]: the PCM producer for the shared ring the page's
//! AudioWorklet consumes.
//!
//! The seam is `vitaslop_runtime::audio::AudioSink`, whose one hard rule is that
//! `submit` must not block - the guest's audio thread calls it once per grain from
//! inside a host call, and stalling there stalls the whole emulator. Writing into a
//! `SharedArrayBuffer` ring satisfies that trivially: it is two `copy_from` calls and an
//! atomic store, and it never waits for the audio device.
//!
//! # Why this cannot just be `postMessage` per grain
//! The emulator runs in a Web Worker; Web Audio is a main-thread API. A grain is a few
//! milliseconds, so a message per grain puts the guest's audio thread behind the main
//! thread's event queue - on a page that is also rendering, that queue is exactly what is
//! congested. The ring puts the PCM where the AUDIO RENDER THREAD can read it directly,
//! and that thread is the only one whose deadline matters.
//!
//! # What it does NOT do
//! It does not stretch, resync or drop to hide the emulator's frame rate. If the guest
//! produces audio more slowly than the device consumes it - which is what a title running
//! below real time does - the worklet emits silence and counts it. That gap is the honest
//! report of "the emulator is not fast enough yet", and papering over it would make a
//! performance problem look like an audio bug.

use vitaslop_runtime::audio::{AudioFormat, AudioSink};
use wasm_bindgen::prelude::*;

/// Control-block layout, in `Int32` slots. Mirrored in `web/audio.js` and
/// `web/audio-worklet.js` - all three must agree.
const CTL_WRITE: u32 = 0;
const CTL_READ: u32 = 1;
const CTL_OVERRUN: u32 = 3;
const CTL_CAPACITY: u32 = 4;
const CTL_CHANNELS: u32 = 5;
const CTL_SAMPLE_RATE: u32 = 6;
const CTL_HEADER_BYTES: u32 = 32;

/// A `sceAudioOut` port as this sink sees it: the format the guest opened it with, which
/// is all that is needed to turn its grains into ring frames.
struct Port {
    id: i32,
    format: AudioFormat,
    /// Per-channel gain in 0..=1, from `sceAudioOutSetVolume` (Vita range 0..=32768).
    gain: [f32; 2],
    /// Fractional read position into the source grain, carried ACROSS grains when the
    /// port's rate differs from the device's. Resetting it per grain would drop or
    /// duplicate a sample at every grain boundary, which is a periodic click at the
    /// grain rate - audible, and easy to misdiagnose as a decoder fault.
    resample_pos: f64,
    /// Whether the rate mismatch has been reported. Once is enough; it cannot change.
    reported_rate: bool,
}

/// Writes guest PCM into the page's shared audio ring.
pub struct WebAudioSink {
    ctl: js_sys::Int32Array,
    data: js_sys::Float32Array,
    capacity: u32,
    channels: u32,
    sample_rate: u32,
    ports: Vec<Port>,
    next_port: i32,
    /// Scratch for one grain of device-rate interleaved f32, reused across submits so a
    /// call at grain rate allocates nothing.
    scratch: Vec<f32>,
}

impl WebAudioSink {
    /// Attach to the ring the page created (`web/audio.js` `startAudio`). `ring` is the
    /// `SharedArrayBuffer`; `None` if it is not one, which is a caller error rather than
    /// something to paper over.
    pub fn new(ring: &JsValue) -> Option<WebAudioSink> {
        let buf: js_sys::SharedArrayBuffer = ring.clone().dyn_into().ok()?;
        let ctl = js_sys::Int32Array::new_with_byte_offset_and_length(
            &buf,
            0,
            CTL_HEADER_BYTES / 4,
        );
        let total_floats =
            (buf.byte_length() - CTL_HEADER_BYTES) / 4;
        let data =
            js_sys::Float32Array::new_with_byte_offset_and_length(&buf, CTL_HEADER_BYTES, total_floats);
        let capacity = ctl.get_index(CTL_CAPACITY) as u32;
        let channels = ctl.get_index(CTL_CHANNELS) as u32;
        let sample_rate = ctl.get_index(CTL_SAMPLE_RATE) as u32;
        if capacity == 0 || channels == 0 || sample_rate == 0 {
            return None;
        }
        Some(WebAudioSink {
            ctl,
            data,
            capacity,
            channels,
            sample_rate,
            ports: Vec::new(),
            next_port: 0,
            scratch: Vec::new(),
        })
    }

    fn port(&mut self, id: i32) -> Option<&mut Port> {
        self.ports.iter_mut().find(|p| p.id == id)
    }

    /// Publish `frames` of interleaved device-rate f32 from `self.scratch`, dropping
    /// whatever will not fit and counting the drop.
    ///
    /// Single producer, single consumer, monotonic counters: the write index is only
    /// advanced by this, the read index only by the worklet, so the free space is a plain
    /// subtraction and no lock is needed. The data goes in BEFORE the index is published,
    /// which is the whole ordering requirement.
    fn publish(&mut self, frames: u32) {
        let write = self.ctl.get_index(CTL_WRITE) as u32;
        let read = js_sys::Atomics::load(&self.ctl, CTL_READ).unwrap_or(0) as u32;
        let free = self.capacity.saturating_sub(write.wrapping_sub(read));
        let take = frames.min(free);
        if take < frames {
            let _ = js_sys::Atomics::add(&self.ctl, CTL_OVERRUN, (frames - take) as i32);
        }
        if take == 0 {
            return;
        }
        // The ring wraps, so at most two contiguous runs.
        let start = write % self.capacity;
        let first = take.min(self.capacity - start);
        let ch = self.channels;
        self.data
            .subarray(start * ch, (start + first) * ch)
            .copy_from(&self.scratch[..(first * ch) as usize]);
        if first < take {
            let rest = take - first;
            self.data
                .subarray(0, rest * ch)
                .copy_from(&self.scratch[(first * ch) as usize..(take * ch) as usize]);
        }
        let _ = js_sys::Atomics::store(&self.ctl, CTL_WRITE, write.wrapping_add(take) as i32);
    }
}

impl AudioSink for WebAudioSink {
    fn open_port(&mut self, format: AudioFormat) -> i32 {
        let id = self.next_port;
        self.next_port += 1;
        web_sys::console::log_1(&JsValue::from_str(&format!(
            "[audio] guest opened a port: {} ch, {} Hz, grain {} (device {} Hz, {} ch)",
            format.channels, format.sample_rate, format.grain, self.sample_rate, self.channels
        )));
        self.ports.push(Port {
            id,
            format,
            gain: [1.0, 1.0],
            resample_pos: 0.0,
            reported_rate: false,
        });
        id
    }

    fn submit(&mut self, port: i32, pcm: &[i16]) {
        let device_rate = self.sample_rate;
        let device_ch = self.channels as usize;
        // Read everything needed off the port and DROP the borrow: the conversion below
        // fills `self.scratch`, which is a sibling field, and the resample cursor is
        // written back once at the end.
        let (src_ch, src_rate, gain, pos0, report_rate) = {
            let Some(p) = self.port(port) else { return };
            let src_rate = p.format.sample_rate;
            let report = src_rate != device_rate && !p.reported_rate;
            if report {
                p.reported_rate = true;
            }
            (p.format.channels.max(1) as usize, src_rate, p.gain, p.resample_pos, report)
        };
        if src_rate == 0 || pcm.len() < src_ch {
            return;
        }
        let src_frames = pcm.len() / src_ch;

        // Rate conversion. The Vita's MAIN/BGM ports are 48 kHz and the page asks its
        // AudioContext for the same, so the usual case is a straight copy. A VOICE port
        // (16 kHz) or a device that refused the requested rate lands here instead, and
        // says so once - a silently resampled stream is a title playing at the wrong
        // PITCH, which sounds like a decoder bug and is not one.
        let ratio = src_rate as f64 / device_rate as f64;
        if report_rate {
            web_sys::console::warn_1(&JsValue::from_str(&format!(
                "[audio] port {port} runs at {src_rate} Hz but the device is at {device_rate} Hz \
                 - resampling (linear). Ask the AudioContext for {src_rate} Hz to avoid it."
            )));
        }

        // How many device frames this grain becomes, given the fraction carried over.
        let out_frames = (((src_frames as f64) - pos0) / ratio).ceil().max(0.0) as usize;
        let mut pos = pos0;

        self.scratch.clear();
        self.scratch.reserve(out_frames * device_ch);
        for _ in 0..out_frames {
            let i = pos as usize;
            let frac = pos - i as f64;
            for c in 0..device_ch {
                // A mono source feeds both device channels; a stereo one maps in order.
                let sc = c % src_ch;
                let a = pcm.get(i * src_ch + sc).copied().unwrap_or(0) as f32;
                let b = pcm.get((i + 1) * src_ch + sc).copied().map_or(a, f32::from);
                let s = a + (b - a) * frac as f32;
                self.scratch.push(s / 32768.0 * gain[c.min(1)]);
            }
            pos += ratio;
        }
        // Carry the leftover fraction into the next grain - see `Port::resample_pos`.
        if let Some(p) = self.port(port) {
            p.resample_pos = pos - src_frames as f64;
        }
        self.publish(out_frames as u32);
    }

    fn set_volume(&mut self, port: i32, vols: &[i32]) {
        let Some(p) = self.port(port) else { return };
        for (c, v) in vols.iter().enumerate().take(2) {
            // The Vita range is 0..=32768, full scale at the top.
            p.gain[c] = (*v as f32 / 32768.0).clamp(0.0, 1.0);
        }
        // A one-channel set on a stereo port applies to both, which is what the mask form
        // of `sceAudioOutSetVolume` means when a title sets only the left.
        if vols.len() == 1 {
            p.gain[1] = p.gain[0];
        }
    }

    fn close_port(&mut self, port: i32) {
        self.ports.retain(|p| p.id != port);
    }
}
