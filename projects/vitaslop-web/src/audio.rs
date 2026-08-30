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
/// Loudest sample this run, as `|sample| * 32767`.
///
/// >>> THE ONLY WHOLE-RUN PROOF THAT ANYTHING WAS AUDIBLE. The ring itself is half a
/// second deep and circular, so reading its contents at the end of a run samples the
/// last 0.5s and nothing else - and a title whose audio is sparse (one measured front
/// end is silent for 95% of its running time) will show an empty ring on a run that
/// produced minutes of sound. A high-water mark cannot miss it.
const CTL_PEAK: u32 = 7;
/// Slots 8 and 9 belong to the consumer (latency skip, live backlog); the header is
/// sized past them so the layout has room to grow without moving the PCM.
const CTL_HEADER_BYTES: u32 = 64;

/// A `sceAudioOut` port as this sink sees it: the format the guest opened it with, which
/// is all that is needed to turn its grains into ring frames.
struct Port {
    id: i32,
    format: AudioFormat,
    /// Fractional read position into the source grain, carried ACROSS grains when the
    /// port's rate differs from the device's. Resetting it per grain would drop or
    /// duplicate a sample at every grain boundary, which is a periodic click at the
    /// grain rate - audible, and easy to misdiagnose as a decoder fault.
    resample_pos: f64,
    /// Whether the rate mismatch has been reported. Once is enough; it cannot change.
    reported_rate: bool,
    /// Where this port's next grain belongs in the ring, as an absolute frame count in the
    /// same monotonic space as `CTL_WRITE` / `CTL_READ`. See [`WebAudioSink::publish_at`]:
    /// ports play at the same time and are SUMMED, so each needs its own position rather
    /// than sharing one append cursor.
    cursor: u32,
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
    /// Scratch for the ring's own samples where a grain overlaps another port's, so the sum
    /// in [`WebAudioSink::publish_at`] allocates nothing either.
    mixbuf: Vec<f32>,
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
            mixbuf: Vec::new(),
        })
    }

    fn port(&mut self, id: i32) -> Option<&mut Port> {
        self.ports.iter_mut().find(|p| p.id == id)
    }

    /// Copy `src` into the ring at absolute frame `start`, wrapping at the end.
    fn write_range(&self, start: u32, src: &[f32]) {
        let ch = self.channels;
        let frames = (src.len() / ch as usize) as u32;
        let at = start % self.capacity;
        let first = frames.min(self.capacity - at);
        self.data
            .subarray(at * ch, (at + first) * ch)
            .copy_from(&src[..(first * ch) as usize]);
        if first < frames {
            self.data
                .subarray(0, (frames - first) * ch)
                .copy_from(&src[(first * ch) as usize..]);
        }
    }

    /// Read the ring's own samples at absolute frame `start` into `out`, wrapping.
    fn read_range(&self, start: u32, out: &mut [f32]) {
        let ch = self.channels;
        let frames = (out.len() / ch as usize) as u32;
        let at = start % self.capacity;
        let first = frames.min(self.capacity - at);
        self.data
            .subarray(at * ch, (at + first) * ch)
            .copy_to(&mut out[..(first * ch) as usize]);
        if first < frames {
            self.data
                .subarray(0, (frames - first) * ch)
                .copy_to(&mut out[(first * ch) as usize..]);
        }
    }

    /// Publish `frames` of interleaved device-rate f32 from `self.scratch` at the PORT's own
    /// position in the ring, SUMMING with whatever another port has already put there.
    /// Returns the port's new position.
    ///
    /// # Why a port has a position at all, and why appending was wrong
    /// Every open `sceAudioOut` port plays SIMULTANEOUSLY on hardware - the console's audio
    /// block sums them into one stereo stream. This sink used to append each grain at a single
    /// shared write cursor, so two live ports did not mix: they took turns, each pushing its
    /// own grains into the same timeline. MEASURED on a retail sports title's opening, which
    /// keeps two 48 kHz stereo ports open (grain 256 and grain 1024): the ring was fed roughly
    /// TWICE real time, `overrun 2,166,784` frames, and what the device played was alternating
    /// blocks of two different streams - a hiccup and a stutter, from a decoder and a mixer
    /// that were both working correctly.
    ///
    /// So each port carries an absolute frame position, and a grain is ADDED where it belongs
    /// in time rather than appended. Frames past the shared frontier are stale ring memory and
    /// are overwritten; frames before it hold another port's audio and are summed with it.
    ///
    /// The sum is deliberately NOT clamped here. A port can still submit into this window, and
    /// clamping now would throw away headroom a later grain could have cancelled; the device
    /// is the thing that clips, exactly as the console's DAC is.
    ///
    /// Single producer, single consumer: the write index is only advanced here and the read
    /// index only by the worklet, so the free space is a plain subtraction and no lock is
    /// needed. The data goes in BEFORE the index is published, which is the whole ordering
    /// requirement.
    fn publish_at(&mut self, cursor: u32, frames: u32) -> u32 {
        let write = self.ctl.get_index(CTL_WRITE) as u32;
        let read = js_sys::Atomics::load(&self.ctl, CTL_READ).unwrap_or(0) as u32;
        // A port that has fallen behind the consumer - it stopped submitting for a while, or
        // it is new - joins at the read cursor rather than writing into frames already played.
        let mut at = if read.wrapping_sub(cursor) as i32 > 0 { read } else { cursor };
        let free = self.capacity.saturating_sub(at.wrapping_sub(read));
        let take = frames.min(free);
        if take < frames {
            let _ = js_sys::Atomics::add(&self.ctl, CTL_OVERRUN, (frames - take) as i32);
        }
        if take > 0 {
            let ch = self.channels as usize;
            // The part that lands before the frontier already holds another port's audio.
            let overlap = if write.wrapping_sub(at) as i32 > 0 {
                take.min(write.wrapping_sub(at))
            } else {
                0
            };
            if overlap > 0 {
                let n = overlap as usize * ch;
                self.mixbuf.clear();
                self.mixbuf.resize(n, 0.0);
                let mut held = std::mem::take(&mut self.mixbuf);
                self.read_range(at, &mut held);
                for (h, s) in held.iter_mut().zip(self.scratch[..n].iter()) {
                    *h += *s;
                }
                self.write_range(at, &held);
                self.mixbuf = held;
            }
            if take > overlap {
                let from = overlap as usize * ch;
                let to = take as usize * ch;
                let src = std::mem::take(&mut self.scratch);
                self.write_range(at.wrapping_add(overlap), &src[from..to]);
                self.scratch = src;
            }
            let end = at.wrapping_add(take);
            if end.wrapping_sub(write) as i32 > 0 {
                let _ = js_sys::Atomics::store(&self.ctl, CTL_WRITE, end as i32);
            }
        }
        // Advance by the WHOLE grain even where the ring refused it: the port's position is
        // where its sound belongs in time, and a producer running ahead of the device drops
        // audio without shifting everything that follows.
        at = at.wrapping_add(frames);
        at
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
        // A new port joins at the frontier: whatever is already queued belongs to the ports
        // that queued it, and this one's first grain is the next sound to be heard.
        let cursor = self.ctl.get_index(CTL_WRITE) as u32;
        self.ports.push(Port {
            id,
            format,
            resample_pos: 0.0,
            reported_rate: false,
            cursor,
        });
        id
    }

    fn submit(&mut self, port: i32, pcm: &[i16]) {
        let device_rate = self.sample_rate;
        let device_ch = self.channels as usize;
        // Read everything needed off the port and DROP the borrow: the conversion below
        // fills `self.scratch`, which is a sibling field, and the resample cursor is
        // written back once at the end.
        let (src_ch, src_rate, pos0, report_rate, cursor) = {
            let Some(p) = self.port(port) else { return };
            let src_rate = p.format.sample_rate;
            let report = src_rate != device_rate && !p.reported_rate;
            if report {
                p.reported_rate = true;
            }
            (p.format.channels.max(1) as usize, src_rate, p.resample_pos, report, p.cursor)
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
        // The run's high-water mark is taken HERE, inside the conversion that already
        // touches every sample, rather than in a pass of its own - this is the guest's
        // audio thread at grain rate, and a second walk over the grain would be pure
        // overhead for a diagnostic.
        let mut peak = 0.0f32;
        for _ in 0..out_frames {
            let i = pos as usize;
            let frac = pos - i as f64;
            for c in 0..device_ch {
                // A mono source feeds both device channels; a stereo one maps in order.
                let sc = c % src_ch;
                let a = pcm.get(i * src_ch + sc).copied().unwrap_or(0) as f32;
                let b = pcm.get((i + 1) * src_ch + sc).copied().map_or(a, f32::from);
                let s = a + (b - a) * frac as f32;
                let out = s / 32768.0;
                peak = peak.max(out.abs());
                self.scratch.push(out);
            }
            pos += ratio;
        }
        let peak_i = (peak * 32767.0) as i32;
        if peak_i > self.ctl.get_index(CTL_PEAK) {
            let _ = js_sys::Atomics::store(&self.ctl, CTL_PEAK, peak_i);
        }
        // Carry the leftover fraction into the next grain - see `Port::resample_pos`.
        if let Some(p) = self.port(port) {
            p.resample_pos = pos - src_frames as f64;
        }
        let next = self.publish_at(cursor, out_frames as u32);
        if let Some(p) = self.port(port) {
            p.cursor = next;
        }
    }

    /// >>> DELIBERATELY IGNORED. The port volume is applied by the MIXER now, not here.
    ///
    /// This sink only ever sees a grain that has already been clamped to i16, so a volume
    /// applied here scales a signal the clamp has already destroyed: one racer's mix reaches
    /// 4.94x full scale and the volume the title sets is 0.355, so clamping first threw away
    /// everything above 1.0 and then made the remainder quiet. `vita::audio::out_output`
    /// applies it to the pre-clamp mix instead, which is both the audible fix and the order
    /// the hardware runs in. Left as an explicit no-op rather than removed from the trait so
    /// that a sink which DOES have pre-clamp samples can still take it.
    fn set_volume(&mut self, port: i32, vols: &[i32]) {
        let _ = (port, vols);
    }

    fn close_port(&mut self, port: i32) {
        self.ports.retain(|p| p.id != port);
    }
}
