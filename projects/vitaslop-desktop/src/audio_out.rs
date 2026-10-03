//! The desktop's [`AudioSink`]: guest PCM to this machine's speakers, through `cpal`.
//!
//! It is the browser's audio path transplanted, not a new design. The producer half is
//! `vitaslop-web/src/audio.rs` (`WebAudioSink`) and the consumer half is
//! `vitaslop-web/web/audio-worklet.js`; every policy below - ports SUMMED at their own
//! positions, the rejoin rules, the carried resample fraction, the sustained-lead cap, the
//! resume hysteresis, the edge ramps, what counts as an underrun - is theirs, MEASURED there,
//! and kept identical so a sound difference between the two products is a difference in the
//! guest's timing and never in how its samples reach a speaker.
//!
//! Between them sits the same single-producer single-consumer ring the page keeps in a
//! `SharedArrayBuffer`: interleaved f32 at the DEVICE's rate and channel count, with
//! monotonic WRITE / READ frame counters. Here it is a slice of atomics shared by an `Arc`.
//! `submit` never blocks - the guest's audio thread calls it from inside a host call - and
//! the device callback never waits on the producer.
//!
//! # What it does NOT do
//! Exactly what the browser's does not: it does not stretch or resync to hide the emulator's
//! frame rate. A title running below real time starves the device, the callback plays
//! silence, and that silence is COUNTED ([`AudioStats::underrun_s`]) - the honest report that
//! the emulator is not fast enough, not an audio fault to paper over.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::*};
use std::sync::Arc;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use vitaslop_runtime::audio::{AudioFormat, AudioSink};

/// The rate the Vita's MAIN and BGM ports run at, asked of the device first so the usual
/// grain is a straight copy (`web/audio.js` asks its AudioContext for the same).
const PREFERRED_RATE: u32 = 48_000;
/// Ring length. `web/audio.js` `RING_SECONDS`.
const RING_SECONDS: f64 = 0.5;
/// `web/audio-worklet.js` `MAX_LEAD_SECONDS`: the most audio allowed to sit ahead of the
/// device, judged by the backlog's TROUGH over [`LEAD_WINDOW_SECONDS`] - the sustained lead,
/// not the height of a burst.
const MAX_LEAD_SECONDS: f64 = 0.12;
const LEAD_WINDOW_SECONDS: f64 = 1.0;
const LEAD_BUCKETS: usize = 16;
/// `web/audio-worklet.js` `RAMP_FRAMES`: the fade at each gap's edges, so neither is a step.
const RAMP_FRAMES: f32 = 64.0;
/// `web/audio-worklet.js` `RESUME_SECONDS`: once starved, how much must be buffered before
/// playback resumes. Without it a ring that is chronically nearly empty warbles.
const RESUME_SECONDS: f64 = 0.04;

/// The shared ring. Samples are f32 BITS in atomics: the producer and the device callback run
/// on different threads, and an atomic per sample is what makes that sound without a lock.
/// Relaxed per-sample access is enough, because WRITE is published with `Release` after the
/// samples and READ with `Release` after they were consumed - the same ordering the page's
/// `Atomics.store` gives it.
struct Ring {
    data: Box<[AtomicU32]>,
    capacity: u32,
    channels: u32,
    sample_rate: u32,
    write: AtomicU32,
    read: AtomicU32,
    underrun: AtomicU64,
    overrun: AtomicU64,
    latency_skip: AtomicU64,
    rejoins: AtomicU64,
    /// Loudest sample this run, `|s| * 32767` - the only whole-run proof anything was audible.
    peak: AtomicU32,
    /// The person's pause (menu, Space, unfocused window): the guest is not producing, so
    /// the device plays silence WITHOUT counting underrun and the ring is left as it is.
    paused: AtomicBool,
    muted: AtomicBool,
}

impl Ring {
    fn sample(&self, i: u32) -> f32 {
        f32::from_bits(self.data[i as usize].load(Relaxed))
    }

    fn set_sample(&self, i: u32, v: f32) {
        self.data[i as usize].store(v.to_bits(), Relaxed);
    }
}

/// What the device side has seen, for the stats line and the end-of-run report.
#[derive(Clone, Copy, Debug, Default)]
pub struct AudioStats {
    pub underrun_s: f64,
    pub overrun_s: f64,
    pub latency_skip_s: f64,
    pub rejoins: u64,
    pub peak: f32,
}

/// The open output device and the ring it plays. Owned by the window's session: `cpal`'s
/// stream is not `Send` on every platform, so it stays on the thread that opened it, and the
/// guest gets only a [`NativeAudioSink`] over the shared ring.
pub struct AudioOut {
    ring: Arc<Ring>,
    _stream: cpal::Stream,
    pub device: String,
}

impl AudioOut {
    /// Open the default output device. `Err` (no device, or none that takes f32/i16/u16) is
    /// not fatal: the title runs silent, as it always has here, and the caller says why.
    pub fn open() -> Result<AudioOut, String> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or("no audio output device")?;
        let name = device.description().map(|d| d.to_string()).unwrap_or_else(|_| "default output".into());
        let config = pick_config(&device)?;
        let rate = config.sample_rate();
        let channels = u32::from(config.channels()).max(1);
        let capacity = ((rate as f64 * RING_SECONDS) as u32).max(1024);
        let ring = Arc::new(Ring {
            data: (0..capacity * channels).map(|_| AtomicU32::new(0)).collect(),
            capacity,
            channels,
            sample_rate: rate,
            write: AtomicU32::new(0),
            read: AtomicU32::new(0),
            underrun: AtomicU64::new(0),
            overrun: AtomicU64::new(0),
            latency_skip: AtomicU64::new(0),
            rejoins: AtomicU64::new(0),
            peak: AtomicU32::new(0),
            paused: AtomicBool::new(false),
            muted: AtomicBool::new(false),
        });
        let format = config.sample_format();
        let config = config.config();
        let stream = match format {
            cpal::SampleFormat::F32 => build::<f32>(&device, &config, ring.clone()),
            cpal::SampleFormat::I16 => build::<i16>(&device, &config, ring.clone()),
            cpal::SampleFormat::U16 => build::<u16>(&device, &config, ring.clone()),
            other => return Err(format!("the output device only takes {other:?} samples")),
        }?;
        stream.play().map_err(|e| format!("could not start audio output: {e}"))?;
        Ok(AudioOut { ring, _stream: stream, device: format!("{name} ({rate} Hz, {channels} ch)") })
    }

    /// A producer over this device's ring, for `VitaState::audio`.
    pub fn sink(&self) -> NativeAudioSink {
        NativeAudioSink { ring: self.ring.clone(), ports: Vec::new(), next_port: 0, scratch: Vec::new() }
    }

    pub fn set_paused(&self, paused: bool) {
        self.ring.paused.store(paused, Relaxed);
    }

    pub fn set_muted(&self, muted: bool) {
        self.ring.muted.store(muted, Relaxed);
    }

    pub fn muted(&self) -> bool {
        self.ring.muted.load(Relaxed)
    }

    pub fn stats(&self) -> AudioStats {
        let r = &self.ring;
        let s = |v: u64| v as f64 / r.sample_rate as f64;
        AudioStats {
            underrun_s: s(r.underrun.load(Relaxed)),
            overrun_s: s(r.overrun.load(Relaxed)),
            latency_skip_s: s(r.latency_skip.load(Relaxed)),
            rejoins: r.rejoins.load(Relaxed),
            peak: r.peak.load(Relaxed) as f32 / 32767.0,
        }
    }
}

/// 48 kHz in a sample format the callback can write, if the device has one; otherwise the
/// device's own default, and [`NativeAudioSink::submit`] resamples (and says so once).
fn pick_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig, String> {
    let usable = |f: cpal::SampleFormat| matches!(f, cpal::SampleFormat::F32 | cpal::SampleFormat::I16 | cpal::SampleFormat::U16);
    if let Ok(ranges) = device.supported_output_configs() {
        let mut best: Option<cpal::SupportedStreamConfig> = None;
        for r in ranges {
            if !usable(r.sample_format()) || r.channels() < 2 {
                continue;
            }
            let Some(c) = r.try_with_sample_rate(PREFERRED_RATE) else { continue };
            // Prefer f32, then the fewest channels (stereo over a surround layout).
            let better = match &best {
                None => true,
                Some(b) => {
                    let f32_now = c.sample_format() == cpal::SampleFormat::F32;
                    let f32_best = b.sample_format() == cpal::SampleFormat::F32;
                    (f32_now && !f32_best) || (f32_now == f32_best && c.channels() < b.channels())
                }
            };
            if better {
                best = Some(c);
            }
        }
        if let Some(c) = best {
            return Ok(c);
        }
    }
    device.default_output_config().map_err(|e| format!("no usable audio output configuration: {e}"))
}

fn build<T>(device: &cpal::Device, config: &cpal::StreamConfig, ring: Arc<Ring>) -> Result<cpal::Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let mut consumer = Consumer::new(ring);
    device
        .build_output_stream::<T, _, _>(
            *config,
            move |out: &mut [T], _| consumer.fill(out),
            |e| tracing::warn!(target: "vitaslop::audio", error = %e, "audio output stream error"),
            None,
        )
        .map_err(|e| format!("could not open the audio output: {e}"))
}

/// The device side - `web/audio-worklet.js` `VitaslopAudio.process`, line for line in policy.
struct Consumer {
    ring: Arc<Ring>,
    max_lead: u32,
    resume_at: u32,
    lead_buckets: [i64; LEAD_BUCKETS],
    lead_bucket: usize,
    lead_bucket_frames: u32,
    frames_per_bucket: u32,
    env: f32,
    starved: bool,
}

impl Consumer {
    fn new(ring: Arc<Ring>) -> Consumer {
        let rate = ring.sample_rate as f64;
        Consumer {
            max_lead: ((rate * MAX_LEAD_SECONDS) as u32).max(1),
            resume_at: ((rate * RESUME_SECONDS) as u32).max(1),
            lead_buckets: [i64::MAX; LEAD_BUCKETS],
            lead_bucket: 0,
            lead_bucket_frames: 0,
            // By FRAMES rather than callbacks: a device's callback size is its own choice and
            // can change between calls, which a block count would silently rescale.
            frames_per_bucket: ((rate * LEAD_WINDOW_SECONDS / LEAD_BUCKETS as f64) as u32).max(1),
            env: 0.0,
            starved: true,
            ring,
        }
    }

    fn fill<T: SizedSample + FromSample<f32>>(&mut self, out: &mut [T]) {
        let r = self.ring.clone();
        let out_ch = r.channels as usize;
        let frames = (out.len() / out_ch.max(1)) as u32;
        let silence = T::from_sample(0.0f32);
        if r.paused.load(Relaxed) {
            out.fill(silence);
            // The envelope restarts from silence, so the first block after a resume ramps in.
            self.env = 0.0;
            return;
        }
        let write = r.write.load(Acquire);
        let mut read = r.read.load(Relaxed);
        let mut available = write.wrapping_sub(read) as i64;

        // The SUSTAINED backlog: the trough over the last LEAD_WINDOW_SECONDS.
        let b = &mut self.lead_buckets[self.lead_bucket];
        *b = (*b).min(available);
        self.lead_bucket_frames += frames;
        if self.lead_bucket_frames >= self.frames_per_bucket {
            self.lead_bucket_frames = 0;
            self.lead_bucket = (self.lead_bucket + 1) % LEAD_BUCKETS;
            self.lead_buckets[self.lead_bucket] = i64::MAX;
        }
        let sustained = self.lead_buckets.iter().copied().min().unwrap_or(i64::MAX);
        // Only what the ring never drops below is past the cap; a window that has not seen a
        // full sweep yet (a bucket still at MAX is not the minimum) trims nothing extra.
        if sustained != i64::MAX && sustained > self.max_lead as i64 {
            let skip = sustained - self.max_lead as i64;
            read = read.wrapping_add(skip as u32);
            available -= skip;
            r.latency_skip.fetch_add(skip as u64, Relaxed);
            for b in self.lead_buckets.iter_mut().filter(|b| **b != i64::MAX) {
                *b -= skip;
            }
        }

        if self.starved {
            if available >= self.resume_at as i64 {
                self.starved = false;
            }
        } else if available <= 0 {
            self.starved = true;
        }
        let take = if self.starved { 0 } else { (frames as i64).min(available.max(0)) as u32 };
        let muted = r.muted.load(Relaxed);
        let step = 1.0 / RAMP_FRAMES;
        for i in 0..frames {
            let target = if i < take { 1.0 } else { 0.0 };
            self.env = if target > self.env { (self.env + step).min(1.0) } else { (self.env - step).max(0.0) };
            let o = &mut out[i as usize * out_ch..(i as usize + 1) * out_ch];
            if i < take && !muted {
                let base = (read.wrapping_add(i) % r.capacity) * r.channels;
                for (c, s) in o.iter_mut().enumerate() {
                    // The ring is at the device's own channel count, so this is in order.
                    *s = T::from_sample(r.sample(base + c as u32) * self.env);
                }
            } else {
                o.fill(silence);
            }
        }
        // Before the first grain ever written there is nothing to underrun (the boot is not
        // the audio path's fault) - see the worklet.
        if take < frames && write > 0 {
            r.underrun.fetch_add((frames - take) as u64, Relaxed);
        }
        r.read.store(read.wrapping_add(take), Release);
    }
}

/// A `sceAudioOut` port as the sink sees it. `WebAudioSink`'s `Port`, field for field.
struct Port {
    id: i32,
    format: AudioFormat,
    /// Fractional read position into the source, carried ACROSS grains when the port's rate
    /// differs from the device's - resetting it per grain is a click at the grain rate.
    resample_pos: f64,
    reported_rate: bool,
    /// Where this port's next grain belongs, in the ring's absolute frame space. Ports play
    /// at once on hardware and are SUMMED, so each needs its own position.
    cursor: u32,
}

/// The producer half: `vitaslop-web/src/audio.rs` `WebAudioSink`, over the native ring.
pub struct NativeAudioSink {
    ring: Arc<Ring>,
    ports: Vec<Port>,
    next_port: i32,
    /// One grain of device-rate interleaved f32, reused so a call at grain rate allocates nothing.
    scratch: Vec<f32>,
}

impl NativeAudioSink {
    /// `WebAudioSink::publish_into`: SUM `frames` of `self.scratch` into the ring at `cursor`
    /// and return the port's next position. Same rejoin, overrun and frontier rules - see
    /// the web sink for the measurements behind each.
    fn publish(&mut self, cursor: u32, frames: u32) -> u32 {
        let r = &*self.ring;
        let write = r.write.load(Relaxed);
        let read = r.read.load(Acquire);
        // Behind the consumer (stopped submitting, or new): join at READ.
        let mut at = if read.wrapping_sub(cursor) as i32 > 0 { read } else { cursor };
        // A whole ring AHEAD of the consumer: nothing it submits could ever land, so rejoin at
        // the frontier rather than go silent for the rest of the run.
        if at.wrapping_sub(read) >= r.capacity {
            let frontier = if write.wrapping_sub(read) < r.capacity { write } else { read };
            r.rejoins.fetch_add(1, Relaxed);
            at = frontier;
        }
        let free = r.capacity.saturating_sub(at.wrapping_sub(read));
        let take = frames.min(free);
        if take < frames {
            r.overrun.fetch_add((frames - take) as u64, Relaxed);
        }
        if take > 0 {
            let ch = r.channels;
            // Frames before the frontier already hold another port's audio: add to them.
            let overlap = if write.wrapping_sub(at) as i32 > 0 { take.min(write.wrapping_sub(at)) } else { 0 };
            for f in 0..take {
                let slot = (at.wrapping_add(f) % r.capacity) * ch;
                for c in 0..ch {
                    let s = self.scratch[(f * ch + c) as usize];
                    let i = slot + c;
                    // Deliberately NOT clamped: the device is what clips, as the console's DAC is.
                    r.set_sample(i, if f < overlap { r.sample(i) + s } else { s });
                }
            }
            let end = at.wrapping_add(take);
            if end.wrapping_sub(write) as i32 > 0 {
                r.write.store(end, Release);
            }
        }
        // Advance by the WHOLE grain even where the ring refused it.
        at.wrapping_add(frames)
    }
}

impl AudioSink for NativeAudioSink {
    fn open_port(&mut self, format: AudioFormat) -> i32 {
        let id = self.next_port;
        self.next_port += 1;
        tracing::info!(
            target: "vitaslop::audio",
            "guest opened a port: {} ch, {} Hz, grain {} (device {} Hz, {} ch)",
            format.channels, format.sample_rate, format.grain, self.ring.sample_rate, self.ring.channels
        );
        // A new port joins at the frontier: what is queued belongs to the ports that queued it.
        let cursor = self.ring.write.load(Relaxed);
        self.ports.push(Port { id, format, resample_pos: 0.0, reported_rate: false, cursor });
        id
    }

    fn submit(&mut self, port: i32, pcm: &[i16]) {
        let device_rate = self.ring.sample_rate;
        let device_ch = self.ring.channels as usize;
        let Some(p) = self.ports.iter_mut().find(|p| p.id == port) else { return };
        let src_ch = p.format.channels.max(1) as usize;
        let src_rate = p.format.sample_rate;
        if src_rate == 0 || pcm.len() < src_ch {
            return;
        }
        if src_rate != device_rate && !p.reported_rate {
            p.reported_rate = true;
            tracing::info!(
                target: "vitaslop::audio",
                "port {port} runs at {src_rate} Hz but the device is at {device_rate} Hz - resampling (linear)"
            );
        }
        let (pos0, cursor) = (p.resample_pos, p.cursor);
        let src_frames = pcm.len() / src_ch;
        let ratio = src_rate as f64 / device_rate as f64;
        let out_frames = (((src_frames as f64) - pos0) / ratio).ceil().max(0.0) as usize;
        let mut pos = pos0;
        self.scratch.clear();
        self.scratch.reserve(out_frames * device_ch);
        let mut peak = 0.0f32;
        for _ in 0..out_frames {
            let i = pos as usize;
            let frac = (pos - i as f64) as f32;
            for c in 0..device_ch {
                // Mono feeds every device channel; stereo maps in order and repeats past two.
                let sc = c % src_ch;
                let a = pcm.get(i * src_ch + sc).copied().unwrap_or(0) as f32;
                let b = pcm.get((i + 1) * src_ch + sc).copied().map_or(a, f32::from);
                let out = (a + (b - a) * frac) / 32768.0;
                peak = peak.max(out.abs());
                self.scratch.push(out);
            }
            pos += ratio;
        }
        self.ring.peak.fetch_max((peak * 32767.0) as u32, Relaxed);
        let next = self.publish(cursor, out_frames as u32);
        if let Some(p) = self.ports.iter_mut().find(|p| p.id == port) {
            p.resample_pos = pos - src_frames as f64;
            p.cursor = next;
        }
    }

    /// Ignored, as in the browser: `vita::audio::out_output` applies the port volume to the
    /// pre-clamp mix, which is the order the hardware runs in.
    fn set_volume(&mut self, port: i32, vols: &[i32]) {
        let _ = (port, vols);
    }

    fn close_port(&mut self, port: i32) {
        self.ports.retain(|p| p.id != port);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(capacity: u32, channels: u32, rate: u32) -> Arc<Ring> {
        Arc::new(Ring {
            data: (0..capacity * channels).map(|_| AtomicU32::new(0)).collect(),
            capacity,
            channels,
            sample_rate: rate,
            write: AtomicU32::new(0),
            read: AtomicU32::new(0),
            underrun: AtomicU64::new(0),
            overrun: AtomicU64::new(0),
            latency_skip: AtomicU64::new(0),
            rejoins: AtomicU64::new(0),
            peak: AtomicU32::new(0),
            paused: AtomicBool::new(false),
            muted: AtomicBool::new(false),
        })
    }

    fn sink(r: &Arc<Ring>) -> NativeAudioSink {
        NativeAudioSink { ring: r.clone(), ports: Vec::new(), next_port: 0, scratch: Vec::new() }
    }

    /// Two open ports play at the same time and are SUMMED where they overlap - the
    /// measured defect in the web sink's history was appending them, which took turns.
    #[test]
    fn two_ports_mix_rather_than_append() {
        let r = ring(4096, 2, 48_000);
        let mut s = sink(&r);
        let fmt = AudioFormat { channels: 2, sample_rate: 48_000, grain: 4 };
        let a = s.open_port(fmt);
        let b = s.open_port(fmt);
        s.submit(a, &[8192; 8]);
        s.submit(b, &[4096; 8]);
        assert_eq!(r.write.load(Relaxed), 4, "two 4-frame grains at one time occupy 4 frames, not 8");
        assert!((r.sample(0) - 0.375).abs() < 1e-6, "0.25 + 0.125 summed: {}", r.sample(0));
    }

    /// Mono to a stereo device fills both channels, and a 16 kHz port is stretched 3x to a
    /// 48 kHz device with the fraction carried across grains (no frame dropped or doubled).
    #[test]
    fn mono_upmixes_and_rates_convert_without_seams() {
        let r = ring(4096, 2, 48_000);
        let mut s = sink(&r);
        let p = s.open_port(AudioFormat { channels: 1, sample_rate: 16_000, grain: 4 });
        s.submit(p, &[1000, 2000, 3000, 4000]);
        s.submit(p, &[5000, 6000, 7000, 8000]);
        assert_eq!(r.write.load(Relaxed), 24, "8 source frames at 3x are 24 device frames");
        assert_eq!(r.sample(0), r.sample(1), "mono feeds both channels");
    }

    /// The consumer: silence until RESUME_SECONDS of audio is buffered, then plays it; a
    /// paused device emits silence and counts no underrun.
    #[test]
    fn the_device_side_waits_for_a_buffer_and_pause_is_not_underrun() {
        let r = ring(48_000, 2, 48_000);
        let mut c = Consumer::new(r.clone());
        let mut s = sink(&r);
        let p = s.open_port(AudioFormat { channels: 2, sample_rate: 48_000, grain: 256 });
        s.submit(p, &[16384; 512]);
        let mut out = [1.0f32; 256];
        c.fill(&mut out);
        assert!(out.iter().all(|v| *v == 0.0), "256 frames buffered is under the 40 ms resume mark");
        assert_eq!(r.read.load(Relaxed), 0);
        for _ in 0..10 {
            s.submit(p, &[16384; 512]);
        }
        c.fill(&mut out);
        assert!(out.iter().any(|v| *v > 0.0), "past the resume mark it plays");
        r.paused.store(true, Relaxed);
        let before = r.underrun.load(Relaxed);
        c.fill(&mut out);
        assert!(out.iter().all(|v| *v == 0.0));
        assert_eq!(r.underrun.load(Relaxed), before, "a pause is not an underrun");
    }
}
