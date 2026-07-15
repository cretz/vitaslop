//! The audio-output seam: a thin, non-blocking sink for the PCM a title produces
//! through `sceAudioOut`. Kept deliberately small so different hosts implement it
//! differently but accurately - a native backend feeds a device ring buffer, a
//! browser backend queues into a Web Audio `AudioBuffer` - without either doing any
//! heavy work on the call path (the guest's audio thread calls `submit` at grain
//! rate and must not stall). NGS mixing happens in guest code; only the final
//! `sceAudioOut` stream crosses this boundary.
//!
//! Playback pacing (the real `sceAudioOutOutput` blocks until the previous grain
//! drains) is the host/scheduler's concern, not the sink's: `submit` always
//! returns immediately. The default [`NullSink`] discards everything, so bring-up
//! and headless runs stay silent and allocation-free.

/// A single opened output port's format, handed to the backend at open time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioFormat {
    /// Channels per frame (1 = mono, 2 = stereo).
    pub channels: u32,
    /// Sample rate in Hz (e.g. 48000).
    pub sample_rate: u32,
    /// Frames per `submit` grain (Vita ports fix this at open time).
    pub grain: u32,
}

/// A non-blocking PCM output sink. One backend port per guest `sceAudioOut` port.
/// Implementations must not block in `submit`; they queue and let their own clock
/// pace playback.
pub trait AudioSink {
    /// Open an output port with `format`. Returns a backend port id (`>= 0`) the
    /// guest handle maps to, or a negative value if the backend cannot open it.
    fn open_port(&mut self, format: AudioFormat) -> i32;

    /// Submit one grain of interleaved signed-16 PCM (`format.grain * channels`
    /// samples). Non-blocking.
    fn submit(&mut self, port: i32, pcm: &[i16]);

    /// Set per-channel volume in the Vita range (0..=32768). `vols.len()` matches
    /// the port's channel count.
    fn set_volume(&mut self, port: i32, vols: &[i32]) {
        let _ = (port, vols);
    }

    /// Release a port opened by [`open_port`](Self::open_port).
    fn close_port(&mut self, port: i32) {
        let _ = port;
    }
}

/// The default sink: discards all audio and hands out sequential port ids. Keeps
/// silent runs (bring-up, headless probes, replay) free of any device dependency.
#[derive(Default)]
pub struct NullSink {
    next_port: i32,
}

impl AudioSink for NullSink {
    fn open_port(&mut self, _format: AudioFormat) -> i32 {
        let p = self.next_port;
        self.next_port += 1;
        p
    }
    fn submit(&mut self, _port: i32, _pcm: &[i16]) {}
}
