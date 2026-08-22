//! SceAudio: the low-level PCM output ports (`sceAudioOut*`). A title opens a port
//! with a fixed grain, sample rate, and channel count, then hands one grain of
//! interleaved signed-16 PCM per call to `sceAudioOutOutput`. This module marshals
//! that stream across the [`AudioSink`](crate::audio::AudioSink) seam; the sink
//! (native device / Web Audio / the silent default) does the actual playback.
//!
//! Real `sceAudioOutOutput` blocks until the previous grain drains - that is how a
//! title paces its audio thread. We mirror that under the preemptive scheduler by
//! parking the caller for the grain's play duration (`grain / sample_rate`): the
//! audio thread stops busy-spinning at full speed, and because it now yields the
//! scheduler goes idle and advances the virtual clock - so a title's time-based
//! loading wait (which the spinning audio thread would otherwise starve) progresses.

use crate::audio::AudioFormat;
use crate::host::{GuestCtx, VitaState};
use crate::{hostcall, SvcOutcome};

/// One guest-visible audio port: the format it was opened with and the backend
/// port id the sink handed back.
struct AudioPort {
    /// Guest-facing port number returned to the title.
    guest_port: i32,
    /// Backend port id from [`AudioSink::open_port`](crate::audio::AudioSink::open_port).
    backend_port: i32,
    /// `SceAudioOutPortType` the port was opened with (MAIN 0 / BGM 1 / VOICE 2).
    /// Tracked so `sceAudioOutGetAdopt` can report whether a type is in use.
    ty: i32,
    format: AudioFormat,
}

/// Audio-output and NGS bookkeeping owned by [`VitaState`](crate::host::VitaState):
/// the set of open `sceAudioOut` ports plus the small amount of state the NGS HLE
/// (`vita::ngs`) needs so it hands back stable handles and buffers across frames.
#[derive(Default)]
pub struct AudioState {
    ports: Vec<AudioPort>,
    next_guest_port: i32,
    /// One shared, zeroed guest blob returned for every NGS voice-definition getter
    /// (the title treats these as opaque tokens). Lazily allocated; 0 = not yet.
    pub(crate) ngs_def_blob: u32,
    /// Per-`(voice, module, param)` params buffer handed back by
    /// `sceNgsVoiceLockParams`. Cached so the per-frame lock/unlock cycle reuses one
    /// buffer instead of leaking a fresh allocation every frame.
    pub(crate) ngs_param_bufs: Vec<((u32, u32, u32), u32)>,
    /// AT9 source voices, decoded and mixed into the output at `sceAudioOutOutput`.
    pub(crate) at9: super::at9::At9Bank,
    /// Which source voice each NGS patch handle carries, from
    /// `sceNgsPatchCreateRouting`. A routing volume names a PATCH; the mixer works in
    /// voices, and this is the only link between the two.
    pub(crate) ngs_patch_voice: Vec<(u32, u32)>,
    /// Optional raw-s16le capture of the mixed output stream (env
    /// `VITASLOP_AUDIO_RAW`), for headless verification. `None` = disabled.
    capture: Option<std::fs::File>,
    capture_inited: bool,
}

impl AudioState {
    fn format_of(&self, guest_port: i32) -> Option<(i32, AudioFormat)> {
        self.ports
            .iter()
            .find(|p| p.guest_port == guest_port)
            .map(|p| (p.backend_port, p.format))
    }

    /// Route a patch's routing volume to the voice that patch carries.
    ///
    /// A volume for a patch we never saw created is DROPPED rather than applied to some
    /// other voice - a misattributed gain is a voice at the wrong level, which is worse
    /// than one at unity because it looks deliberate.
    pub(crate) fn set_patch_volume(&mut self, patch: u32, volume: f32) {
        if !volume.is_finite() || volume < 0.0 {
            return;
        }
        match self.ngs_patch_voice.iter().find(|(p, _)| *p == patch) {
            Some((_, voice)) => {
                let voice = *voice;
                tracing::debug!(
                    target: "vitaslop::at9",
                    patch = format_args!("{patch:#x}"),
                    voice = format_args!("{voice:#x}"),
                    volume,
                    "routing volume applied"
                );
                self.at9.set_gain(voice, volume);
            }
            None => tracing::debug!(
                target: "vitaslop::at9",
                patch = format_args!("{patch:#x}"),
                volume,
                "routing volume for a patch that was never created here - DROPPED"
            ),
        }
    }

    /// The cached params buffer for a `(voice, module, param)` triple, if one was
    /// already handed out.
    pub(crate) fn ngs_param_buf(&self, key: (u32, u32, u32)) -> Option<u32> {
        self.ngs_param_bufs.iter().find(|(k, _)| *k == key).map(|(_, a)| *a)
    }

    /// Append one grain of mixed PCM to the raw-s16le capture file, if
    /// `VITASLOP_AUDIO_RAW` names one. Diagnostic; opens the file on first use.
    fn capture_pcm(&mut self, pcm: &[i16]) {
        if !self.capture_inited {
            self.capture_inited = true;
            if let Some(path) = std::env::var_os("VITASLOP_AUDIO_RAW") {
                self.capture = std::fs::File::create(path).ok();
            }
        }
        if let Some(f) = self.capture.as_mut() {
            use std::io::Write;
            let mut bytes = Vec::with_capacity(pcm.len() * 2);
            for s in pcm {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            let _ = f.write_all(&bytes);
        }
    }
}

/// int sceAudioOutOpenPort(SceAudioOutPortType type, int len, int freq, SceAudioOutMode mode)
/// `len` is the grain (frames per output), `mode` 0 = mono / 1 = stereo. Returns
/// the guest port number (>= 0).
#[hostcall]
pub(super) fn out_open_port(_ctx: &mut GuestCtx, st: &mut VitaState, ty: i32, len: i32, freq: i32, mode: i32) -> i32 {
    let channels = if mode == 1 { 2 } else { 1 };
    let format = AudioFormat {
        channels,
        sample_rate: freq.max(0) as u32,
        grain: len.max(0) as u32,
    };
    let backend_port = st.audio.open_port(format);
    if backend_port < 0 {
        backend_port
    } else {
        let guest_port = st.audio_state.next_guest_port;
        st.audio_state.next_guest_port += 1;
        st.audio_state.ports.push(AudioPort { guest_port, backend_port, ty, format });
        guest_port
    }
}

/// `VITASLOP_NO_NGS_MIX`: skip the NGS decode-and-mix entirely, leaving the guest's
/// output buffer as it found it (silent). The A/B arm for pricing the audio path: the
/// engine decodes ATRAC9 and PS-ADPCM and mixes every playing voice inside a host call
/// at grain rate, and "designed to be cheap" is not the same as measured cheap. Turning
/// it off changes nothing else about the run, so a wall-clock difference between the two
/// arms is the audio path and nothing else.
fn no_ngs_mix() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_NO_NGS_MIX"))
}

/// int sceAudioOutGetAdopt(SceAudioOutPortType type)
/// "Get status of port type": returns (1) if a port of `type` is currently in use
/// for sound generation, (0) otherwise. A title polls this before opening a port to
/// see whether the type is already claimed; we report it from the open-port set.
#[hostcall]
pub(super) fn out_get_adopt(_ctx: &mut GuestCtx, st: &mut VitaState, ty: i32) -> i32 {
    i32::from(st.audio_state.ports.iter().any(|p| p.ty == ty))
}

/// int sceAudioOutOutput(int port, const void *buf)
/// Submit one grain of interleaved S16 PCM. Returns 0 on success. On real hardware
/// this blocks until the previous grain has drained; under the preemptive scheduler
/// we reproduce that pacing by parking the caller for the grain's play duration.
pub(super) fn out_output(ctx: &mut GuestCtx, st: &mut VitaState) -> SvcOutcome {
    let port = ctx.arg(0) as i32;
    let buf = ctx.arg(1);
    if buf != 0 && tracing::enabled!(target: "vitaslop::ngs", tracing::Level::TRACE) {
        let head = ctx.read_bytes(buf, 32);
        let nonzero = head.iter().any(|&b| b != 0);
        tracing::trace!(
            target: "vitaslop::ngs",
            port, buf = format_args!("{buf:#x}"), nonzero, head = format_args!("{head:02x?}"),
            "AudioOutOutput"
        );
    }
    let (backend_port, format) = match st.audio_state.format_of(port) {
        None => {
            ctx.ret(-1i32 as u32);
            return SvcOutcome::Continue;
        }
        Some(x) => x,
    };
    // A null buffer means "drain/stop" on real hardware; nothing to submit.
    if buf != 0 {
        let grain = format.grain as usize;
        let channels = format.channels as usize;
        // The Vita's NGS DSP would have mixed the playing AT9 voices into the master
        // buss that the title copies here. Our NGS is host-side, so we do that mix
        // now and write it into the (otherwise silent) output buffer.
        if st.audio_state.at9.any_playing() && !no_ngs_mix() {
            let mut mix = vec![0i32; grain * channels];
            st.audio_state.at9.mix_grain(ctx, &mut mix, grain, channels, format.sample_rate);
            // How far past full scale the summed voices reach BEFORE the clamp. This is
            // the size of the missing master stage: the device does not clip here, so
            // whatever it applies must bring this back under 1.0. Reported as a
            // high-water mark so one number describes the whole run.
            if tracing::enabled!(target: "vitaslop::at9", tracing::Level::DEBUG) {
                use std::sync::atomic::{AtomicU32, Ordering};
                static PEAK_PERMILLE: AtomicU32 = AtomicU32::new(0);
                let peak = mix.iter().map(|s| s.unsigned_abs()).max().unwrap_or(0);
                let permille = (peak as u64 * 1000 / 32768) as u32;
                if permille > PEAK_PERMILLE.fetch_max(permille, Ordering::Relaxed) {
                    tracing::debug!(
                        target: "vitaslop::at9",
                        headroom_needed = format_args!("{:.3}x full scale", permille as f32 / 1000.0),
                        "mix peak before clamp (new high)"
                    );
                }
            }
            let mut bytes = Vec::with_capacity(mix.len() * 2);
            for &s in &mix {
                bytes.extend_from_slice(&(s.clamp(-32768, 32767) as i16).to_le_bytes());
            }
            ctx.write_bytes(buf, &bytes);
        }
        let samples = grain * channels;
        let raw = ctx.read_bytes(buf, samples * 2);
        let pcm: Vec<i16> =
            raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        st.audio_state.capture_pcm(&pcm);
        st.audio.submit(backend_port, &pcm);
    }
    ctx.ret(0);
    // One grain plays for `grain / sample_rate` seconds.
    let paced_us = if format.sample_rate > 0 && format.grain > 0 {
        (format.grain as u64 * 1_000_000) / format.sample_rate as u64
    } else {
        0
    };
    // Pace the audio thread on the preemptive scheduler; a run-to-completion host
    // has no clock to advance, so it just continues.
    if paced_us > 0 && st.is_preemptive() {
        st.sleep_park(paced_us);
        SvcOutcome::Block
    } else {
        SvcOutcome::Continue
    }
}

/// int sceAudioOutSetVolume(int port, SceAudioOutChannelFlag ch, int *vol)
/// `ch` is a bitmask of channels to set; `vol` points to one value per set channel.
#[hostcall]
pub(super) fn out_set_volume(ctx: &mut GuestCtx, st: &mut VitaState, port: i32, ch: i32, vol: Ptr) -> i32 {
    match st.audio_state.format_of(port) {
        None => -1,
        Some((backend_port, _)) => {
            // `vol` carries one entry per bit set in `ch` (L=1, R=2), ascending.
            let count = (ch & 0x3).count_ones() as usize;
            if vol.addr() != 0 && count > 0 {
                let vols: Vec<i32> =
                    (0..count).map(|i| ctx.read_u32(vol.addr() + (i as u32) * 4) as i32).collect();
                // What the title asks the PORT for, against the 32768 full-scale point.
                // It matters for reading a mix that clips: the grain is clamped to i16
                // before this is applied, so a title running the port well below full
                // scale still loses everything the clamp took.
                tracing::debug!(
                    target: "vitaslop::at9",
                    port,
                    ?vols,
                    full_scale = 32768,
                    "sceAudioOutSetVolume"
                );
                st.audio.set_volume(backend_port, &vols);
            }
            0
        }
    }
}

/// int sceAudioOutReleasePort(int port)
#[hostcall]
pub(super) fn out_release_port(_ctx: &mut GuestCtx, st: &mut VitaState, port: i32) -> i32 {
    if let Some(pos) = st.audio_state.ports.iter().position(|p| p.guest_port == port) {
        let backend = st.audio_state.ports.remove(pos).backend_port;
        st.audio.close_port(backend);
        0
    } else {
        -1
    }
}
