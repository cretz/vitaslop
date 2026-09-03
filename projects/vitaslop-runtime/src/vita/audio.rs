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

/// Where `sceAudioIn` port ids start. Deliberately above anything `sceAudioOutOpenPort`
/// hands out, so the two families cannot be confused for each other.
const IN_PORT_BASE: i32 = 0x100;

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
    /// Per-channel output volume from `sceAudioOutSetVolume`, 0..=1 (the Vita range is
    /// 0..=32768). Held HERE, in the mixer, rather than in the sink - see
    /// [`out_output`] for why the order matters.
    gain: [f32; 2],
}

/// Audio-output and NGS bookkeeping owned by [`VitaState`](crate::host::VitaState):
/// the set of open `sceAudioOut` ports plus the small amount of state the NGS HLE
/// (`vita::ngs`) needs so it hands back stable handles and buffers across frames.
#[derive(Default)]
pub struct AudioState {
    ports: Vec<AudioPort>,
    next_guest_port: i32,
    /// Open `sceAudioIn` (microphone) ports; see [`AudioInPort`].
    in_ports: Vec<AudioInPort>,
    /// One zeroed guest blob per NGS voice-definition getter, as `(func nid, address)`.
    ///
    /// The title treats these as opaque tokens, so ONE shared blob served them all - and that
    /// threw away the only thing a rack description says about itself. A rack names its
    /// definition by pointer, so with one pointer for every definition there is no way to
    /// tell a rack of sample players from a rack of COMPRESSOR busses, which is exactly the
    /// distinction an unaccounted-for attenuation in a mix turns on. One blob per getter
    /// costs a few allocations for a whole run and makes the rack readable.
    pub(crate) ngs_defs: Vec<(u32, u32)>,
    /// Per-`(voice, module, param)` params buffer handed back by
    /// `sceNgsVoiceLockParams`. Cached so the per-frame lock/unlock cycle reuses one
    /// buffer instead of leaking a fresh allocation every frame.
    pub(crate) ngs_param_bufs: Vec<((u32, u32, u32), u32)>,
    /// The guest function a title registered with `sceNgsVoiceSetModuleCallback`, as
    /// `((voice, module), (entry, userdata))`. Only the player module (0) raises anything
    /// here - a buffer boundary, see [`super::at9::PlayerEvent`] - but the key carries the
    /// module so a registration on any other module is recorded rather than misdelivered.
    pub(crate) ngs_module_cbs: Vec<((u32, u32), (u32, u32))>,
    /// `sceNgsVoiceSetFinishedCallback` registrations, as `(voice, (entry, userdata))`:
    /// raised once when a voice's data runs out.
    pub(crate) ngs_finished_cbs: Vec<(u32, (u32, u32))>,
    /// The voice handle for each `(rack, index)` a title has asked about. A rack is a fixed
    /// array of voices and this is a LOOKUP - see `ngs::rack_get_voice_handle` for what
    /// allocating a fresh handle per query did to the mixer.
    pub(crate) ngs_voice_handles: Vec<((u32, u32), u32)>,
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
    /// Scratch buffers for one grain of output, reused across calls.
    ///
    /// `sceAudioOutOutput` runs at the audio rate - thousands of times a minute, forever -
    /// and it used to allocate THREE vectors per call (the i32 mix, its little-endian bytes,
    /// and the i16 samples read back out of guest memory). MEASURED with a V8 worker profile
    /// of one retail title's browser race, where audio is 46% of the thread: `out_output`'s own
    /// body was **12% of every sample on the thread**, which for a function whose job is to
    /// move one grain of PCM is all overhead. Taken out with `mem::take` and put back, the
    /// way the texture-set re-proof borrows its dependency list - the guest cannot run
    /// inside a host call, so an empty vector here is not a state anything can observe.
    scratch_mix: Vec<i32>,
    scratch_bytes: Vec<u8>,
    scratch_pcm: Vec<i16>,
    /// Output frames the guest has submitted through `sceAudioOutOutput`, PER PORT, with the
    /// rate each was submitted at: `(port, frames, rate)`.
    ///
    /// >>> PER PORT, BECAUSE PORTS PLAY AT THE SAME TIME AND A SUM COUNTS WALL TIME TWICE.
    /// This was one running total, and a title playing a movie holds a second output port
    /// open beside its game audio - both submitting a grain per period, both heard together.
    /// The sum then says the guest produced two seconds of sound per second of clock, which
    /// reads exactly like the audio path running at double rate. MEASURED 2026-09-02 on one
    /// title's intro movie: `1.98x over this window`, entirely from the second port.
    /// [[vitaslop-audio-ports-are-mixed-not-appended]]
    submitted: Vec<(i32, u64, u32)>,
}

impl AudioState {
    /// SECONDS OF SOUND THE GUEST HAS PRODUCED, to be read against the EMULATED CLOCK.
    ///
    /// # The two ratios this feeds, and which one means what
    /// `sceAudioOutOutput` parks the audio thread for one grain of VIRTUAL time, so audio is
    /// billed in game clock. Two different things can therefore be wrong, and a device's ring
    /// counters (`written` / `read` / `underrun` / `overrun`) cannot tell them apart because
    /// they describe the RING:
    ///
    /// * **sound / emulated clock** is 1.00 on a healthy path WHATEVER the frame rate. It
    ///   does not move with the display, and it is what says the audio path itself tracks the
    ///   clock it is paced on. Measured 0.95-0.99 on all five titles.
    /// * **clock per displayed frame, in display periods** is the title's own vblank divisor
    ///   - a WHOLE number (1 for 60 fps, 2 for 30). That is the one that catches a clock
    ///   running fast, and it is NOT visible in the ratio above: PCSA00009 read 0.985 sound /
    ///   clock while charging **2.99 periods a frame for a limiter that asks for two**, and
    ///   that extra period is what made the guest produce 1.7 s of audio per second of real
    ///   time on a phone, fill the ring, drop a third of it, and starve on the next hitch.
    ///
    /// So a device capture showing `UNDERRUN 24.7%` beside `OVERRUN 49.5%` is a RATE problem,
    /// not a buffer one, and the period count is where to look for it.
    /// >>> AND IT IS THE BUSIEST PORT, NOT THE SUM OF THEM. Concurrent ports are MIXED into
    /// one output, so a second of movie sound played over a second of game sound is one
    /// second of sound. The port that has submitted the most is the one whose length this
    /// measures; a port that opens late and plays briefly cannot inflate the figure.
    pub fn produced_seconds(&self) -> f64 {
        self.submitted
            .iter()
            .filter(|(_, _, rate)| *rate != 0)
            .map(|(_, frames, rate)| *frames as f64 / f64::from(*rate))
            .fold(0.0, f64::max)
    }
}

/// One open `sceAudioIn` port. There is no backend behind it - see `vita::audioin` for
/// why a muted microphone is the honest model - so all that is recorded is what the
/// guest opened it with, which is what paces its reads.
struct AudioInPort {
    port: i32,
    ty: i32,
    grain: u32,
    freq: u32,
}

impl AudioState {
    /// `sceAudioInOpenPort`: allocate an input port. Ids start at 1, disjoint from the
    /// OUTPUT port numbering, so an input id handed to `sceAudioOutOutput` (or the
    /// reverse) is refused instead of naming somebody else's port.
    pub(crate) fn in_open(&mut self, ty: i32, grain: u32, freq: u32) -> i32 {
        let port = IN_PORT_BASE + self.in_ports.len() as i32;
        self.in_ports.push(AudioInPort { port, ty, grain, freq });
        port
    }

    /// `sceAudioInReleasePort`.
    pub(crate) fn in_close(&mut self, port: i32) -> bool {
        let before = self.in_ports.len();
        self.in_ports.retain(|p| p.port != port);
        self.in_ports.len() != before
    }

    /// `(grain, freq)` of an open input port.
    pub(crate) fn in_format(&self, port: i32) -> Option<(u32, u32)> {
        self.in_ports.iter().find(|p| p.port == port).map(|p| (p.grain, p.freq))
    }

    /// Whether a port of `ty` is open (`sceAudioInGetAdopt`).
    pub(crate) fn in_adopted(&self, ty: i32) -> bool {
        self.in_ports.iter().any(|p| p.ty == ty)
    }

    fn format_of(&self, guest_port: i32) -> Option<(i32, AudioFormat)> {
        self.ports
            .iter()
            .find(|p| p.guest_port == guest_port)
            .map(|p| (p.backend_port, p.format))
    }

    /// The output volume the title set on `guest_port`, or unity if it never set one.
    fn gain_of(&self, guest_port: i32) -> [f32; 2] {
        self.ports
            .iter()
            .find(|p| p.guest_port == guest_port)
            .map_or([1.0, 1.0], |p| p.gain)
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
        st.audio_state.ports.push(AudioPort {
            guest_port,
            backend_port,
            ty,
            format,
            gain: [1.0, 1.0],
        });
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
    // >>> THE OUTPUT PORT'S OWN VOLUME, APPLIED BEFORE THE CLAMP AND NOT AFTER.
    //
    // MEASURED on one racer's race: the NGS mix reaches **4.944x full scale** and **49.8% of
    // all grains clip**, hundreds of thousands of samples squared off. The port volume that
    // was supposed to bring that down is `11626/32768 = 0.355`, and it was applied in the
    // SINK - i.e. after this function had already clamped the mix into i16. Clamping to 1.0
    // and then scaling by 0.355 discards everything above full scale and then makes the
    // wreckage quiet; scaling by 0.355 and then clamping keeps it.
    //
    // This is also the order the hardware runs in: the mix is a sum the DSP holds at more
    // than 16 bits of headroom and `sceAudioOut` attenuates on the way out. Applying a
    // downstream gain upstream of a saturating conversion is not an optimisation, it is the
    // difference between the signal and a distorted copy of it.
    //
    // The GUEST's own buffer still receives the UNSCALED clamped mix below: on hardware the
    // port volume is downstream of that buffer, and a title that reads its own output back
    // must see what the DSP wrote, not what the speaker got.
    let port_gain = st.audio_state.gain_of(port);
    // A null buffer means "drain/stop" on real hardware; nothing to submit.
    if buf != 0 {
        let grain = format.grain as usize;
        let channels = format.channels as usize;
        // The Vita's NGS DSP would have mixed the playing AT9 voices into the master
        // buss that the title copies here. Our NGS is host-side, so we do that mix
        // now and write it into the (otherwise silent) output buffer.
        let samples = grain * channels;
        // The i16 grain that will be submitted, whether it came from our own NGS mix or out
        // of the guest's buffer. Held here so the mixed path does not have to read back what
        // it just wrote.
        let mut pcm = std::mem::take(&mut st.audio_state.scratch_pcm);
        pcm.clear();
        let mixed = st.audio_state.at9.any_playing() && !no_ngs_mix();
        if mixed {
            let mut mix = std::mem::take(&mut st.audio_state.scratch_mix);
            mix.clear();
            mix.resize(samples, 0);
            st.audio_state.at9.mix_grain(ctx, &mut mix, grain, channels, format.sample_rate);
            // >>> CLAMPED ONCE, INTO BOTH FORMS. The guest's buffer has to receive the mix
            // (the title may read its own buffer back, and on the device the DSP would have
            // written it), but the SUBMISSION does not have to come from there. It used to:
            // this wrote the bytes into guest memory and then read the very same bytes out
            // again - two full crossings of one grain, plus a per-sample `extend_from_slice`
            // and a `chunks_exact().map().collect()` that allocated a third buffer to
            // reconstruct what it had just had in hand.
            let mut bytes = std::mem::take(&mut st.audio_state.scratch_bytes);
            // >>> SIZED ONCE AND WRITTEN THROUGH SLICES, NOT GROWN A SAMPLE AT A TIME.
            //
            // A V8 worker profile of a browser race put **4.21% of the whole thread** in this
            // function's own body - the largest single named engine function in the profile,
            // for something whose job is to move one grain of PCM. The body was a `push` and
            // a two-byte `extend_from_slice` per sample, i.e. a capacity check and a
            // slow-path branch per sample, ~245,000 samples a second.
            //
            // `resize` then `chunks_exact_mut(2).zip(...)` gives the compiler three walks it
            // can prove are in bounds and equal in length, so the per-sample bookkeeping goes
            // away entirely. The arithmetic is unchanged.
            bytes.clear();
            bytes.resize(samples * 2, 0);
            pcm.clear();
            pcm.resize(samples, 0);
            // The pre-clamp peak and the clip count ride the loop that was already walking
            // every sample, so measuring the headroom costs a compare and a branch rather
            // than the second pass the old debug-gated version made. See
            // `at9::note_mix_headroom` for why this is no longer behind a tracing gate.
            let mut peak = 0i32;
            let mut clipped = 0u64;
            // Stereo is the case that matters and its gain alternates L,R,L,R; a mono port
            // uses one gain throughout. Hoisted out of the loop so the common path has no
            // remainder in it.
            let (g0, g1) = (port_gain[0], if channels > 1 { port_gain[1] } else { port_gain[0] });
            for (i, ((b, p), &s)) in
                bytes.chunks_exact_mut(2).zip(pcm.iter_mut()).zip(mix.iter()).enumerate()
            {
                peak = peak.max(s.saturating_abs());
                // The guest's buffer gets the mix as the DSP would have written it; the
                // SUBMITTED sample is the same value through the port's volume. See the
                // note where `port_gain` is read.
                let unscaled = s.clamp(-32768, 32767) as i16;
                b.copy_from_slice(&unscaled.to_le_bytes());
                let g = if i & 1 == 0 { g0 } else { g1 };
                let scaled = (s as f32 * g) as i32;
                let v = scaled.clamp(-32768, 32767) as i16;
                clipped += u64::from(i32::from(v) != scaled);
                *p = v;
            }
            super::at9::note_mix_headroom(peak, clipped, port_gain[0].min(port_gain[1]));
            ctx.write_bytes(buf, &bytes);
            // The one case the two forms could differ: a `buf` the write cannot reach. The
            // old read-back would then submit whatever was already in guest memory; this
            // submits the grain we actually mixed, which is the one the device would have
            // heard. Both are the same for every reachable buffer.
            st.audio_state.scratch_bytes = bytes;
            st.audio_state.scratch_mix = mix;
        } else {
            // Nothing of ours in it: the grain is whatever the title wrote, so it has to be
            // read. One crossing, into the same reused buffer - `read_bytes` would allocate a
            // grain-sized `Vec` per call on a path that runs at the audio rate forever, which
            // is the allocation this scratch exists to remove.
            let mut bytes = std::mem::take(&mut st.audio_state.scratch_bytes);
            bytes.clear();
            bytes.resize(samples * 2, 0);
            ctx.read_into(buf, &mut bytes);
            // The port volume applies to a guest-authored grain too - it is a property of the
            // OUTPUT, not of our mixer. There is no clipping to avoid here (the source is
            // already i16 and the gain is <= 1), so this is arithmetically what the sink used
            // to do; it lives here so there is ONE place the port volume is applied. Sized
            // and zipped for the same reason as the mixed path above.
            let (g0, g1) = (port_gain[0], if channels > 1 { port_gain[1] } else { port_gain[0] });
            pcm.resize(samples, 0);
            for (i, (p, c)) in pcm.iter_mut().zip(bytes.chunks_exact(2)).enumerate() {
                let s = i16::from_le_bytes([c[0], c[1]]);
                *p = (f32::from(s) * if i & 1 == 0 { g0 } else { g1 }) as i16;
            }
            st.audio_state.scratch_bytes = bytes;
        }
        st.audio_state.capture_pcm(&pcm);
        // Counted where the grain is SUBMITTED, so it measures what the guest actually
        // handed the device rather than what it mixed - see `produced_seconds`.
        match st.audio_state.submitted.iter_mut().find(|(p, ..)| *p == port) {
            Some((_, frames, rate)) => {
                *frames += grain as u64;
                *rate = format.sample_rate;
            }
            None => st.audio_state.submitted.push((port, grain as u64, format.sample_rate)),
        }
        st.audio.submit(backend_port, &pcm);
        st.audio_state.scratch_pcm = pcm;
        // The buffer boundaries the mix just crossed are module callbacks the title is
        // owed. On the device they fire from the DSP's update; here the update IS the
        // mix, so they fire from it.
        super::ngs::deliver_player_events(ctx, st);
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
                tracing::debug!(
                    target: "vitaslop::at9",
                    port,
                    ?vols,
                    full_scale = 32768,
                    "sceAudioOutSetVolume"
                );
                // >>> KEPT HERE, NOT PUSHED TO THE SINK. See `out_output`: the port volume
                // has to be applied to the mix BEFORE it is clamped to i16, and the sink
                // only ever sees the clamped grain. Still handed to the sink as well, for
                // backends that mix the guest's own untouched PCM - `WebAudioSink` ignores
                // it now precisely so this is not applied twice.
                for (c, v) in vols.iter().enumerate().take(2) {
                    let g = (*v as f32 / 32768.0).clamp(0.0, 1.0);
                    if let Some(p) =
                        st.audio_state.ports.iter_mut().find(|p| p.guest_port == port)
                    {
                        p.gain[c] = g;
                        // A one-channel set on a stereo port applies to both, which is what
                        // the mask form of `sceAudioOutSetVolume` means when a title sets
                        // only the left.
                        if vols.len() == 1 {
                            p.gain[1] = g;
                        }
                    }
                }
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
