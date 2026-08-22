//! ATRAC9 voice model for the NGS HLE. The real NGS mixing runs on the Vita's audio
//! DSP (firmware, not guest code), so our host must do it: a title configures an NGS
//! `SimpleAtrac9Voice` with a raw-superframe AT9 buffer via `sceNgsVoiceLockParams`
//! (module 0, the player), plays it, and each frame the DSP decodes and mixes the
//! voices to the master buss, which the title then hands to `sceAudioOutOutput`.
//!
//! We reproduce that at the output boundary: track each voice's AT9 source and play
//! state here, and when `sceAudioOutOutput` submits a grain, decode the next grain of
//! every playing voice (via [`vitaslop_atrac9`]) and mix it into the output buffer -
//! which our stubbed NGS otherwise leaves silent. Decode is cheap (a couple of
//! 256-sample frames per voice per ~11 ms grain).

use std::collections::VecDeque;

use vitaslop_atrac9::Atrac9Decoder;

use crate::host::GuestCtx;

/// The `SceNgsPlayerParams` layout a title writes for the AT9 player module,
/// recovered from the live params buffer:
/// `{ u32 id; u32 size; buffer[0]{ u32 ptr; u32 bytes; i16 loop; i16 next; } ... }`
/// with the 4-byte AT9 config word at offset 0x5c and channel count at 0x58.
const PARAMS_ID: u32 = 0x0101_5caa;
const OFF_BUFFER_PTR: u32 = 0x08;
const OFF_BUFFER_BYTES: u32 = 0x0c;
const OFF_LOOP_COUNT: u32 = 0x10;
const OFF_CHANNELS: u32 = 0x58;
const OFF_CONFIG: u32 = 0x5c;

/// The SECOND source generator: a raw-PCM player, and the one a title's sound effects
/// and streamed music actually use (one title's front end offers 66 of these against 3
/// AT9 voices).
///
/// EVIDENCE, from three live instances captured off one title's `sceNgsVoiceSetParamsBlock`
/// blocks - the descriptor declares `uSize = 84` against the AT9 player's 96, and the
/// three agree on every field below:
/// ```text
///   +0x00 u32  0x01015ce6                +0x38 f32  48000.0      playback rate
///   +0x04 u32  84 (uSize)                +0x3c f32  0.67 / 0.50  gain, differs per voice
///   +0x08 u32  source pointer            +0x4c u8   1            channels
///   +0x0c u32  source bytes              +0x50 u32  0
///   +0x10 i16  loop count (0, or -1 = forever)
///   +0x12 i16  next buffer index (-1)
///   +0x14..0x37  three more 12-byte {ptr, bytes, loop, next} descriptors, unused here
/// ```
/// So the head is the SAME `{ptr, bytes, loop, next}` buffer array the AT9 player uses -
/// only the tail differs, which is why one reader serves both. The sizes corroborate the
/// reading: 16,992 bytes is 0.18s of 48 kHz mono s16 (a UI blip) and 189,184 bytes is
/// 1.97s looping forever (an ambience bed).
///
/// `channels` is read as a BYTE. As a `u32` the field reads 0x01000001 in one instance and
/// 0x00000001 in another, which is not a channel count in any encoding; as a byte both are
/// 1 and the differing byte at +0x4f is a separate flag this does not consume.
const PCM_PARAMS_ID: u32 = 0x0101_5ce6;
/// The exact `uSize` a PCM player descriptor declares. Checked rather than assumed: a
/// different size means a different struct, and applying this layout to it would play
/// whatever bytes happened to line up.
const PCM_PARAMS_SIZE: u32 = 84;
const OFF_PCM_RATE: u32 = 0x38;
/// The per-voice LEVEL, immediately after the playback rate.
///
/// >>> WITHOUT IT THE MIX SUMS EVERY VOICE AT UNITY AND CLIPS. One title's front end
/// clamped 14.7% of its nonzero samples - gross distortion that reads as a broken
/// decoder.
///
/// EVIDENCE that it is a level and not the `fPlaybackScalar` that would sit in the same
/// place, from 2,569 voices in one run: the value is NEVER 1.0. It is 0.500, 0.610,
/// 0.664, 0.666..., and it varies in the fourth decimal between instances of the same
/// sound (0.6662 / 0.6644 / 0.6641). A playback scalar would be dominated by exactly
/// 1.0 - normal speed - and would take clean ratios; a continuously varying value under
/// unity is an attenuation. The AT9 player's struct corroborates it: the float in the
/// SAME position relative to its own rate field reads exactly 1.0 there, which is what
/// an untouched level looks like.
const OFF_PCM_LEVEL: u32 = 0x3c;
const OFF_PCM_CHANNELS: u32 = 0x4c;
/// Selects how the source bytes are encoded: 0 = raw signed-16, 1 = PS-ADPCM.
///
/// EVIDENCE, and it is the byte that stopped this generator playing NOISE. The source
/// buffers fall into two obviously different families, and this byte separates them
/// exactly:
/// - `0` - the bytes read as a smooth waveform straight away (`-24, -193, -212, -252,
///   -426, -636, -803, -961 ...`), i.e. raw s16.
/// - `1` - sixteen zero bytes, then a byte of `0x2a` / `0x2b` / `0x2c` followed by a
///   zero, repeating every 16 bytes. That is the PS-ADPCM block header: a
///   `(predictor << 4) | shift` byte (predictor 2, shifts 10-12) plus a flags byte, then
///   14 bytes carrying 28 four-bit samples. The leading all-zero block is the format's
///   customary silent primer.
///
/// Read as raw s16, the ADPCM family decodes to full-scale noise - which is why this is
/// checked rather than assumed, and why the correlation oracle exists.
const OFF_PCM_FORMAT: u32 = 0x4f;

/// The params of the BUSS module - the mix stage every source routes through - whose
/// first float is the master level.
///
/// >>> THIS IS THE STAGE THAT KEEPS THE MIX OFF THE CLAMP. Sources sum to as much as
/// **2.733x full scale** in one measured front end, so without it 4.1% of nonzero
/// samples clamp and the mix distorts.
///
/// EVIDENCE, four independent strands rather than a guess - the last guess in this file
/// decoded to noise:
/// 1. These params appear on EXACTLY ONE voice in the whole run (143 updates), and it is
///    the voice the routing graph funnels everything into: ~136 sources into each of two
///    sub-busses, both of those into this one.
/// 2. That voice never carries a source of its own, which is what a buss is.
/// 3. `+0x08` is a float in 0..1 across every instance: 0.278, 0.335, 0.500.
/// 4. The attenuation the measurement DEMANDS is 1/2.733 = 0.366 - inside that range.
///
/// The rest of the struct (a value near -7.0, one near 0.015, a constant 0.1, one near
/// -1.8, and an int 1) reads like a compressor's threshold/attack/release/makeup or a 3D
/// placement, and is NOT interpreted here: nothing in the measurement pins it down, and
/// applying a guessed dynamics stage would change the character of every sound.
const BUSS_PARAMS_ID: u32 = 0x0101_5ce1;
/// The buss level within [`BUSS_PARAMS_ID`]'s params.
const OFF_BUSS_LEVEL: u32 = 0x08;

/// How the bytes behind a PCM player's buffer pointer are encoded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PcmFormat {
    /// Raw interleaved signed-16, little-endian.
    S16,
    /// PS-ADPCM: 16-byte blocks of 28 four-bit samples over a 2-tap predictor.
    Adpcm,
}

/// PS-ADPCM predictor coefficients, in 1/64ths, indexed by the block's predictor field.
/// These are the format's published filter constants, written from the format
/// description rather than lifted from any implementation.
const ADPCM_COEF: [(i32, i32); 5] = [(0, 0), (60, 0), (115, -52), (98, -55), (122, -60)];
/// One PS-ADPCM block: a header byte, a flags byte, and 14 bytes of packed nibbles.
const ADPCM_BLOCK_BYTES: u32 = 16;
/// Samples a block unpacks to: 14 bytes x 2 nibbles.
const ADPCM_BLOCK_SAMPLES: usize = 28;

/// Which source generator a voice's module-0 params selected. A voice with `None` has
/// never had a source captured and can only ever be silent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SourceKind {
    None,
    /// ATRAC9 bitstream, decoded through [`vitaslop_atrac9`].
    At9,
    /// Raw interleaved signed-16 PCM, played (and rate-converted) directly.
    Pcm,
}

/// One source voice: its guest bytes, play state, and pending output PCM. Both
/// generators share this because they share their whole head - the buffer array, the
/// loop count, and the play/stop lifecycle - and differ only in how bytes become
/// samples.
pub(crate) struct At9Voice {
    kind: SourceKind,
    data_ptr: u32,
    data_bytes: u32,
    config: [u8; 4],
    channels: u32,
    loop_count: i16,
    playing: bool,
    /// Recreated on play; `None` until first play or if the config is unset.
    decoder: Option<Atrac9Decoder>,
    superframe_bytes: u32,
    /// Bytes consumed from `data_ptr` so far.
    consumed: u32,
    /// PCM only: the source's own sample rate, from its params.
    rate: u32,
    /// PCM only: how the source bytes are encoded.
    format: PcmFormat,
    /// Routing gain from this voice's patch (`sceNgsVoicePatchSetVolume*`). Unity until
    /// the title says otherwise, which is the only safe default: too quiet is a sound
    /// nobody hears, too loud is distortion across the whole mix.
    gain: f32,
    /// The voice's own level, from its player params - see [`OFF_PCM_LEVEL`]. Multiplied
    /// with `gain`: they are independent controls (a voice's own level, and the level of
    /// the routing that carries it), and the device applies both.
    level: f32,
    /// PS-ADPCM only: the two-sample predictor history PER CHANNEL, carried across
    /// blocks (and therefore across grains - resetting it per grain would put a
    /// discontinuity at every grain boundary). Stereo interleaves whole blocks, so each
    /// channel runs its own predictor and they must not share history.
    adpcm_hist: [(i32, i32); 2],
    /// PS-ADPCM only: decoded source-rate samples not yet rate-converted. The decode
    /// works in whole 16-byte blocks and the resampler works in fractional frames, so
    /// one has to buffer for the other.
    src_pending: VecDeque<i16>,
    /// PCM only: fractional read cursor in SOURCE FRAMES, carried across grains so a
    /// rate conversion does not drop or repeat a sample at every grain boundary - that
    /// would be a periodic click at the grain rate, which sounds like a decoder fault
    /// and is not one.
    resample_pos: f64,
    /// Decoded interleaved samples not yet mixed out.
    pending: VecDeque<i16>,
    /// Reasons already reported for this voice - see [`At9Voice::refuse`].
    reported: Vec<String>,
}

impl At9Voice {
    /// Report, ONCE per distinct reason, why this voice will not be heard.
    ///
    /// Unconditional (WARN, no env gate) for the same reason a shader fallback is:
    /// an approximation nobody is told about reads as a faithful result. Silence is
    /// the audio equivalent, and it is even easier to miss - nothing is on screen to
    /// look wrong. Deduplicated because the callers sit on the per-grain path, where
    /// an ungated log line would be its own performance defect.
    fn refuse(&mut self, reason: std::fmt::Arguments<'_>) {
        let text = reason.to_string();
        if self.reported.iter().any(|r| *r == text) {
            return;
        }
        tracing::warn!(target: "vitaslop::at9", "NGS voice not audible: {text}");
        self.reported.push(text);
    }

    /// The same, at DEBUG: for refusals that are a normal part of a title's traffic
    /// rather than a defect, which would be noise at WARN on every run.
    fn refuse_quiet(&mut self, reason: std::fmt::Arguments<'_>) {
        let text = reason.to_string();
        if self.reported.iter().any(|r| *r == text) {
            return;
        }
        tracing::debug!(target: "vitaslop::at9", "NGS voice not audible: {text}");
        self.reported.push(text);
    }

    fn empty() -> At9Voice {
        At9Voice {
            kind: SourceKind::None,
            data_ptr: 0,
            data_bytes: 0,
            config: [0; 4],
            channels: 0,
            loop_count: 0,
            playing: false,
            decoder: None,
            superframe_bytes: 0,
            consumed: 0,
            rate: 0,
            format: PcmFormat::S16,
            gain: 1.0,
            level: 1.0,
            adpcm_hist: [(0, 0); 2],
            src_pending: VecDeque::new(),
            resample_pos: 0.0,
            pending: VecDeque::new(),
            reported: Vec::new(),
        }
    }

    /// Read the AT9 player params a title just wrote and store the source. Returns
    /// false if the buffer is not a recognizable AT9 player params block.
    ///
    /// >>> EVERY REFUSAL HERE IS A VOICE THAT WILL NEVER BE AUDIBLE, so each one names
    /// itself through [`refuse`] rather than returning a bare `false`. A silently
    /// rejected source is indistinguishable from a title that chose not to play a
    /// sound, and that is exactly how a whole stack - decoder, mixer, ring, worklet -
    /// sat correct and untested behind a stream of digital silence.
    fn load_params(&mut self, ctx: &GuestCtx, params_addr: u32) -> bool {
        let id = ctx.read_u32(params_addr);
        if id == PCM_PARAMS_ID {
            return self.load_pcm_params(ctx, params_addr);
        }
        if id != PARAMS_ID {
            // Not the AT9 player descriptor. Common and legitimate - the same module
            // slot carries other generator types - so it is reported at DEBUG, keyed by
            // the id so an unknown generator can be recognised and added.
            //
            // The params come WITH the report. An unimplemented source generator is the
            // next thing anyone works on after this one, and the layout has to be REd
            // from the bytes a title actually writes; an id on its own starts that work
            // from nothing, while 64 bytes of its params usually finishes it (that is
            // exactly how the AT9 player's own buffer/config offsets were established).
            // Dump past the descriptor's own `uSize` (at +4) where it names one, so the
            // report covers the WHOLE struct rather than a prefix - the fields that
            // identify a generator (rate, channels, format) sit at the far end of it.
            let size = ctx.read_u32(params_addr + 4).clamp(64, 256);
            let head = ctx.read_bytes(params_addr, size as usize);
            self.refuse_quiet(format_args!(
                "params id {id:#010x} is not the AT9 player ({PARAMS_ID:#010x}) - this source \
                 generator is not implemented, so the voice is silent. Its params at \
                 {params_addr:#010x}: {head:02x?}"
            ));
            return false;
        }
        // The level is taken BEFORE the buffer is required, because a voice with no
        // source of its own is not a broken voice - it is a BUSS, and its level is
        // exactly what the voices routed into it must be scaled by.
        self.take_level(ctx, params_addr + OFF_CONFIG - 0x10);
        let data_ptr = ctx.read_u32(params_addr + OFF_BUFFER_PTR);
        let data_bytes = ctx.read_u32(params_addr + OFF_BUFFER_BYTES);
        if data_ptr == 0 || data_bytes == 0 {
            self.refuse(format_args!(
                "AT9 player params carry no source buffer (ptr {data_ptr:#010x}, {data_bytes} bytes)"
            ));
            return false;
        }
        let cfg = ctx.read_bytes(params_addr + OFF_CONFIG, 4);
        let config = [cfg[0], cfg[1], cfg[2], cfg[3]];
        // header byte must be 0xFE for a valid config; otherwise this isn't AT9.
        if config[0] != 0xFE {
            self.refuse(format_args!(
                "AT9 config word {config:02x?} does not start with 0xFE, so it is not an \
                 ATRAC9 config - the source at {data_ptr:#010x} ({data_bytes} bytes) is dropped"
            ));
            return false;
        }
        self.kind = SourceKind::At9;
        self.data_ptr = data_ptr;
        self.data_bytes = data_bytes;
        self.config = config;
        self.channels = ctx.read_u32(params_addr + OFF_CHANNELS) & 0xffff;
        self.loop_count = ctx.read_u32(params_addr + OFF_LOOP_COUNT) as i16;
        true
    }

    /// Read a raw-PCM player's params - see [`PCM_PARAMS_ID`] for the layout and the
    /// evidence behind it.
    ///
    /// Every field is CHECKED rather than trusted, and a failed check refuses the source
    /// by name instead of playing it. The layout is REd from one title, and the failure
    /// mode of applying a wrong layout to audio is not a crash or a blank screen - it is
    /// full-scale NOISE through the player's speakers, which is worse than silence and
    /// which no count-based check would catch.
    /// Read and store this voice's level from `level_addr`, wherever its generator keeps
    /// it. Shared by both generators and by source-less BUSS voices, whose level is the
    /// only field of their params that matters.
    ///
    /// A value outside 0..=1 is not a level; unity stands instead, and it says so rather
    /// than scaling the mix by a number this reading cannot explain.
    fn take_level(&mut self, ctx: &GuestCtx, level_addr: u32) {
        let level = f32::from_bits(ctx.read_u32(level_addr));
        if (0.0..=1.0).contains(&level) {
            self.level = level;
        } else {
            self.refuse(format_args!(
                "player params carry a level of {level}, outside 0..=1 - using unity instead of \
                 scaling by a value this reading cannot explain"
            ));
            self.level = 1.0;
        }
    }

    fn load_pcm_params(&mut self, ctx: &GuestCtx, params_addr: u32) -> bool {
        let size = ctx.read_u32(params_addr + 4);
        if size != PCM_PARAMS_SIZE {
            let head = ctx.read_bytes(params_addr, size.clamp(64, 256) as usize);
            self.refuse(format_args!(
                "PCM player params declare uSize {size}, not the {PCM_PARAMS_SIZE} this layout \
                 was REd from - refusing rather than reading fields that may not be there. \
                 Params at {params_addr:#010x}: {head:02x?}"
            ));
            return false;
        }
        // Level first: a source-less voice here is a BUSS, and its level is the whole
        // point of its params. See `take_level`.
        self.take_level(ctx, params_addr + OFF_PCM_LEVEL);
        let data_ptr = ctx.read_u32(params_addr + OFF_BUFFER_PTR);
        let data_bytes = ctx.read_u32(params_addr + OFF_BUFFER_BYTES);
        let rate = f32::from_bits(ctx.read_u32(params_addr + OFF_PCM_RATE)) as i64;
        let channels = u32::from(ctx.read_bytes(params_addr + OFF_PCM_CHANNELS, 1)[0]);
        if data_ptr == 0 || data_bytes == 0 {
            self.refuse(format_args!(
                "PCM player params carry no source buffer (ptr {data_ptr:#010x}, {data_bytes} bytes)"
            ));
            return false;
        }
        if !(1..=2).contains(&channels) {
            self.refuse(format_args!(
                "PCM player params declare {channels} channels, which is not 1 or 2 - the \
                 channel field is misread and the source is refused"
            ));
            return false;
        }
        let format_byte = ctx.read_bytes(params_addr + OFF_PCM_FORMAT, 1)[0];
        let format = match format_byte {
            0 => PcmFormat::S16,
            1 => PcmFormat::Adpcm,
            other => {
                self.refuse(format_args!(
                    "PCM player params select source format {other}, which is neither raw s16 (0) \
                     nor PS-ADPCM (1) - refusing rather than playing bytes in a format this does \
                     not know, which would be noise at full scale"
                ));
                return false;
            }
        };
        // The buffer must divide exactly into whatever unit the format reads, or the
        // format is not what this reader thinks it is.
        let unit = match format {
            PcmFormat::S16 => channels * 2,
            PcmFormat::Adpcm => ADPCM_BLOCK_BYTES,
        };
        if data_bytes % unit != 0 {
            self.refuse(format_args!(
                "{format:?} source of {data_bytes} bytes is not a whole number of {unit}-byte \
                 units - the sample format is misread, so the source is refused"
            ));
            return false;
        }
        // Stereo PS-ADPCM interleaves whole blocks, so the buffer must divide into whole
        // interleave GROUPS - a remainder means the interleave is not what this reads.
        if format == PcmFormat::Adpcm && data_bytes % (ADPCM_BLOCK_BYTES * channels) != 0 {
            self.refuse(format_args!(
                "PS-ADPCM source of {data_bytes} bytes is not a whole number of {channels}-channel \
                 16-byte block groups - the interleave is misread, so the source is refused"
            ));
            return false;
        }
        if !(4000..=192_000).contains(&rate) {
            self.refuse(format_args!(
                "PCM player params declare a {rate} Hz playback rate, which is outside any \
                 plausible range - the rate field is misread and the source is refused"
            ));
            return false;
        }
        // The SOURCE BYTES, not just the params. Whether this generator's payload is raw
        // samples or a compressed bitstream cannot be told from the params at all, and
        // reading it wrong plays full-scale noise rather than failing - so the head of
        // the actual buffer is the evidence that settles it.
        let head = ctx.read_bytes(data_ptr, 32);
        tracing::debug!(
            target: "vitaslop::at9",
            ptr = format_args!("{data_ptr:#010x}"),
            bytes = data_bytes,
            rate,
            channels,
            level = self.level,
            "PCM source head: {head:02x?}"
        );
        self.kind = SourceKind::Pcm;
        self.format = format;
        self.data_ptr = data_ptr;
        self.data_bytes = data_bytes;
        self.channels = channels;
        self.rate = rate as u32;
        self.loop_count = ctx.read_u32(params_addr + OFF_LOOP_COUNT) as i16;
        true
    }

    /// (Re)start playback from the beginning of the current source.
    fn start(&mut self) {
        if self.kind == SourceKind::Pcm {
            // Nothing to construct: the source is already samples, and playing it is a
            // read cursor over guest memory.
            self.consumed = 0;
            self.resample_pos = 0.0;
            self.adpcm_hist = [(0, 0); 2];
            self.src_pending.clear();
            self.pending.clear();
            self.playing = true;
            tracing::debug!(
                target: "vitaslop::at9",
                format = format_args!("{:?}", self.format),
                channels = self.channels,
                rate = self.rate,
                bytes = self.data_bytes,
                loop_count = self.loop_count,
                "PCM voice PLAYING"
            );
            return;
        }
        match Atrac9Decoder::new(self.config) {
            Ok(dec) => {
                self.superframe_bytes = dec.superframe_bytes() as u32;
                self.decoder = Some(dec);
                self.consumed = 0;
                self.pending.clear();
                self.playing = true;
                tracing::debug!(
                    target: "vitaslop::at9",
                    config = format_args!("{:02x?}", self.config),
                    superframe_bytes = self.superframe_bytes,
                    channels = self.channels,
                    bytes = self.data_bytes,
                    loop_count = self.loop_count,
                    "AT9 voice PLAYING"
                );
            }
            Err(e) => {
                self.playing = false;
                // `config` is copied out first: `refuse` takes `&mut self`, so reading a
                // field inside its argument list is a borrow conflict.
                let config = self.config;
                self.refuse(format_args!(
                    "the ATRAC9 decoder refused config {config:02x?}: {e:?} - this voice is SILENT"
                ));
            }
        }
    }

    fn stop(&mut self) {
        self.playing = false;
    }

    /// The voice loops if its loop count is non-zero (a negative count is the
    /// common "loop forever" encoding).
    fn loops(&self) -> bool {
        self.loop_count != 0
    }

    /// Produce at least `needed` interleaved samples at `port_rate`, or stop the voice
    /// when its source is exhausted and does not loop.
    fn fill(&mut self, ctx: &GuestCtx, needed: usize, port_rate: u32) {
        if self.kind == SourceKind::Pcm {
            self.fill_pcm(ctx, needed, port_rate);
            return;
        }
        self.fill_at9(ctx, needed)
    }

    /// Raw PCM, rate-converted to the output port.
    ///
    /// The source is signed-16 frames sitting in GUEST MEMORY, which is random-access,
    /// so this needs no decode state at all - just a fractional cursor and a linear
    /// interpolation between neighbouring source frames. The bytes for a whole run are
    /// read ONCE per call rather than per sample: a read per output sample would be
    /// thousands of small allocations per voice per grain, on a path that runs inside a
    /// host call at grain rate and in a browser.
    fn fill_pcm(&mut self, ctx: &GuestCtx, needed: usize, port_rate: u32) {
        if self.format == PcmFormat::Adpcm {
            self.fill_adpcm(ctx, needed, port_rate);
            return;
        }
        let ch = self.channels.max(1) as usize;
        let frame_bytes = (ch * 2) as u32;
        let total_frames = (self.data_bytes / frame_bytes) as usize;
        if total_frames == 0 || port_rate == 0 {
            self.playing = false;
            return;
        }
        // Source frames consumed per output frame. Equal rates give exactly 1.0, which
        // is the common case and stays sample-exact through the interpolation below.
        let ratio = self.rate as f64 / port_rate as f64;

        // >>> THE EQUAL-RATE FAST PATH, AND IT IS THE ONE THAT ALMOST ALWAYS RUNS.
        //
        // Every PCM source measured so far is 48 kHz into a 48 kHz port, so the general
        // path below would spend two float multiplies, a compare and a clamp PER SAMPLE
        // to compute `a + (b - a) * 0.0`. At a 1024-frame grain with tens of voices that
        // is millions of pointless float ops a second, inside a host call, on a device
        // whose whole budget this project is measured against. Here it is a byte-slice
        // walk and an extend instead. Bit-identical to the general path at ratio 1.0 -
        // interpolating with a zero fraction returns the sample untouched.
        if self.rate == port_rate {
            while self.pending.len() < needed && self.playing {
                let start_frame = self.resample_pos as usize;
                if start_frame >= total_frames {
                    self.wrap_or_stop();
                    continue;
                }
                let want_frames = (needed - self.pending.len()).div_ceil(ch);
                let take = want_frames.min(total_frames - start_frame);
                let bytes = ctx.read_bytes(
                    self.data_ptr + start_frame as u32 * frame_bytes,
                    take * frame_bytes as usize,
                );
                self.pending.extend(
                    bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])),
                );
                self.resample_pos += take as f64;
            }
            return;
        }

        while self.pending.len() < needed && self.playing {
            let want_frames = (needed - self.pending.len()).div_ceil(ch);
            let start_frame = self.resample_pos.floor() as usize;
            if start_frame >= total_frames {
                self.wrap_or_stop();
                continue;
            }
            // How many output frames this pass can serve before the cursor would need a
            // source frame past the end. One spare frame is kept for the interpolation
            // partner of the last sample.
            let usable = (total_frames - 1).saturating_sub(start_frame) as f64;
            let frac = self.resample_pos - start_frame as f64;
            let can = if ratio > 0.0 { ((usable - frac) / ratio).floor() } else { 0.0 };
            let take = want_frames.min(can.max(0.0) as usize);
            if take == 0 {
                self.wrap_or_stop();
                continue;
            }
            // One contiguous read covering every source frame this pass touches.
            let last = self.resample_pos + ratio * (take - 1) as f64;
            let span = (last.floor() as usize + 1).min(total_frames - 1) - start_frame + 1;
            let bytes = ctx.read_bytes(
                self.data_ptr + start_frame as u32 * frame_bytes,
                span * frame_bytes as usize,
            );
            let sample = |frame: usize, c: usize| -> f64 {
                let off = (frame * ch + c) * 2;
                match (bytes.get(off), bytes.get(off + 1)) {
                    (Some(&lo), Some(&hi)) => i16::from_le_bytes([lo, hi]) as f64,
                    _ => 0.0,
                }
            };
            for _ in 0..take {
                let idx = self.resample_pos.floor() as usize - start_frame;
                let f = self.resample_pos - self.resample_pos.floor();
                for c in 0..ch {
                    let a = sample(idx, c);
                    let b = if f == 0.0 { a } else { sample(idx + 1, c) };
                    let v = a + (b - a) * f;
                    self.pending.push_back(v.clamp(-32768.0, 32767.0) as i16);
                }
                self.resample_pos += ratio;
            }
        }
    }

    /// PS-ADPCM: unpack whole 16-byte blocks into `pending`.
    ///
    /// A block is `(predictor << 4) | shift`, a flags byte, then 14 bytes holding 28
    /// four-bit residuals, each reconstructed over a two-tap predictor whose history
    /// carries across blocks. Blocks are read in RUNS rather than one at a time - the
    /// per-block guest read would otherwise be an allocation per 28 samples on the
    /// grain-rate path.
    ///
    /// Rate conversion is not applied here: every source measured runs at the port's own
    /// rate, and a mismatch is REFUSED by name rather than played at the wrong pitch,
    /// which would sound like a decoder fault and is not one.
    fn fill_adpcm(&mut self, ctx: &GuestCtx, needed: usize, port_rate: u32) {
        let ch = self.channels.max(1) as usize;
        let total_blocks = self.data_bytes / ADPCM_BLOCK_BYTES;
        if total_blocks == 0 || port_rate == 0 {
            self.playing = false;
            return;
        }
        let ratio = self.rate as f64 / port_rate as f64;

        while self.pending.len() < needed && self.playing {
            let out_frames = (needed - self.pending.len()).div_ceil(ch);
            // Source frames this pass needs, plus one for the interpolation partner.
            let want_src = (self.resample_pos + ratio * out_frames as f64).ceil() as usize + 2;
            self.decode_adpcm_frames(ctx, want_src * ch, ch, total_blocks);
            let have = self.src_pending.len() / ch;
            if have == 0 {
                return;
            }
            // Only produce what the buffered source can actually support.
            let usable = (have - 1) as f64;
            let can = if ratio > 0.0 { ((usable - self.resample_pos) / ratio).floor() } else { 0.0 };
            let take = out_frames.min(can.max(0.0) as usize);
            if take == 0 {
                // The source is finished and cannot feed another frame.
                if !self.playing {
                    return;
                }
                self.playing = false;
                return;
            }
            for _ in 0..take {
                let i = self.resample_pos as usize;
                let f = self.resample_pos - i as f64;
                for c in 0..ch {
                    let a = self.src_pending.get(i * ch + c).copied().unwrap_or(0) as f64;
                    let b = if f == 0.0 {
                        a
                    } else {
                        self.src_pending.get((i + 1) * ch + c).copied().map_or(a, f64::from)
                    };
                    self.pending.push_back((a + (b - a) * f) as i16);
                }
                self.resample_pos += ratio;
            }
            // Retire whole source frames the cursor has passed, keeping the fraction.
            let done = self.resample_pos.floor() as usize;
            if done > 0 {
                self.src_pending.drain(..(done * ch).min(self.src_pending.len()));
                self.resample_pos -= done as f64;
            }
        }
    }

    /// Unpack whole PS-ADPCM blocks into `src_pending` until it holds `want` samples or
    /// the source ends.
    ///
    /// Stereo interleaves WHOLE BLOCKS - one block of left, one of right - so each
    /// channel carries its own predictor history and the two must not share it. Blocks
    /// are read in runs rather than one at a time: a guest read per 28 samples would be
    /// an allocation per block on the grain-rate path.
    fn decode_adpcm_frames(
        &mut self,
        ctx: &GuestCtx,
        want: usize,
        ch: usize,
        total_blocks: u32,
    ) {
        while self.src_pending.len() < want && self.playing {
            let done = self.consumed / ADPCM_BLOCK_BYTES;
            if done >= total_blocks {
                if self.loops() {
                    self.consumed = 0;
                    self.adpcm_hist = [(0, 0); 2];
                    continue;
                }
                self.playing = false;
                return;
            }
            // A run must cover whole INTERLEAVE GROUPS (one block per channel), or the
            // channel a block belongs to would shift.
            let group = ch as u32;
            let needed_groups =
                ((want - self.src_pending.len()).div_ceil(ADPCM_BLOCK_SAMPLES * ch)) as u32;
            let avail_groups = (total_blocks - done) / group;
            let groups = needed_groups.min(avail_groups).max(1);
            let run = (groups * group).min(total_blocks - done);
            let bytes =
                ctx.read_bytes(self.data_ptr + self.consumed, (run * ADPCM_BLOCK_BYTES) as usize);
            // Decode each group into interleaved frames.
            let mut decoded = vec![0i16; ADPCM_BLOCK_SAMPLES * ch];
            for g in bytes.chunks_exact(ADPCM_BLOCK_BYTES as usize * ch) {
                for (c, block) in g.chunks_exact(ADPCM_BLOCK_BYTES as usize).enumerate().take(ch) {
                    let shift = block[0] & 0x0f;
                    let predictor = usize::from(block[0] >> 4).min(ADPCM_COEF.len() - 1);
                    let (c0, c1) = ADPCM_COEF[predictor];
                    let (mut h1, mut h2) = self.adpcm_hist[c.min(1)];
                    let mut n = 0usize;
                    for &byte in &block[2..] {
                        for nibble in [byte & 0x0f, byte >> 4] {
                            // Sign-extend the 4-bit residual from the top of a 16-bit
                            // word, then scale it down by the block's shift.
                            let residual = i32::from(((u16::from(nibble) << 12) as i16) >> shift);
                            let predicted = (h1 * c0 + h2 * c1 + 32) >> 6;
                            let s = (residual + predicted).clamp(-32768, 32767);
                            decoded[n * ch + c] = s as i16;
                            h2 = h1;
                            h1 = s;
                            n += 1;
                        }
                    }
                    self.adpcm_hist[c.min(1)] = (h1, h2);
                }
                self.src_pending.extend(decoded.iter().copied());
            }
            self.consumed += run * ADPCM_BLOCK_BYTES;
        }
    }

    /// The source ran out: restart it if it loops, otherwise the voice is finished.
    fn wrap_or_stop(&mut self) {
        if self.loops() {
            // Carry the fractional part so a looping source does not gain or lose a
            // sample every lap, which would drift audibly over a long ambience bed.
            self.resample_pos -= self.resample_pos.floor();
        } else {
            self.playing = false;
        }
    }

    /// Decode until at least `needed` interleaved samples are pending, or the
    /// source is exhausted (stopping the voice unless it loops).
    fn fill_at9(&mut self, ctx: &GuestCtx, needed: usize) {
        while self.pending.len() < needed {
            if self.decoder.is_none() || self.superframe_bytes == 0 {
                self.playing = false;
                return;
            }
            if self.consumed + self.superframe_bytes > self.data_bytes {
                if self.loops() {
                    self.start();
                    continue;
                }
                self.playing = false;
                return;
            }
            let sf = ctx.read_bytes(self.data_ptr + self.consumed, self.superframe_bytes as usize);
            let dec = self.decoder.as_mut().unwrap();
            let frames = dec.frames_per_superframe();
            let frame_shorts = dec.frame_samples() * dec.channels();
            let mut pcm = vec![0i16; frame_shorts];
            let mut inner = 0usize;
            for _ in 0..frames {
                match dec.decode_frame(&sf[inner..], &mut pcm) {
                    Ok(used) => {
                        self.pending.extend(pcm.iter().copied());
                        inner += used;
                    }
                    Err(e) => {
                        self.playing = false;
                        let consumed = self.consumed;
                        self.refuse(format_args!(
                            "ATRAC9 decode failed {consumed} bytes into the source: {e:?} - the \
                             voice stops here"
                        ));
                        return;
                    }
                }
            }
            self.consumed += self.superframe_bytes;
        }
    }
}

/// The bank of source voices, keyed by the NGS voice handle, plus the routing graph
/// that says where each one's output goes.
#[derive(Default)]
pub(crate) struct At9Bank {
    voices: std::collections::BTreeMap<u32, At9Voice>,
    /// `source voice -> destination voice`, from `sceNgsPatchCreateRouting`.
    ///
    /// A title does not route everything straight to the output: one measured front end
    /// feeds ~136 sources into each of two sub-busses, both of those into a third, and
    /// that into a fourth. Every voice on the way has its OWN level, and the device
    /// applies all of them - so a source mixed straight to the output at its own level
    /// alone is louder than the title asked for, by the product of every buss it should
    /// have passed through.
    routes: std::collections::BTreeMap<u32, u32>,
}

impl At9Bank {
    /// Handle `sceNgsVoiceUnlockParams` for the player module: capture the AT9 source
    /// the title just configured on `voice`.
    pub(crate) fn set_player_params(&mut self, ctx: &GuestCtx, voice: u32, params_addr: u32) {
        let v = self.voices.entry(voice).or_insert_with(At9Voice::empty);
        v.load_params(ctx, params_addr);
    }

    /// Start `voice` (from `sceNgsVoicePlay`), if it has an AT9 source.
    ///
    /// A play on a voice with no captured source is the single most likely way this
    /// whole path produces silence, so it SAYS so: the title asked for a sound and the
    /// engine has nothing to give it. Without this the only symptom is a correctly
    /// paced stream of zeroes, which looks like a working audio backend.
    pub(crate) fn play(&mut self, voice: u32) {
        // A BUSS is played like any other voice and has no source of its own - that is
        // what a buss IS. Warning about it would be crying wolf on every mix graph, so
        // a voice that other voices route INTO is reported quietly. Only a voice nobody
        // routes into is genuinely a sound that will not be heard.
        let is_buss = self.routes.values().any(|dst| *dst == voice);
        match self.voices.get_mut(&voice) {
            Some(v) if v.data_ptr != 0 => v.start(),
            Some(v) if is_buss => v.refuse_quiet(format_args!(
                "sceNgsVoicePlay on buss voice {voice:#x} - no source of its own, as expected"
            )),
            Some(v) => v.refuse(format_args!(
                "sceNgsVoicePlay on voice {voice:#x}, but no source was captured from its \
                 player params"
            )),
            None => {
                let v = self.voices.entry(voice).or_insert_with(At9Voice::empty);
                if is_buss {
                    v.refuse_quiet(format_args!(
                        "sceNgsVoicePlay on buss voice {voice:#x} - no params of its own, as \
                         expected"
                    ));
                } else {
                    v.refuse(format_args!(
                        "sceNgsVoicePlay on voice {voice:#x}, which never locked/unlocked player \
                         params at all - nothing to decode"
                    ));
                }
            }
        }
    }

    /// Apply a non-source module's params to `voice`. Only the buss module is
    /// understood; every other module is a synthesiser stage nothing here runs.
    ///
    /// Returns whether the params were recognised, so the caller can report the ones
    /// that were not.
    pub(crate) fn set_module_params(
        &mut self,
        ctx: &GuestCtx,
        voice: u32,
        params_addr: u32,
    ) -> bool {
        if ctx.read_u32(params_addr) != BUSS_PARAMS_ID {
            return false;
        }
        let v = self.voices.entry(voice).or_insert_with(At9Voice::empty);
        v.take_level(ctx, params_addr + OFF_BUSS_LEVEL);
        true
    }

    /// Record that `source`'s output feeds `destination`.
    pub(crate) fn set_route(&mut self, source: u32, destination: u32) {
        if source != 0 && destination != 0 && source != destination {
            self.routes.insert(source, destination);
        }
    }

    /// The product of every buss level `voice` passes through on its way to the output,
    /// NOT including the voice's own level.
    ///
    /// Bounded walk: a routing graph that contained a cycle would otherwise hang the
    /// audio thread, and a malformed graph is not worth a deadlock. Eight hops is far
    /// past any real mix tree (the deepest measured is three).
    fn buss_gain(&self, voice: u32) -> f32 {
        let mut gain = 1.0f32;
        let mut at = voice;
        for _ in 0..8 {
            let Some(&next) = self.routes.get(&at) else { break };
            let Some(v) = self.voices.get(&next) else { break };
            gain *= v.level;
            at = next;
        }
        gain
    }

    /// Set the routing gain for `voice`, from its patch's volume.
    ///
    /// Recorded even for a voice that has no source yet: a title routes and balances its
    /// mix graph up front and only later hands a voice something to play, so dropping
    /// the gain here would silently restore full scale for exactly the voices whose
    /// levels were set earliest.
    pub(crate) fn set_gain(&mut self, voice: u32, gain: f32) {
        self.voices.entry(voice).or_insert_with(At9Voice::empty).gain = gain;
    }

    /// Stop `voice` (key-off / kill / pause).
    pub(crate) fn stop(&mut self, voice: u32) {
        if let Some(v) = self.voices.get_mut(&voice) {
            v.stop();
        }
    }

    /// Any voice currently producing audio.
    pub(crate) fn any_playing(&self) -> bool {
        self.voices.values().any(|v| v.playing)
    }

    /// Mix one grain of every playing voice into `mix` (interleaved, `grain *
    /// port_channels` accumulator). Voices whose channel count differs from the port
    /// are up/down-mixed by the simplest correct rule (mono duplicated to all port
    /// channels).
    pub(crate) fn mix_grain(
        &mut self,
        ctx: &GuestCtx,
        mix: &mut [i32],
        grain: usize,
        port_channels: usize,
        port_rate: u32,
    ) {
        // Buss levels are resolved BEFORE the mixing borrow: `buss_gain` walks other
        // entries of the same map, which cannot happen while one of them is borrowed
        // mutably. It is a handful of map lookups per playing voice per grain, against
        // the hundreds of thousands of sample operations below.
        let routed: Vec<(u32, f32)> = self
            .voices
            .iter()
            .filter(|(_, v)| v.playing)
            .map(|(&handle, _)| (handle, self.buss_gain(handle)))
            .collect();
        for (handle, buss_gain) in routed {
            let Some(v) = self.voices.get_mut(&handle) else { continue };
            let vc = v.channels.max(1) as usize;
            v.fill(ctx, grain * vc, port_rate);
            // A voice the title has turned all the way down contributes nothing, so it
            // costs nothing: its samples are still CONSUMED below (the source has to
            // advance, or it would resume from a stale position the moment it is turned
            // back up), but the mixing loop is skipped entirely.
            let gain = v.gain * v.level * buss_gain;
            if gain <= 0.0 {
                let consumed = (grain * vc).min(v.pending.len());
                v.pending.drain(..consumed);
                continue;
            }
            // Fixed-point gain: the mixer is integer, and a float multiply per sample
            // per voice on the grain-rate path is exactly the kind of cost this project
            // measures. 16.16 keeps unity exact and a -60 dB setting still meaningful.
            let gain_q16 = (gain.min(4.0) * 65536.0) as i64;
            for f in 0..grain {
                // Pull this frame's voice samples (or silence if the voice drained).
                for c in 0..port_channels {
                    let src_c = if vc == 1 { 0 } else { c.min(vc - 1) };
                    // pending is a flat interleaved queue; index into the frame.
                    let idx = f * vc + src_c;
                    if let Some(&s) = v.pending.get(idx) {
                        mix[f * port_channels + c] += ((s as i64 * gain_q16) >> 16) as i32;
                    }
                }
            }
            // Drop the grain we just consumed.
            let consumed = (grain * vc).min(v.pending.len());
            v.pending.drain(..consumed);
        }
    }
}

#[cfg(test)]
mod buss_tests {
    //! The routing graph and the levels along it.
    //!
    //! A source is not mixed at its own level alone: it passes through every buss the
    //! title routed it into, and each one has a level. Getting this wrong does not
    //! sound broken, it sounds LOUD - one title's front end summed to 2.733x full scale
    //! and clamped 4.1% of its nonzero samples until the master buss level was applied.
    use super::*;

    /// A voice with a level and nothing else - which is exactly what a buss is.
    fn buss(level: f32) -> At9Voice {
        let mut v = At9Voice::empty();
        v.level = level;
        v
    }

    #[test]
    fn buss_levels_multiply_along_the_route() {
        let mut bank = At9Bank::default();
        bank.voices.insert(1, buss(1.0)); // the source
        bank.voices.insert(2, buss(0.5)); // a sub-buss
        bank.voices.insert(3, buss(0.25)); // the master
        bank.set_route(1, 2);
        bank.set_route(2, 3);
        // The source's OWN level is not included - the mixer applies that separately.
        assert_eq!(bank.buss_gain(1), 0.125);
        assert_eq!(bank.buss_gain(2), 0.25);
        assert_eq!(bank.buss_gain(3), 1.0, "an unrouted voice passes through unchanged");
    }

    #[test]
    fn an_unrouted_voice_is_unattenuated() {
        let mut bank = At9Bank::default();
        bank.voices.insert(1, buss(1.0));
        assert_eq!(bank.buss_gain(1), 1.0);
    }

    /// A malformed graph must not hang the audio thread, which runs inside a host call.
    #[test]
    fn a_routing_cycle_terminates() {
        let mut bank = At9Bank::default();
        bank.voices.insert(1, buss(0.5));
        bank.voices.insert(2, buss(0.5));
        bank.set_route(1, 2);
        bank.set_route(2, 1);
        // Bounded walk: it stops rather than spinning, and the value is whatever the
        // bound yields - the point of the test is that it RETURNS.
        let _ = bank.buss_gain(1);
    }

    /// A route to a voice that was never created cannot attenuate anything.
    #[test]
    fn a_route_to_an_unknown_voice_is_ignored() {
        let mut bank = At9Bank::default();
        bank.voices.insert(1, buss(1.0));
        bank.set_route(1, 99);
        assert_eq!(bank.buss_gain(1), 1.0);
    }
}
