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
    /// Whether [`At9Voice::take_level`] ever ran for this voice, as opposed to the level
    /// being the 1.0 this struct starts at.
    ///
    /// >>> A DEFAULT AND A MEASUREMENT LOOK IDENTICAL ONCE THEY ARE BOTH `1.0`, and that
    /// distinction is the open question about this mixer. The race sums to ~4.9x full scale
    /// and clips half its grains; the output port is not the cause (its mean over the grains
    /// is 0.995) and no voice is unrouted (0.0 a grain), so the attenuation that is missing
    /// is a LEVEL somewhere in the graph. A buss voice whose params arrived through a module
    /// this engine does not parse keeps `level = 1.0` and contributes a silent no-op to
    /// `buss_gain` - indistinguishable, in the mix, from a buss the title really did leave at
    /// unity. This flag is what tells those two apart.
    level_set: bool,
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
            level_set: false,
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
    /// >>> COUNTED, NOT WARNED PER VALUE. See [`note_level_out_of_range`].
    fn take_level(&mut self, ctx: &GuestCtx, level_addr: u32) {
        let level = f32::from_bits(ctx.read_u32(level_addr));
        self.level_set = true;
        if (0.0..=1.0).contains(&level) {
            self.level = level;
        } else {
            note_level_out_of_range(level);
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
        if self.playing {
            note_voice_stopped();
        }
        self.playing = false;
    }

    /// The voice loops if its loop count is non-zero (a negative count is the
    /// common "loop forever" encoding).
    fn loops(&self) -> bool {
        self.loop_count != 0
    }

    /// Produce at least `needed` interleaved samples at `port_rate`, or stop the voice
    /// when its source is exhausted and does not loop.
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn fill(&mut self, ctx: &GuestCtx, needed: usize, port_rate: u32, scratch: &mut MixScratch) {
        if self.kind == SourceKind::Pcm {
            self.fill_pcm(ctx, needed, port_rate, scratch);
            return;
        }
        self.fill_at9(ctx, needed, scratch)
    }

    /// Raw PCM, rate-converted to the output port.
    ///
    /// The source is signed-16 frames sitting in GUEST MEMORY, which is random-access,
    /// so this needs no decode state at all - just a fractional cursor and a linear
    /// interpolation between neighbouring source frames. The bytes for a whole run are
    /// read ONCE per call rather than per sample: a read per output sample would be
    /// thousands of small allocations per voice per grain, on a path that runs inside a
    /// host call at grain rate and in a browser.
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn fill_pcm(&mut self, ctx: &GuestCtx, needed: usize, port_rate: u32, scratch: &mut MixScratch) {
        if self.format == PcmFormat::Adpcm {
            self.fill_adpcm(ctx, needed, port_rate, scratch);
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
                // Into the BANK'S scratch, not a fresh `Vec`. `read_bytes` allocates, and this
                // runs once per playing voice per grain - 21 voices at 120 grains a second on
                // one racer's race - so it was thousands of allocations a second on the audio
                // path for bytes that are consumed immediately. `scratch.src` is the buffer
                // this function is already handed for exactly that reason; the ATRAC9 path
                // next door has always used it.
                let want_bytes = take * frame_bytes as usize;
                scratch.src.clear();
                scratch.src.resize(want_bytes, 0);
                ctx.read_into(self.data_ptr + start_frame as u32 * frame_bytes, &mut scratch.src);
                self.pending.reserve(take * ch);
                self.pending.extend(
                    scratch.src.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])),
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
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn fill_adpcm(
        &mut self,
        ctx: &GuestCtx,
        needed: usize,
        port_rate: u32,
        scratch: &mut MixScratch,
    ) {
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
            self.decode_adpcm_frames(ctx, want_src * ch, ch, total_blocks, scratch);
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
            // >>> FLATTENING THIS LOOP'S SOURCE WAS TRIED AND IS REFUTED. Do not re-do it.
            //
            // A V8 worker profile built with `profile-symbols` (so the audio path is not all
            // inlined into `out_output`) puts **2.08% of the whole thread right here** - more
            // than the ATRAC9 decoder at 1.87% and four times the mixer at 0.48%. The obvious
            // reading is that `src_pending` is a ring, so each of the two samples an
            // interpolation needs costs a `VecDeque::get` - a bounds check and a wrap test -
            // and each output sample is an individual `push_back` onto a second ring.
            //
            // Resolving the wrap once into a flat slice, reserving the output up front and
            // interpolating in f32 instead of f64 - the same rewrite that DID pay in
            // `mix_grain` twenty lines up - MEASURED AS A LOSS on the same profile:
            // **2.08% -> 2.76%**, with the run's idle time down 14.57% -> 13.36%. So the cost
            // is not the ring probes, and whatever it is, one of the flattening copy or the
            // `VecDeque::reserve` (which can rotate a wrapped ring) costs more than they save.
            //
            // Reverted rather than tuned further: two of the three parts of that rewrite are
            // plausible causes and separating them is another measurement each, for a
            // sub-1% item. If someone picks this up, measure the `reserve` on its own first.
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
        scratch: &mut MixScratch,
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
            // The shared buffers again - see [`MixScratch`]. `src` is read into rather than
            // allocated per run, and `pcm` holds one interleaved group.
            scratch.src.resize((run * ADPCM_BLOCK_BYTES) as usize, 0);
            ctx.read_into(self.data_ptr + self.consumed, &mut scratch.src);
            scratch.pcm.clear();
            scratch.pcm.resize(ADPCM_BLOCK_SAMPLES * ch, 0);
            let decoded = &mut scratch.pcm;
            for g in scratch.src.chunks_exact(ADPCM_BLOCK_BYTES as usize * ch) {
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
            if self.playing {
                note_voice_ended();
            }
            self.playing = false;
        }
    }

    /// Decode until at least `needed` interleaved samples are pending, or the
    /// source is exhausted (stopping the voice unless it loops).
    #[cfg_attr(feature = "profile-symbols", inline(never))]
    fn fill_at9(&mut self, ctx: &GuestCtx, needed: usize, scratch: &mut MixScratch) {
        while self.pending.len() < needed {
            if self.decoder.is_none() || self.superframe_bytes == 0 {
                if self.playing {
                    note_voice_ended();
                }
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
            // Into the shared buffers rather than into two fresh allocations per superframe.
            scratch.src.resize(self.superframe_bytes as usize, 0);
            ctx.read_into(self.data_ptr + self.consumed, &mut scratch.src);
            let sf = &scratch.src;
            let dec = self.decoder.as_mut().unwrap();
            let frames = dec.frames_per_superframe();
            let frame_shorts = dec.frame_samples() * dec.channels();
            scratch.pcm.clear();
            scratch.pcm.resize(frame_shorts, 0);
            let pcm = &mut scratch.pcm;
            let mut inner = 0usize;
            for _ in 0..frames {
                match dec.decode_frame(&sf[inner..], pcm) {
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

/// Grains mixed, voices walked, and voices that were AUDIBLE (a non-zero product of source
/// level, voice gain and every buss level on the way out). Peak and total, so one line says
/// both "how many at once" and "how much work over the run".
static MIX_GRAINS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MIX_VOICES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MIX_AUDIBLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MIX_PEAK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// >>> VOICES THAT WERE MIXED AND CONTRIBUTED NOTHING BUT ZEROES, summed per grain.
///
/// "Audible" above is a claim about the GAIN GRAPH - a non-zero product of levels - and a
/// racing title's race reports every one of its 312 playing voices as audible by that test.
/// It is not the same claim as "this voice produced sound", and only the second one says
/// whether the decode was work worth doing. This counts what the samples actually were.
/// [[vitaslop-count-frames-cannot-tell-silence-from-music]] is the same distinction.
static MIX_SILENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Voices in the bank at the last grain, and how many of them are LOOPING - a voice that
/// loops never ends on its own, so if the count only ever grows, this is where to look.
static MIX_BANK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MIX_LOOPING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// >>> WHERE THE MIX'S LOUDNESS COMES FROM: THE GAIN GRAPH, NOT THE OUTPUT PORT.
///
/// The race sums to ~5x full scale and clips half its grains. The output port's volume was
/// the obvious suspect and has been ruled out by measurement: applying it before the clamp
/// (which is the correct order, and is now what happens) moved the clip count by 0.07%,
/// because its MEAN over the grains is 0.995 - the NGS mix goes out on a port the title
/// never attenuates.
///
/// So the attenuation that is missing is inside the NGS graph. The one candidate the code
/// already warns about is a voice with no ROUTE: `buss_gain` walks `routes` and returns 1.0
/// when there is no entry, so a source we failed to see routed goes straight to the output
/// at its own level, louder than the title asked by the product of every buss it should have
/// passed through. These two counters discriminate that from "the levels themselves are
/// simply large": UNROUTED counts playing voices with no route entry, and the gain sum gives
/// the mean gain actually applied per voice.
static MIX_UNROUTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Applied per-voice gain (`v.gain * v.level * buss_gain`) in permille, summed over every
/// voice of every grain.
static MIX_GAIN_SUM_PERMILLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Buss hops walked per grain whose level was never set by the title, summed over voices.
/// See `At9Voice::level_set`.
static MIX_DEFAULTED_BUSSES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// Voices started by `sceNgsVoicePlay`, stopped by the title (key-off / kill / pause), and
/// ended on their own because a source ran out and did not loop. Starts against the other
/// two is the whole question: if voices only ever start, they accumulate for the run.
static VOICES_STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static VOICES_STOPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static VOICES_ENDED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// >>> LEVELS OUTSIDE 0..=1, AS A COUNT RATHER THAN A LINE EACH.
///
/// `refuse` dedups by the exact TEXT of the refusal, and this refusal used to carry the
/// offending float in it - so every distinct value was a distinct line. On a device that was
/// **231 lines of the on-screen diagnostics panel**, which is the whole panel: the report a
/// person can actually read has room for a few dozen lines, and this pushed every real
/// finding out of it. A diagnostic that buries the findings is worse than no diagnostic
/// ([[vitaslop-a-diagnostic-can-bury-the-findings]]).
///
/// The COUNT plus the range is all of the information the 231 lines carried. Kept as bits
/// because there is no atomic f32; min and max are folded with a compare-exchange loop,
/// which runs only on the failing path.
static LEVEL_OOR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LEVEL_OOR_MIN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static LEVEL_OOR_MAX: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static LEVEL_OOR_NAN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn note_level_out_of_range(level: f32) {
    use std::sync::atomic::Ordering::Relaxed;
    if LEVEL_OOR.fetch_add(1, Relaxed) == 0 {
        LEVEL_OOR_MIN.store(level.to_bits(), Relaxed);
        LEVEL_OOR_MAX.store(level.to_bits(), Relaxed);
    }
    // A NaN fails EVERY comparison, including the range test that sent it here, so it would
    // otherwise never move min or max and would read as no observation at all. It is also
    // the one value that says the field is being read at the wrong OFFSET rather than
    // carrying an unexpected scale, so it is worth its own count.
    if level.is_nan() {
        LEVEL_OOR_NAN.fetch_add(1, Relaxed);
        return;
    }
    let _ = LEVEL_OOR_MIN.fetch_update(Relaxed, Relaxed, |b| {
        let cur = f32::from_bits(b);
        (level < cur).then(|| level.to_bits())
    });
    let _ = LEVEL_OOR_MAX.fetch_update(Relaxed, Relaxed, |b| {
        let cur = f32::from_bits(b);
        (level > cur).then(|| level.to_bits())
    });
}

/// >>> HOW FAR PAST FULL SCALE THE SUMMED VOICES REACH BEFORE THE CLAMP, AND HOW OFTEN.
///
/// This used to live behind a `debug` tracing gate, so nobody could say whether the mix
/// CLIPS without turning on a target that floods the panel - and "does it clip" is the
/// first question a distorted mix asks. It is a `fetch_max` and two adds on a loop that
/// already walks every sample, so it is cheap enough to leave on.
///
/// The peak alone is not enough: a device reported `peak=1.0000`, which is suspicious and is
/// not evidence, because a range says nothing about how much of the distribution is up
/// against it ([[vitaslop-a-range-is-not-a-distribution]]). The clipped-SAMPLE count and the
/// clipped-GRAIN count are what separate "one transient touched full scale" from "the master
/// stage is missing and the whole mix is squared off".
static MIX_PEAK_PERMILLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The output port's volume as this grain was submitted, in permille, and the smallest one
/// ever seen. Reported next to the clip count because the two are one reading: "half the
/// grains clip" means something different at gain 1.000 (nothing is attenuating the mix)
/// than at 0.355 (the title's own attenuation is applied and the mix is STILL too hot), and
/// without this the report cannot distinguish them - which cost one whole measurement.
static MIX_GAIN_PERMILLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1000);
/// The same summed over grains, so the report can state the MEAN as well as the minimum.
/// A minimum alone cannot distinguish "every grain was attenuated" from "one was": both
/// read 0.354, and those two say opposite things about why the mix still clips
/// ([[vitaslop-a-range-is-not-a-distribution]], for the second time on this one line).
static MIX_GAIN_PERMILLE_SUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MIX_CLIPPED_SAMPLES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MIX_CLIPPED_GRAINS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Fold one grain's statistics in. `peak` is the largest absolute sample NGS summed to,
/// before the output port's volume and before the clamp; `clipped` is how many samples were
/// still outside i16 AFTER that volume, i.e. how many the listener actually loses. The two
/// are deliberately taken at different points: the first sizes the mix, the second sizes the
/// damage, and collapsing them into one number hides which of the two changed.
pub fn note_mix_headroom(peak: i32, clipped: u64, gain: f32) {
    use std::sync::atomic::Ordering::Relaxed;
    MIX_PEAK_PERMILLE.fetch_max((peak.unsigned_abs() as u64) * 1000 / 32768, Relaxed);
    let g_permille = (gain.clamp(0.0, 1.0) * 1000.0) as u64;
    MIX_GAIN_PERMILLE.fetch_min(g_permille, Relaxed);
    MIX_GAIN_PERMILLE_SUM.fetch_add(g_permille, Relaxed);
    if clipped > 0 {
        MIX_CLIPPED_SAMPLES.fetch_add(clipped, Relaxed);
        MIX_CLIPPED_GRAINS.fetch_add(1, Relaxed);
    }
}

/// `sceNgsVoiceInit` calls, and how many carried a non-null preset. See `ngs::voice_init`.
static VOICE_INITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static VOICE_INITS_WITH_PRESET: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Playing voices whose OWN level was never set by the title (`At9Voice::level_set`), summed
/// over the grains.
///
/// The sibling of `MIX_DEFAULTED_BUSSES`, one stage lower. A mix that is ~5x too hot is a
/// missing attenuation, and there are exactly two places it can be missing from: a buss on the
/// way out, or the source voice itself. Counting only the first half made "the source voices
/// are all at unity because the title left them there" indistinguishable from "we never parsed
/// the params that carry their level".
static MIX_DEFAULTED_LEVELS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The product of the PATCH volumes on the busses above each playing voice, in permille,
/// summed over voices. 1.000 means every buss-to-buss routing in the graph is at unity and
/// applying them attenuates nothing.
static MIX_BUSS_PATCH_PERMILLE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// The census of NGS param modules this engine does not interpret, keyed by
/// `(params id, module index, the voice is a buss)` and carrying `(writes, first bytes seen)`.
///
/// >>> A LIST OF UNKNOWN MODULE IDS IS NOT ENOUGH TO CHASE ONE.
/// The missing attenuation in this mixer is a buss LEVEL - the counters say every source
/// voice has its own level set and every routing between busses is at unity, so what is left
/// is the levels of the busses themselves, which arrive on a module nothing here reads. Which
/// module that is cannot be answered by the id alone: the same id turns up on source voices
/// too, where it is a synthesiser stage and no concern of the mix. Splitting the census on
/// "does anything route INTO this voice" is what separates the two, and the byte sample makes
/// the candidate readable without a second run.
static UNKNOWN_MODULES: std::sync::Mutex<
    std::collections::BTreeMap<(u32, u32, bool), UnknownModule>,
> = std::sync::Mutex::new(std::collections::BTreeMap::new());

/// How many 32-bit words of an uninterpreted params block are tracked. Every layout REd here
/// so far has its whole meaning inside the first few dozen bytes.
const UNKNOWN_MODULE_WORDS: usize = 12;

/// What was seen of one uninterpreted module.
///
/// The byte sample alone shows ONE instance, and one instance of a struct written ten thousand
/// times says nothing about which of its fields are constants and which are the per-voice
/// controls. The per-word range does: a word that never moves is a layout constant or a
/// default, and a word that sweeps 0..1 across a race is a GAIN - which is the field this
/// mixer is missing and cannot otherwise be picked out without guessing at semantics.
struct UnknownModule {
    writes: u64,
    /// The first block seen, verbatim, so the layout is readable.
    first: Vec<u8>,
    /// Per-word `(min, max)` as raw bits, and separately as f32 where the bits are finite.
    word_bits: Vec<(u32, u32)>,
    word_f32: Vec<(f32, f32)>,
}

/// Record one write of a module this engine does not interpret. `bytes` is the head of the
/// params block.
pub(crate) fn note_unknown_module(id: u32, module: u32, is_buss: bool, bytes: Vec<u8>) {
    let mut g = UNKNOWN_MODULES.lock().unwrap();
    let e = g.entry((id, module, is_buss)).or_insert_with(|| UnknownModule {
        writes: 0,
        first: bytes.clone(),
        word_bits: Vec::new(),
        word_f32: Vec::new(),
    });
    e.writes += 1;
    for i in 0..UNKNOWN_MODULE_WORDS.min(bytes.len() / 4) {
        let w = u32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]);
        let f = f32::from_bits(w);
        if e.word_bits.len() <= i {
            e.word_bits.push((w, w));
            // A non-finite word is not a float; seed the float range from the first FINITE
            // one so one garbage word does not make the whole column unreadable.
            e.word_f32.push(if f.is_finite() { (f, f) } else { (f32::NAN, f32::NAN) });
        } else {
            e.word_bits[i].0 = e.word_bits[i].0.min(w);
            e.word_bits[i].1 = e.word_bits[i].1.max(w);
            if f.is_finite() {
                let slot = &mut e.word_f32[i];
                if slot.0.is_nan() {
                    *slot = (f, f);
                } else {
                    slot.0 = slot.0.min(f);
                    slot.1 = slot.1.max(f);
                }
            }
        }
    }
}

/// Voices played with no source captured from any params path - these are SILENT sounds the
/// title asked for. `refuse` already names each once; this is how many there were.
static VOICES_NO_SOURCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// ...of which turned out to be BUSSES once their routing arrived. See `At9Bank::no_source`.
static VOICES_LATE_BUSS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Record a `sceNgsVoiceInit`, and whether it carried a preset. See `ngs::voice_init`.
pub(crate) fn note_voice_init(with_preset: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    VOICE_INITS.fetch_add(1, Relaxed);
    if with_preset {
        VOICE_INITS_WITH_PRESET.fetch_add(1, Relaxed);
    }
}

pub(crate) fn note_voice_started() {
    VOICES_STARTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn note_voice_stopped() {
    VOICES_STOPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn note_voice_ended() {
    VOICES_ENDED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn note_mix_grain(
    voices: usize,
    audible: usize,
    silent: usize,
    bank: usize,
    looping: usize,
    unrouted: usize,
    gain_sum: f32,
    defaulted_busses: usize,
    defaulted_levels: usize,
    buss_patch_sum: f32,
) {
    use std::sync::atomic::Ordering::Relaxed;
    MIX_UNROUTED.fetch_add(unrouted as u64, Relaxed);
    MIX_DEFAULTED_BUSSES.fetch_add(defaulted_busses as u64, Relaxed);
    MIX_DEFAULTED_LEVELS.fetch_add(defaulted_levels as u64, Relaxed);
    MIX_BUSS_PATCH_PERMILLE.fetch_add((buss_patch_sum * 1000.0) as u64, Relaxed);
    MIX_GAIN_SUM_PERMILLE.fetch_add((gain_sum * 1000.0) as u64, Relaxed);
    MIX_GRAINS.fetch_add(1, Relaxed);
    MIX_VOICES.fetch_add(voices as u64, Relaxed);
    MIX_AUDIBLE.fetch_add(audible as u64, Relaxed);
    MIX_PEAK.fetch_max(voices as u64, Relaxed);
    MIX_SILENT.fetch_add(silent as u64, Relaxed);
    MIX_BANK.store(bank as u64, Relaxed);
    MIX_LOOPING.store(looping as u64, Relaxed);
}

/// One line on what the NGS mix carried. Silent when nothing ever played.
///
/// The number that matters is AUDIBLE against PLAYING: a voice at zero gain is mixed by
/// nothing and decoded in full, because its source has to advance or it would resume from a
/// stale position. If most playing voices are inaudible, that is where the decode time is
/// going and the trade is worth measuring; if they are all audible, the decoder itself is.
pub fn report_mix() {
    for line in mix_report() {
        tracing::info!(target: "vitaslop::perf", "{line}");
    }
}

/// The same counters as TEXT, one string per line, empty when nothing ever mixed.
///
/// >>> THE BROWSER IS THE HOST THAT NEEDS THESE AND IT COULD NOT PRINT THEM.
/// `report_mix` writes to `tracing`, and it was called from exactly one place: the desktop
/// binary's shutdown. So the audio counters existed for the host where audio already works
/// and were unreachable on the one where a user reports crackling - a phone, which has no
/// console, and whose only report is the on-screen diagnostics panel. Returning the lines
/// lets that panel carry them alongside the render split.
pub fn mix_report() -> Vec<String> {
    use std::sync::atomic::Ordering::Relaxed;
    let grains = MIX_GRAINS.load(Relaxed);
    if grains == 0 {
        return Vec::new();
    }
    let voices = MIX_VOICES.load(Relaxed);
    let audible = MIX_AUDIBLE.load(Relaxed);
    let unrouted = MIX_UNROUTED.load(Relaxed);
    let gain_sum = MIX_GAIN_SUM_PERMILLE.load(Relaxed) as f64 / 1000.0;
    let mut out = vec![format!(
        "ngs mix: {grains} grains, {:.1} playing voices each (peak {}), of which {:.1} AUDIBLE          by the gain graph but {:.1} produced ONLY ZEROES; bank {} voices, {} of them looping;          lifetime: {} started, {} stopped by the title, {} ended on their own. \
         Of the playing voices {:.1} a grain have NO ROUTE, so they reach the output at their \
         own level with no buss applied; the mean gain actually applied per voice is {:.3}, \
         and {:.2} buss hops per voice have a level this engine never saw set (each one is a \
         stage of attenuation that may simply be missing). {:.1}% of playing voices never had          their OWN level set either, and the buss PATCH volumes above a voice multiply out to          {:.3} on average - at 1.000 every routing between busses in the graph is at unity, so          there is no attenuation to find there",
        voices as f64 / grains as f64,
        MIX_PEAK.load(Relaxed),
        audible as f64 / grains as f64,
        MIX_SILENT.load(Relaxed) as f64 / grains as f64,
        MIX_BANK.load(Relaxed),
        MIX_LOOPING.load(Relaxed),
        VOICES_STARTED.load(Relaxed),
        VOICES_STOPPED.load(Relaxed),
        VOICES_ENDED.load(Relaxed),
        unrouted as f64 / grains as f64,
        if voices > 0 { gain_sum / voices as f64 } else { 0.0 },
        if voices > 0 {
            MIX_DEFAULTED_BUSSES.load(Relaxed) as f64 / voices as f64
        } else {
            0.0
        },
        if voices > 0 {
            100.0 * MIX_DEFAULTED_LEVELS.load(Relaxed) as f64 / voices as f64
        } else {
            0.0
        },
        if voices > 0 {
            MIX_BUSS_PATCH_PERMILLE.load(Relaxed) as f64 / 1000.0 / voices as f64
        } else {
            0.0
        },
    )];
    // >>> AND WHETHER THAT MIX CLIPPED, which is the difference between "quiet" and
    // "distorted" and was previously unanswerable outside a debug build. See
    // `note_mix_headroom`. Reported even when nothing clipped: "0 samples clipped" is the
    // result that RULES OUT the master stage, and an instrument that only speaks up when it
    // has bad news cannot be used to clear anything.
    let peak_permille = MIX_PEAK_PERMILLE.load(Relaxed);
    let clipped = MIX_CLIPPED_SAMPLES.load(Relaxed);
    let clipped_grains = MIX_CLIPPED_GRAINS.load(Relaxed);
    out.push(format!(
        "ngs headroom: the NGS mix peaks at {:.3}x full scale BEFORE the output port's          volume, which averaged {:.3} over the grains and was {:.3} at its lowest; after it, {clipped} samples still clip, over          {clipped_grains} of {grains} grains ({:.2}% of grains). The mix figure is what NGS          summed to; the clip figure is what the speaker would have got. A gain of 1.000 means          the title never attenuated the port, so nothing was applied and the clip figure is          the raw mix; anything still clipping BELOW that is a master stage this engine does          not apply",
        peak_permille as f64 / 1000.0,
        MIX_GAIN_PERMILLE_SUM.load(Relaxed) as f64 / 1000.0 / grains as f64,
        MIX_GAIN_PERMILLE.load(Relaxed) as f64 / 1000.0,
        100.0 * clipped_grains as f64 / grains as f64,
    ));
    // >>> AND THE MODULES THIS ENGINE DOES NOT INTERPRET, split by whether they arrived on a
    // BUSS. See `UNKNOWN_MODULES`: a stage on a buss is the only place the missing attenuation
    // can still be, so this is the shortlist, not a curiosity.
    {
        let g = UNKNOWN_MODULES.lock().unwrap();
        if !g.is_empty() {
            let mut rows: Vec<String> = Vec::new();
            for ((id, module, is_buss), m) in g.iter() {
                // Only the words that MOVE, and how far. A constant column is the struct's
                // shape; a column with a range is a control the title is actually using, and
                // one whose range sits inside 0..1 is the shape of a gain.
                let varying: Vec<String> = m
                    .word_bits
                    .iter()
                    .enumerate()
                    .filter(|(_, (lo, hi))| lo != hi)
                    .map(|(i, (lo, hi))| {
                        let (flo, fhi) = m.word_f32[i];
                        format!("+{:#04x} u32 {lo:#x}..{hi:#x} f32 {flo}..{fhi}", i * 4)
                    })
                    .collect();
                rows.push(format!(
                    "{id:#010x} as module {module} on a {} x{} first=[{}] varying words: {}",
                    if *is_buss { "BUSS" } else { "source voice" },
                    m.writes,
                    m.first.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" "),
                    if varying.is_empty() {
                        "NONE - every write of this module is byte-identical, so nothing in it                          is a per-voice control and ignoring it cannot be what varies"
                            .to_string()
                    } else {
                        varying.join(", ")
                    },
                ));
            }
            out.push(format!(
                "ngs modules: {} distinct (params id, module index, buss?) combinations were                  written and NOT interpreted - every one is a synthesiser or mix stage the                  device runs and this engine does not. The ones marked BUSS are the ones a                  whole mix passes through: {}",
                g.len(),
                rows.join("; "),
            ));
        }
    }
    // >>> AND THE SOUNDS THAT NEVER PLAYED AT ALL, with the one candidate for why.
    let no_source = VOICES_NO_SOURCE.load(Relaxed);
    if no_source > 0 {
        out.push(format!(
            "ngs silent voices: {no_source} plays on voices whose source was never captured              from any params path, of which {} were BUSSES whose routing arrived after the              play (harmless - a buss has no source of its own) and {} are still unexplained              SILENT SOUNDS. `sceNgsVoiceInit` was called {} times, {} of them with a preset              (a preset is the other way a source could arrive, and this engine ignores it -              at zero, that candidate is REFUTED for this title)",
            VOICES_LATE_BUSS.load(Relaxed),
            no_source.saturating_sub(VOICES_LATE_BUSS.load(Relaxed)),
            VOICE_INITS.load(Relaxed),
            VOICE_INITS_WITH_PRESET.load(Relaxed),
        ));
    }
    // >>> AND THE LEVELS THAT WERE NOT LEVELS, as one line. See `note_level_out_of_range`.
    let oor = LEVEL_OOR.load(Relaxed);
    if oor > 0 {
        let nan = LEVEL_OOR_NAN.load(Relaxed);
        out.push(format!(
            "ngs levels: {oor} reads outside 0..=1, spanning {} to {} ({nan} of them NaN) -              unity was used for each. One line, not one per distinct value: this used to be              231 lines of a device's diagnostics panel. A NaN here means the level field is              being read at the wrong OFFSET, not that the title asked for an odd scale",
            f32::from_bits(LEVEL_OOR_MIN.load(Relaxed)),
            f32::from_bits(LEVEL_OOR_MAX.load(Relaxed)),
        ));
    }
    out
}

/// >>> THE BUFFERS EVERY VOICE DECODE NEEDS AND NONE OF THEM SHOULD ALLOCATE.
///
/// A superframe decode used to allocate two `Vec`s - the guest bytes it read and the PCM it
/// decoded into - and it runs once per voice per grain for as long as the voice plays. On a
/// racing title that is 96 voices a grain, sixty grains a second, so ~11,500 allocations a
/// second on the path a browser profile already blamed for 32% of its thread. The buffers
/// belong to the BANK rather than to each voice: 807 voices holding a superframe each is
/// megabytes of idle scratch, and only one voice decodes at a time.
#[derive(Default)]
pub(crate) struct MixScratch {
    /// The compressed bytes read out of guest memory for one superframe or ADPCM run.
    src: Vec<u8>,
    /// One decoded frame, interleaved.
    pcm: Vec<i16>,
    /// One voice's grain as a single contiguous slice, for the case where the ring's own
    /// two halves split it. See the mixing loop in [`At9Bank::mix_grain`].
    flat: Vec<i16>,
}

/// The bank of source voices, keyed by the NGS voice handle, plus the routing graph
/// that says where each one's output goes.
#[derive(Default)]
pub(crate) struct At9Bank {
    /// Reused decode buffers - see [`MixScratch`]. Taken out for the mixing loop and put
    /// back, which is what lets one buffer be shared by voices held in the same map.
    scratch: MixScratch,
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
    /// Voices played with no source, still unexplained - see `VOICES_NO_SOURCE`.
    ///
    /// Held rather than only counted because the explanation can arrive LATER: `play` decides
    /// a voice is a buss by asking whether anything routes INTO it, and a title that creates
    /// its routing after starting the voice makes a perfectly ordinary buss look like a
    /// source that was never configured. `set_route` moves any voice it names out of here.
    no_source: std::collections::BTreeSet<u32>,
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
            Some(v) if v.data_ptr != 0 => {
                // Counted HERE and not in `start`, which a loop calls again every lap: the
                // question this answers is how many voices the TITLE started.
                note_voice_started();
                v.start()
            }
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
                    VOICES_NO_SOURCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    self.no_source.insert(voice);
                    // >>> QUIET, BECAUSE THIS TEST CANNOT BE MADE AT PLAY TIME.
                    //
                    // `is_buss` above asks whether anything routes INTO this voice, and a
                    // title is entitled to create that routing AFTER starting it - at which
                    // point an ordinary buss has already been reported as a sound that will
                    // never be heard. MEASURED on one racer's race: **10 of the 11** voices
                    // this branch fired for turned out to be busses within the same run, and
                    // a device panel carried 77 such lines. A warning that is wrong ten times
                    // out of eleven trains its reader to skip the section it lives in.
                    //
                    // The COUNT is the honest instrument and it is on the panel - `mix_report`
                    // subtracts the ones `set_route` later explained and reports only the
                    // residue as silent sounds (1 in that race). This still names each voice
                    // at DEBUG for whoever is chasing that residue.
                    v.refuse_quiet(format_args!(
                        "sceNgsVoicePlay on voice {voice:#x}, which never locked/unlocked player \
                         params at all - nothing to decode (may yet turn out to be a buss: its \
                         routing can arrive after this call)"
                    ));
                }
            }
        }
    }

    /// Whether anything routes INTO `voice` - i.e. whether it is a BUSS rather than a source.
    ///
    /// The distinction is what makes an uninterpreted module worth chasing or not: a stage on
    /// a source voice affects one sound, a stage on a buss affects everything routed through
    /// it, and the missing attenuation in this mixer is at a buss by elimination.
    pub(crate) fn is_buss(&self, voice: u32) -> bool {
        self.routes.values().any(|dst| *dst == voice)
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
            // A voice we already complained had no source, which something now routes INTO,
            // is a BUSS whose routing simply arrived after its play. See `At9Bank::no_source`.
            if self.no_source.remove(&destination) {
                VOICES_LATE_BUSS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// The product of every buss level `voice` passes through on its way to the output,
    /// NOT including the voice's own level.
    ///
    /// Bounded walk: a routing graph that contained a cycle would otherwise hang the
    /// audio thread, and a malformed graph is not worth a deadlock. Eight hops is far
    /// past any real mix tree (the deepest measured is three).
    /// How many busses on `voice`'s way to the output never had a level set - see
    /// `At9Voice::level_set`. Same bounded walk as [`At9Bank::buss_gain`].
    fn defaulted_busses_above(&self, voice: u32) -> usize {
        let mut n = 0;
        let mut at = voice;
        for _ in 0..8 {
            let Some(&next) = self.routes.get(&at) else { break };
            let Some(v) = self.voices.get(&next) else { break };
            if !v.level_set {
                n += 1;
            }
            at = next;
        }
        n
    }

    /// Only the PATCH volumes on the busses above `voice` - the half of [`Self::buss_gain`]
    /// that used to be dropped, on its own, so a report can say whether applying it matters.
    fn buss_patch_gain(&self, voice: u32) -> f32 {
        let mut gain = 1.0f32;
        let mut at = voice;
        for _ in 0..8 {
            let Some(&next) = self.routes.get(&at) else { break };
            let Some(v) = self.voices.get(&next) else { break };
            gain *= v.gain;
            at = next;
        }
        gain
    }

    fn buss_gain(&self, voice: u32) -> f32 {
        let mut gain = 1.0f32;
        let mut at = voice;
        for _ in 0..8 {
            let Some(&next) = self.routes.get(&at) else { break };
            let Some(v) = self.voices.get(&next) else { break };
            // >>> BOTH OF A BUSS'S CONTROLS, not just one.
            //
            // A buss is a voice like any other and has the same two independent gains: its
            // own `level`, from its params module, and the `gain` of the PATCH that carries
            // its output onward, from `sceNgsVoicePatchSetVolume`. Only `level` was applied
            // here, so every routing volume set on a buss-to-buss or buss-to-master patch was
            // recorded and then ignored - and on a title whose graph is nine mixer busses
            // feeding a master, that is where most of the balancing lives. The source voice's
            // own pair is applied by the caller; this walk owes the same treatment to every
            // hop above it.
            gain *= v.gain * v.level;
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
    #[cfg_attr(feature = "profile-symbols", inline(never))]
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
        // Playing voices with nothing to route them anywhere - see `MIX_UNROUTED`.
        let unrouted = routed.iter().filter(|(h, _)| !self.routes.contains_key(h)).count();
        // ...and the busses on their way out whose level is the DEFAULT rather than something
        // the title set. See `At9Voice::level_set`: each of those is a stage of attenuation
        // this mixer may simply not be applying, and the mix is ~5x too hot.
        let defaulted_busses = routed
            .iter()
            .map(|(h, _)| self.defaulted_busses_above(*h))
            .sum::<usize>();
        // ...and the same question one stage lower: playing voices whose own level is still
        // the 1.0 this struct starts at. See `MIX_DEFAULTED_LEVELS`.
        let defaulted_levels = routed
            .iter()
            .filter(|(h, _)| self.voices.get(h).is_some_and(|v| !v.level_set))
            .count();
        // How much the buss PATCH volumes above each voice contribute - the half of
        // `buss_gain` that was being dropped. 1.0 a voice means applying them changes nothing,
        // which is a real answer and the one that sends the search elsewhere.
        let buss_patch_sum: f32 = routed.iter().map(|(h, _)| self.buss_patch_gain(*h)).sum();
        // >>> WHAT THE MIX IS ACTUALLY CARRYING, because the cost of this path is per VOICE
        // and nothing reported how many there were. MEASURED with a V8 worker profile of one
        // title's browser race: ATRAC9 decode was **32% of the whole thread** and
        // `sceAudioOutOutput` another 14%, on a run that was frame-limited at 32 fps. Whether
        // that is "many voices" or "a slow decoder" is the first question anyone asks next,
        // and it is one counter.
        let routed_len = routed.len();
        let bank = self.voices.len();
        let looping = self.voices.values().filter(|v| v.playing && v.loops()).count();
        let audible = routed.iter().filter(|(_, g)| *g > 0.0).count();
        // Counted as the grain is mixed, below.
        let mut silent = 0usize;
        // The gain each voice is actually mixed at, summed - see `MIX_GAIN_SUM_PERMILLE`.
        let mut gain_sum = 0.0f32;
        // Taken out for the loop and put back at the end: the buffers are the bank's, and
        // the voices being decoded are entries of the bank's own map.
        let mut scratch = std::mem::take(&mut self.scratch);
        // The whole grain, so a device run's phase table says what audio costs without
        // anyone having to attach a profiler to it - see [`crate::perf::Phase::AudioMix`].
        let _t = crate::perf::scope(crate::perf::Phase::AudioMix);
        for (handle, buss_gain) in routed {
            let Some(v) = self.voices.get_mut(&handle) else { continue };
            let vc = v.channels.max(1) as usize;
            {
                let _t = crate::perf::scope(crate::perf::Phase::AudioDecode);
                v.fill(ctx, grain * vc, port_rate, &mut scratch);
            }
            // A voice the title has turned all the way down contributes nothing, so it
            // costs nothing: its samples are still CONSUMED below (the source has to
            // advance, or it would resume from a stale position the moment it is turned
            // back up), but the mixing loop is skipped entirely.
            let gain = v.gain * v.level * buss_gain;
            gain_sum += gain;
            if gain <= 0.0 {
                silent += 1;
                let consumed = (grain * vc).min(v.pending.len());
                v.pending.drain(..consumed);
                continue;
            }
            // Fixed-point gain: the mixer is integer, and a float multiply per sample
            // per voice on the grain-rate path is exactly the kind of cost this project
            // measures. 16.16 keeps unity exact and a -60 dB setting still meaningful.
            let gain_q16 = (gain.min(4.0) * 65536.0) as i32;
            // >>> THE GRAIN IS ONE FLAT SLICE, AND A SILENT VOICE IS NOT MIXED AT ALL.
            //
            // `pending` is a ring, and this is the innermost loop of the whole audio path -
            // one read per sample per port channel per PLAYING VOICE. A racing title mixes 21
            // voices a grain (peak 88) at a 1024-sample grain in stereo, sixty times a second,
            // and a V8 worker profile of a browser race puts 4.1% of the entire thread in
            // `out_output`'s body, which is where this loop is inlined.
            //
            // It used to read through a closure that probed the ring's two halves - `head.get`
            // then `tail.get` - so every single sample paid two bounds checks and a branch on
            // the wrap. Resolving the wrap ONCE, into a slice, moves all of that out of the
            // inner loop: the common case is that the grain does not straddle the wrap at all
            // and the ring's own first half IS the slice, and the case that does straddle
            // costs one copy of `want` samples rather than `want` branches.
            //
            // The arithmetic is untouched - same 16.16 gain, same accumulation order, same
            // saturation - so the mix is the same mix.
            let want = grain * vc;
            let src: &[i16] = {
                let (head, tail) = v.pending.as_slices();
                if head.len() >= want {
                    &head[..want]
                } else {
                    let need = want - head.len();
                    scratch.flat.clear();
                    scratch.flat.extend_from_slice(head);
                    scratch.flat.extend_from_slice(&tail[..need.min(tail.len())]);
                    &scratch.flat[..]
                }
            };
            // >>> A VOICE THAT IS ALL ZEROES IS SKIPPED, NOT ADDED.
            //
            // This pass already existed to COUNT such voices - 3.5 of the 21.1 playing in a
            // race grain produce nothing but zeroes - and then the mixer added every one of
            // those zeros anyway. Adding zero is a no-op, so skipping is bit-identical and it
            // is a sixth of the mixing loop. The scan itself short-circuits on the first
            // nonzero sample, so a voice that IS audible pays almost nothing for it.
            if !src.iter().any(|&s| s != 0) {
                silent += 1;
            } else {
                match (vc, port_channels) {
                    // The two layouts that actually occur, written straight: a mono source
                    // fanned out to every port channel, and a source whose channel count
                    // matches the port. A voice that drained mid-grain simply has a shorter
                    // `src`, so it contributes what it has and the drain below still runs -
                    // no early `return`, which sat here for one build and dropped the rest of
                    // the mix.
                    (1, pc) => {
                        for (f, &s) in src.iter().enumerate() {
                            let scaled = (s as i32 * gain_q16) >> 16;
                            let Some(frame) = mix.get_mut(f * pc..f * pc + pc) else { break };
                            for m in frame {
                                *m += scaled;
                            }
                        }
                    }
                    // Source channels match the port's, so the two buffers are the same shape
                    // and the mix is an elementwise add - no index arithmetic at all.
                    (vc, pc) if vc == pc => {
                        for (m, &s) in mix.iter_mut().zip(src.iter()) {
                            *m += (s as i32 * gain_q16) >> 16;
                        }
                    }
                    // Anything else keeps the general rule: the last source channel is
                    // repeated.
                    _ => {
                        for f in 0..grain {
                            for c in 0..port_channels {
                                let src_c = c.min(vc - 1);
                                let Some(&s) = src.get(f * vc + src_c) else { continue };
                                mix[f * port_channels + c] += (s as i32 * gain_q16) >> 16;
                            }
                        }
                    }
                }
            }
            // Drop the grain we just consumed.
            let consumed = (grain * vc).min(v.pending.len());
            v.pending.drain(..consumed);
        }
        note_mix_grain(
            routed_len,
            audible,
            silent,
            bank,
            looping,
            unrouted,
            gain_sum,
            defaulted_busses,
            defaulted_levels,
            buss_patch_sum,
        );
        self.scratch = scratch;
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

    /// A voice with samples ready to mix and nothing behind it. `SourceKind::None` means
    /// `fill` has nothing to decode, so what is in `pending` is exactly what gets mixed -
    /// which is what makes the mixing arithmetic testable on its own.
    fn ready(samples: &[i16], channels: u32, gain: f32) -> At9Voice {
        let mut v = At9Voice::empty();
        v.playing = true;
        v.channels = channels;
        v.gain = gain;
        v.level = 1.0;
        v.pending = samples.iter().copied().collect();
        v
    }

    /// >>> THE MIX ITSELF, SAMPLE FOR SAMPLE.
    ///
    /// The inner loop was rewritten to read the pending ring as its two slices rather than
    /// probing it once per sample per channel per voice (~100,000 probes a grain on a
    /// racing title's 96 voices). A rewrite of the hottest loop in the audio path with no
    /// test on its OUTPUT is a rewrite nobody can check, so this is that test: a mono voice
    /// fanned out, a stereo voice matched to the port, a half-gain voice, and a voice that
    /// runs out mid-grain.
    #[test]
    fn the_mix_is_sample_exact() {
        let mut mem = vec![0u8; 64];
        let mut regs = [0u32; vitaslop_transpiler::abi::REG_COUNT];
        let mut vfp = [0u32; crate::host::VFP_ARG_COUNT];
        let mut slice = crate::host::SliceMemory(&mut mem);
        let ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut slice, 0);

        let mut bank = At9Bank::default();
        // Mono at unity: every port channel gets it.
        bank.voices.insert(1, ready(&[100, 200, 300, 400], 1, 1.0));
        // Stereo at unity: channel for channel.
        bank.voices.insert(2, ready(&[10, -10, 20, -20, 30, -30, 40, -40], 2, 1.0));
        // Mono at half gain, and it runs out after two frames.
        bank.voices.insert(3, ready(&[1000, 2000], 1, 0.5));

        let mut mix = vec![0i32; 4 * 2];
        bank.mix_grain(&ctx, &mut mix, 4, 2, 48_000);

        // Frame 0: 100 (mono) + 10/-10 (stereo) + 500 (half of 1000).
        assert_eq!(mix, vec![
            100 + 10 + 500, 100 - 10 + 500,
            200 + 20 + 1000, 200 - 20 + 1000,
            300 + 30, 300 - 30,
            400 + 40, 400 - 40,
        ]);
        // Every voice consumed its own grain, whatever it contributed.
        assert_eq!(bank.voices[&1].pending.len(), 0);
        assert_eq!(bank.voices[&2].pending.len(), 0);
        assert_eq!(bank.voices[&3].pending.len(), 0, "a drained voice still drains");
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
