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

/// One AT9 source voice: its guest bitstream, decode state, and pending PCM.
pub(crate) struct At9Voice {
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
    /// Decoded interleaved samples not yet mixed out.
    pending: VecDeque<i16>,
}

impl At9Voice {
    fn empty() -> At9Voice {
        At9Voice {
            data_ptr: 0,
            data_bytes: 0,
            config: [0; 4],
            channels: 0,
            loop_count: 0,
            playing: false,
            decoder: None,
            superframe_bytes: 0,
            consumed: 0,
            pending: VecDeque::new(),
        }
    }

    /// Read the AT9 player params a title just wrote and store the source. Returns
    /// false if the buffer is not a recognizable AT9 player params block.
    fn load_params(&mut self, ctx: &GuestCtx, params_addr: u32) -> bool {
        if ctx.read_u32(params_addr) != PARAMS_ID {
            return false;
        }
        let data_ptr = ctx.read_u32(params_addr + OFF_BUFFER_PTR);
        let data_bytes = ctx.read_u32(params_addr + OFF_BUFFER_BYTES);
        if data_ptr == 0 || data_bytes == 0 {
            return false;
        }
        let cfg = ctx.read_bytes(params_addr + OFF_CONFIG, 4);
        let config = [cfg[0], cfg[1], cfg[2], cfg[3]];
        // header byte must be 0xFE for a valid config; otherwise this isn't AT9.
        if config[0] != 0xFE {
            return false;
        }
        self.data_ptr = data_ptr;
        self.data_bytes = data_bytes;
        self.config = config;
        self.channels = ctx.read_u32(params_addr + OFF_CHANNELS) & 0xffff;
        self.loop_count = ctx.read_u32(params_addr + OFF_LOOP_COUNT) as i16;
        true
    }

    /// (Re)start playback from the beginning of the current source.
    fn start(&mut self) {
        match Atrac9Decoder::new(self.config) {
            Ok(dec) => {
                self.superframe_bytes = dec.superframe_bytes() as u32;
                self.decoder = Some(dec);
                self.consumed = 0;
                self.pending.clear();
                self.playing = true;
            }
            Err(_) => {
                self.playing = false;
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

    /// Decode until at least `needed` interleaved samples are pending, or the
    /// source is exhausted (stopping the voice unless it loops).
    fn fill(&mut self, ctx: &GuestCtx, needed: usize) {
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
                    Err(_) => {
                        self.playing = false;
                        return;
                    }
                }
            }
            self.consumed += self.superframe_bytes;
        }
    }
}

/// The bank of AT9 voices, keyed by the NGS voice handle.
#[derive(Default)]
pub(crate) struct At9Bank {
    voices: std::collections::BTreeMap<u32, At9Voice>,
}

impl At9Bank {
    /// Handle `sceNgsVoiceUnlockParams` for the player module: capture the AT9 source
    /// the title just configured on `voice`.
    pub(crate) fn set_player_params(&mut self, ctx: &GuestCtx, voice: u32, params_addr: u32) {
        let v = self.voices.entry(voice).or_insert_with(At9Voice::empty);
        v.load_params(ctx, params_addr);
    }

    /// Start `voice` (from `sceNgsVoicePlay`), if it has an AT9 source.
    pub(crate) fn play(&mut self, voice: u32) {
        if let Some(v) = self.voices.get_mut(&voice) {
            if v.data_ptr != 0 {
                v.start();
            }
        }
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
    ) {
        for v in self.voices.values_mut() {
            if !v.playing {
                continue;
            }
            let vc = v.channels.max(1) as usize;
            v.fill(ctx, grain * vc);
            for f in 0..grain {
                // Pull this frame's voice samples (or silence if the voice drained).
                for c in 0..port_channels {
                    let src_c = if vc == 1 { 0 } else { c.min(vc - 1) };
                    // pending is a flat interleaved queue; index into the frame.
                    let idx = f * vc + src_c;
                    if let Some(&s) = v.pending.get(idx) {
                        mix[f * port_channels + c] += s as i32;
                    }
                }
            }
            // Drop the grain we just consumed.
            let consumed = (grain * vc).min(v.pending.len());
            v.pending.drain(..consumed);
        }
    }
}
