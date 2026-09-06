//! Platform-native H.264 decoding for Windows, macOS, Linux and the browser.
//!
//! This crate does NOT decode H.264. It drives the decoder the platform already has -
//! Media Foundation on Windows, Video Toolbox on macOS, VA-API on Linux, WebCodecs in the
//! browser - behind one API, and does the work those four APIs all leave to the caller:
//! parsing parameter sets, cutting the stream into access units, deriving picture order
//! counts, and putting frames back into presentation order.
//!
//! ```no_run
//! use vitaslop_h264::{Decoder, DecoderConfig, Packet};
//!
//! # fn main() -> Result<(), vitaslop_h264::Error> {
//! # let annex_b_stream: &[u8] = &[];
//! let mut decoder = Decoder::new(DecoderConfig::default())?;
//! decoder.send(Packet::new(annex_b_stream))?;
//! while let Some(frame) = decoder.receive()? {
//!     println!("{}x{} at {}", frame.width, frame.height, frame.pts);
//!     decoder.recycle(frame);
//! }
//! decoder.finish()?; // end of stream: emit the frames still held back
//! # Ok(())
//! # }
//! ```
//!
//! # What it will not do
//!
//! An input this crate does not fully understand is an error, never an approximated
//! picture: 4:2:2 and 4:4:4 chroma, 10-bit, MVC/SVC, and interlaced coding on the VA-API
//! path all come back as [`Error::Unsupported`]. A machine with no decoder comes back as
//! [`Error::NoDecoder`], which a caller is expected to fall back on rather than treat as
//! fatal.
//!
//! # Threading
//!
//! A `Decoder` is owned by one thread. The platform decoders behind it are all internally
//! threaded, so a caller does not gain anything by adding another layer; what a caller
//! DOES gain is a decoder per stream, which works.

#![deny(missing_docs)]
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod backend;
pub mod bitstream;
#[cfg(feature = "synth")]
pub mod conformance;
mod error;
mod frame;
#[cfg(feature = "mp4")]
pub mod mp4;
#[cfg(feature = "synth")]
pub mod synth;

use std::collections::HashMap;

pub use error::{Error, Result};
pub use frame::{ColorInfo, ColorMatrix, Frame, PixelFormat, Plane};

pub use backend::Acceleration;

use backend::{Backend, FramePool, OutputOrder, Reorderer, StreamConfig};

/// How many presentation keys to remember timestamps for. Far more than any decoder holds
/// in flight (a DPB tops out at 16 frames), and small enough that a dropped picture cannot
/// leak memory across a long stream.
const PTS_WINDOW: usize = 256;
use bitstream::avcc::AvcC;
use bitstream::{AccessUnit, AuSplitter};

/// How the caller's packets are framed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum InputFormat {
    /// Decide per packet: a leading start code means Annex B, anything else is treated as
    /// length-prefixed with 4-byte lengths. Right for almost every caller, and exactly
    /// right when [`DecoderConfig::extradata`] is an avcC record (its length size wins).
    #[default]
    Auto,
    /// Start-code delimited (`.h264` files, RTP depacketisers, most hardware capture).
    AnnexB,
    /// ISO-BMFF style: each NAL prefixed by its length in `length_size` bytes.
    LengthPrefixed {
        /// 1, 2 or 4.
        length_size: usize,
    },
}

/// Decoder settings.
#[derive(Debug, Clone, Default)]
pub struct DecoderConfig {
    /// Framing of the packets that will be sent.
    pub input: InputFormat,
    /// Out-of-band parameter sets: an avcC record (from an MP4's `avc1` sample entry) or
    /// raw Annex B parameter set NALs. Optional - a stream that carries its own in-band is
    /// the common case.
    pub extradata: Option<Vec<u8>>,
    /// Cap on recycled pixel buffers held between frames. The default (8) covers a decoder
    /// in flight; raise it only if the caller holds many frames at once.
    pub max_pooled_buffers: Option<usize>,
    /// Whether each packet is exactly one access unit (one coded picture).
    ///
    /// It matters for LATENCY, not correctness. A coded picture is only known to be
    /// complete when the next one starts, so a decoder fed a stream cannot submit a picture
    /// until the packet after it arrives - one frame of added delay. A caller who knows its
    /// packets are whole access units (an MP4 or MOV sample, a WebCodecs chunk, an RTP
    /// depacketiser's output) sets this and gets the frame immediately.
    ///
    /// `None` infers it: length-prefixed input is ISO-BMFF-shaped, where a sample IS an
    /// access unit, so it is assumed to be one; Annex B is assumed to be a stream.
    pub packets_are_access_units: Option<bool>,
    /// Whether to use video hardware. Default (`None`) is yes.
    ///
    /// Turning it off is a real, if narrow, need. On Windows the hardware path decodes
    /// through a driver-sized bitstream buffer and a picture larger than that buffer comes
    /// back PARTLY DECODED with nothing reported - measured here as exact up to about
    /// 450 KB of coded picture and truncated above it. No encoder produces pictures that
    /// large (they run 10-100 KB), so this is about pathological input, not ordinary video.
    /// Check [`Decoder::acceleration`] to see what was actually obtained.
    pub hardware: Option<bool>,
    /// Largest coded picture, in bytes, that may be given to a hardware decoder.
    ///
    /// Windows only, and it exists because that limit is real and silent: a picture bigger
    /// than the driver's DXVA bitstream buffer decodes partially and reports nothing (one
    /// machine here: exact through 588 KB, wrong from 633 KB). A picture over this size is
    /// refused with [`Error::Unsupported`] instead - a caller that hits it should recreate
    /// the decoder with `hardware: Some(false)`, which has no such limit.
    ///
    /// Default 512 KiB, which is below every DXVA buffer size seen and far above any
    /// picture a real encoder produces (10-100 KB).
    pub max_hardware_picture_bytes: Option<usize>,
    /// Ask the decoder to emit each picture as soon as it is decoded rather than holding
    /// a pipeline of them.
    ///
    /// It is not a speed setting; it changes WHEN output appears. A decoder left to its
    /// own devices fills a several-frame pipeline before the first picture comes out, and
    /// a caller that must have a picture back for the access unit it just submitted - one
    /// standing in for a fixed-function block that behaves that way - never gets started.
    ///
    /// It is only safe on a stream that needs no reordering. A stream that does (B-frames,
    /// `max_num_reorder_frames > 0`) still comes out in presentation order, but the
    /// decoder has to hold pictures back to do that, and asking it not to is asking for
    /// the wrong order. Default off.
    pub low_latency: bool,
}

/// One access unit's worth of input, or any fragment of a stream.
#[derive(Debug, Clone, Copy)]
pub struct Packet<'a> {
    /// The bytes, in the configured framing.
    pub data: &'a [u8],
    /// Presentation timestamp in the caller's own units. When `None`, decoded frames carry
    /// a presentation-order index derived from the stream's picture order counts instead,
    /// which is still a correct ordering key - a raw H.264 stream has no timestamps of its
    /// own to carry.
    pub pts: Option<i64>,
}

impl<'a> Packet<'a> {
    /// A packet with no timestamp.
    pub fn new(data: &'a [u8]) -> Packet<'a> {
        Packet { data, pts: None }
    }

    /// A packet with a presentation timestamp.
    pub fn with_pts(data: &'a [u8], pts: i64) -> Packet<'a> {
        Packet { data, pts: Some(pts) }
    }
}

/// What the stream turned out to be, once its first parameter set has been read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamInfo {
    /// Visible width after cropping.
    pub width: u32,
    /// Visible height after cropping.
    pub height: u32,
    /// `profile_idc` (66 baseline, 77 main, 100 high, ...).
    pub profile_idc: u8,
    /// `level_idc` times ten (30 = level 3.0).
    pub level_idc: u8,
    /// Colour signalling from the VUI.
    pub color: ColorInfo,
    /// Sample aspect ratio, when the stream signals one.
    pub sample_aspect_ratio: Option<(u32, u32)>,
    /// `(num_units_in_tick, time_scale)` when the stream signals timing. Frame rate is
    /// `time_scale / (2 * num_units_in_tick)`.
    pub timing: Option<(u32, u32)>,
    /// How many frames may have to be held back to restore presentation order.
    pub max_reorder_frames: u32,
}

/// An H.264 decoder.
///
/// Feed it with [`Decoder::send`], take frames with [`Decoder::receive`], and call
/// [`Decoder::finish`] at the end of the stream. Frames come out in PRESENTATION order on
/// every platform.
pub struct Decoder {
    backend: Box<dyn Backend>,
    splitter: AuSplitter,
    config: DecoderConfig,
    pool: FramePool,
    reorderer: Reorderer,
    ready: Vec<Frame>,
    /// Access units are handed to the backend keyed on a monotonic presentation index; this
    /// maps that key back to the caller's own timestamp.
    pts_by_key: HashMap<i64, i64>,
    /// Base added to a picture order count to keep keys monotonic across IDRs, which reset
    /// the count.
    key_base: i64,
    /// Highest key handed out, so the next sequence can start above it.
    max_key: i64,
    info: Option<StreamInfo>,
    configured: bool,
    scratch: Vec<u8>,
    au_scratch: Vec<AccessUnit>,
}

impl Decoder {
    /// Create a decoder on this platform's video decoder.
    ///
    /// Fails with [`Error::NoDecoder`] when the machine has none - an "N" edition of
    /// Windows without the Media Feature Pack, a Linux box with no libva or no VA-API
    /// driver, a browser without WebCodecs. That case is a normal outcome, not a bug.
    pub fn new(config: DecoderConfig) -> Result<Decoder> {
        let backend = backend::create(backend::BackendOptions {
            hardware: config.hardware.unwrap_or(true),
            low_latency: config.low_latency,
            max_hardware_picture_bytes: config
                .max_hardware_picture_bytes
                .unwrap_or(512 * 1024),
        })?;
        let mut decoder = Decoder {
            backend,
            splitter: AuSplitter::new(),
            pool: FramePool::new(config.max_pooled_buffers.unwrap_or(8)),
            config,
            reorderer: Reorderer::default(),
            ready: Vec::new(),
            pts_by_key: HashMap::new(),
            key_base: 0,
            max_key: -1,
            info: None,
            configured: false,
            scratch: Vec::new(),
            au_scratch: Vec::new(),
        };
        if let Some(extra) = decoder.config.extradata.clone() {
            decoder.load_extradata(&extra)?;
        }
        Ok(decoder)
    }

    /// Which platform decoder is running, for diagnostics: `"MediaFoundation"`,
    /// `"VideoToolbox"`, `"VA-API"`, `"WebCodecs"`.
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    /// What the stream is, once a parameter set has been seen.
    pub fn stream_info(&self) -> Option<StreamInfo> {
        self.info
    }

    /// Whether decoding is reaching the machine's video hardware.
    ///
    /// Worth checking once after the first frame: correct-but-software decoding looks
    /// exactly like correct-but-hardware decoding until something has to keep up with a
    /// frame rate. Backends that cannot ask the question report
    /// [`Acceleration::Unknown`] rather than guessing.
    pub fn acceleration(&self) -> Acceleration {
        self.backend.acceleration()
    }

    /// What the backend learned from actually decoding - see [`backend::Backend::detail`].
    /// `None` until it has decoded something, and on backends with nothing to add.
    pub fn backend_detail(&self) -> Option<String> {
        self.backend.detail()
    }

    /// Submit input.
    ///
    /// A packet may hold one access unit, several, or part of one: the framing is worked
    /// out from the bitstream. Frames are NOT produced synchronously - call
    /// [`Decoder::receive`] until it returns `None`.
    pub fn send(&mut self, packet: Packet<'_>) -> Result<()> {
        let mut units = std::mem::take(&mut self.au_scratch);
        units.clear();
        let mut result = self.split(packet.data, packet.pts, &mut units);
        if result.is_ok() && self.packets_are_access_units() {
            // The caller says this packet was a whole picture, so it can be closed now
            // rather than when the next one arrives.
            result = self.splitter.finish(&mut units);
        }
        if result.is_ok() {
            for au in &units {
                self.submit(au)?;
            }
        }
        self.au_scratch = units;
        result
    }

    /// Whether packets are whole access units: the caller's answer, or the inference.
    fn packets_are_access_units(&self) -> bool {
        self.config.packets_are_access_units.unwrap_or(match self.config.input {
            InputFormat::LengthPrefixed { .. } => true,
            InputFormat::AnnexB => false,
            InputFormat::Auto => false,
        })
    }

    /// Take the next decoded frame, or `None` when none is ready yet.
    ///
    /// "Not ready yet" is normal: every one of these decoders is pipelined, and a stream
    /// with B-frames holds pictures back on purpose. A caller decoding a file loops
    /// `send` / `receive` and then calls [`Decoder::finish`]; a caller playing live video
    /// simply calls `receive` again on the next tick.
    pub fn receive(&mut self) -> Result<Option<Frame>> {
        if self.ready.is_empty() {
            self.pump(false)?;
        }
        if self.ready.is_empty() {
            return Ok(None);
        }
        let mut frame = self.ready.remove(0);
        if let Some(pts) = self.pts_by_key.remove(&frame.pts) {
            frame.pts = pts;
        }
        Ok(Some(frame))
    }

    /// [`Decoder::receive`], waiting for the decoder when it still owes frames.
    ///
    /// On the three native platforms this is exactly `receive`: those decoders produce
    /// their output synchronously inside the call, so there is never anything to wait for
    /// and the future completes immediately.
    ///
    /// In the BROWSER it is the one that works. WebCodecs hands frames back on a callback
    /// and their pixels become readable only when a second promise settles, so a browser
    /// caller that only ever calls `receive` sees `None` until it yields to the event loop.
    /// This waits on the callbacks themselves - not on a timer - and returns `None` only
    /// when the decoder owes nothing more.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn receive_async(&mut self) -> Result<Option<Frame>> {
        self.receive()
    }

    /// [`Decoder::receive`], waiting for WebCodecs' callbacks when the decoder still owes
    /// frames. See the native version of this method for what it is for.
    #[cfg(target_arch = "wasm32")]
    pub async fn receive_async(&mut self) -> Result<Option<Frame>> {
        loop {
            if let Some(frame) = self.receive()? {
                return Ok(Some(frame));
            }
            let Some(event) = self.backend.pending_event() else {
                return Ok(None);
            };
            // A rejected promise is not itself a decode failure: the decoder reports those
            // through its error callback, which the next `receive` picks up.
            let _ = wasm_bindgen_futures::JsFuture::from(event).await;
        }
    }

    /// End of stream: decode everything still buffered and hold it for [`Decoder::receive`].
    ///
    /// After this the decoder is ready for more input, so a caller that reached the end of
    /// one file can keep going with the next.
    ///
    /// In the browser, `finish` STARTS the flush rather than completing it - WebCodecs'
    /// own flush is a promise - so a browser caller drains with [`Decoder::receive_async`]
    /// afterwards, which waits for it.
    pub fn finish(&mut self) -> Result<()> {
        let mut units = std::mem::take(&mut self.au_scratch);
        units.clear();
        let split = self.splitter.finish(&mut units);
        if split.is_ok() {
            for au in &units {
                self.submit(au)?;
            }
        }
        self.au_scratch = units;
        split?;
        self.pump(true)
    }

    /// Discard all state for a seek: buffered input, held frames, reference pictures.
    ///
    /// Parameter sets are kept, because a seek within one stream does not invalidate them,
    /// and the next access unit after a seek is an IDR that carries its own anyway.
    pub fn reset(&mut self) -> Result<()> {
        self.splitter.flush_state();
        self.backend.reset()?;
        self.reorderer.clear();
        for frame in self.ready.drain(..) {
            self.pool.put(frame.data);
        }
        self.pts_by_key.clear();
        self.key_base = self.max_key + 1;
        Ok(())
    }

    /// Hand a frame's memory back for reuse. Optional, and worth doing in a playback loop:
    /// it is what keeps steady decoding from allocating a frame-sized block per frame.
    pub fn recycle(&mut self, frame: Frame) {
        self.pool.put(frame.data);
    }

    /// Parse `extradata` (avcC or Annex B parameter sets) into the parameter set store.
    fn load_extradata(&mut self, extra: &[u8]) -> Result<()> {
        let annex_b = if bitstream::nal::is_annex_b(extra) {
            extra.to_vec()
        } else {
            let record = AvcC::parse(extra)?;
            if let InputFormat::Auto = self.config.input {
                self.config.input = InputFormat::LengthPrefixed { length_size: record.length_size };
            }
            record.to_annex_b()
        };
        let mut units = Vec::new();
        self.splitter.push_annex_b(&annex_b, &mut units)?;
        debug_assert!(units.is_empty(), "parameter sets alone cannot complete a picture");
        Ok(())
    }

    /// Normalise a packet into access units.
    fn split(&mut self, data: &[u8], pts: Option<i64>, out: &mut Vec<AccessUnit>) -> Result<()> {
        let length_size = match self.config.input {
            InputFormat::AnnexB => None,
            InputFormat::LengthPrefixed { length_size } => Some(length_size),
            InputFormat::Auto => {
                if bitstream::nal::is_annex_b(data) {
                    None
                } else {
                    Some(4)
                }
            }
        };
        match length_size {
            None => self.splitter.push_annex_b_at(data, pts, out),
            Some(size) => {
                for nal in bitstream::nal::split_length_prefixed(data, size)? {
                    self.splitter.push_nal_at(nal, pts, out)?;
                }
                Ok(())
            }
        }
    }

    /// Configure the backend if needed, then hand it one access unit.
    fn submit(&mut self, au: &AccessUnit) -> Result<()> {
        if !self.configured || au.config_changed {
            self.configure(au)?;
        }
        // Picture order counts restart at every IDR, so they are rebased onto a key that
        // only ever increases. Without this an MP4 with two GOPs would hand the backend two
        // pictures with the same timestamp, and every one of these decoders reorders on
        // timestamps.
        if au.idr {
            self.key_base = self.max_key + 1;
        }
        let key = self.key_base + au.order() as i64;
        self.max_key = self.max_key.max(key);
        if let Some(pts) = au.pts {
            self.pts_by_key.insert(key, pts);
            // A decoder that drops a picture (a corrupt access unit, a stream joined
            // mid-GOP) never claims its key back, so the map is trimmed to a window around
            // what is still in flight rather than growing for the life of the stream.
            if self.pts_by_key.len() > PTS_WINDOW {
                let oldest = key - PTS_WINDOW as i64;
                self.pts_by_key.retain(|&k, _| k >= oldest);
            }
        }
        self.backend.send(au, key)
    }

    /// Push the active parameter sets into the backend.
    fn configure(&mut self, au: &AccessUnit) -> Result<()> {
        let sps = &au.sps;
        if sps.chroma_format_idc != 1 {
            return Err(Error::unsupported(format!(
                "chroma_format_idc {} (only 4:2:0 is supported)",
                sps.chroma_format_idc
            )));
        }
        if sps.bit_depth_luma != 8 || sps.bit_depth_chroma != 8 {
            return Err(Error::unsupported(format!(
                "{}-bit luma / {}-bit chroma (only 8-bit is supported)",
                sps.bit_depth_luma, sps.bit_depth_chroma
            )));
        }
        let avcc = AvcC::from_parameter_sets(
            self.splitter.sets.sps_nals(),
            self.splitter.sets.pps_nals(),
            4,
        )?;
        let info = StreamInfo {
            width: sps.width(),
            height: sps.height(),
            profile_idc: sps.profile_idc,
            level_idc: sps.level_idc,
            color: sps.vui.map(|v| v.color).unwrap_or(ColorInfo::UNSPECIFIED),
            sample_aspect_ratio: sps.vui.and_then(|v| v.sar),
            timing: sps.vui.and_then(|v| v.timing),
            max_reorder_frames: sps.max_reorder_frames(),
        };
        self.backend.configure(StreamConfig {
            sps,
            avcc: &avcc,
            width: info.width,
            height: info.height,
        })?;
        if self.backend.output_order() == OutputOrder::Decode {
            self.reorderer.set_depth(info.max_reorder_frames as usize);
        }
        self.info = Some(info);
        self.configured = true;
        self.scratch.clear();
        Ok(())
    }

    /// Move frames from the backend into `ready`, reordering if the backend needs it.
    fn pump(&mut self, drain: bool) -> Result<()> {
        if !self.configured {
            // Nothing has been submitted yet, and a platform decoder asked for output
            // before its input type is set reports an error rather than "nothing yet".
            return Ok(());
        }
        let mut fresh = Vec::new();
        if drain {
            self.backend.drain(&mut self.pool, &mut fresh)?;
        } else {
            self.backend.poll(&mut self.pool, &mut fresh)?;
        }
        let colour = self.info.map(|i| i.color).unwrap_or(ColorInfo::UNSPECIFIED);
        let reorder = self.backend.output_order() == OutputOrder::Decode;
        for mut frame in fresh {
            // The VUI is the stream's word on colour and no platform decoder reports it
            // back, so it is stamped on here rather than left unspecified.
            if frame.color == ColorInfo::UNSPECIFIED {
                frame.color = colour;
            }
            frame.validate()?;
            if reorder {
                self.reorderer.push(frame, &mut self.ready);
            } else {
                self.ready.push(frame);
            }
        }
        if drain && reorder {
            self.reorderer.drain(&mut self.ready);
        }
        Ok(())
    }
}

impl core::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Decoder")
            .field("backend", &self.backend.name())
            .field("info", &self.info)
            .field("ready", &self.ready.len())
            .finish()
    }
}
