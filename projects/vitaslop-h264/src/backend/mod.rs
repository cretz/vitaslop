//! The platform seam.
//!
//! One trait, four implementations, and a rule: a backend NEVER decodes H.264 itself. Each
//! one drives the decoder the operating system already ships - Media Foundation, Video
//! Toolbox, VA-API, WebCodecs - because that is the only way to reach the fixed-function
//! video block, which is the difference between decoding 1080p on a phone at 3% of a core
//! and not decoding it at all.
//!
//! The common layer above ([`crate::Decoder`]) hands each backend whole access units and
//! takes back frames. Everything a backend needs to know about the stream it gets as
//! parsed structures - it never re-parses the bitstream.

use crate::bitstream::AccessUnit;
use crate::bitstream::avcc::AvcC;
use crate::bitstream::sps::Sps;
use crate::error::Result;
use crate::frame::{Frame, PixelFormat};

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_arch = "wasm32")]
pub mod web;
#[cfg(target_os = "windows")]
pub mod windows;

/// The order a backend emits frames in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputOrder {
    /// Frames come out in presentation order already: the backend runs its own DPB.
    /// Media Foundation, WebCodecs and the VA-API backend (whose DPB is ours) all do.
    Presentation,
    /// Frames come out in decode order and the common layer must reorder them. Video
    /// Toolbox does this unless asked to buffer, and asking it to buffer just moves the
    /// same delay somewhere less visible.
    Decode,
}

/// Whether decoding is actually reaching the machine's video hardware.
///
/// This exists because falling back to software is INVISIBLE otherwise: the frames are
/// correct either way, and the only symptom is that decoding costs ten times what it should.
/// A backend that cannot prove which it got says [`Acceleration::Unknown`] rather than
/// claiming the flattering answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Acceleration {
    /// Frames are being produced by fixed-function video hardware.
    Hardware,
    /// Frames are being decoded on the CPU, for the reason given.
    Software(String),
    /// Not established - either nothing has been decoded yet, or this backend has no way
    /// to ask.
    Unknown,
}

/// What a backend needs to be told when the stream's configuration appears or changes.
pub struct StreamConfig<'a> {
    /// The active sequence parameter set.
    pub sps: &'a Sps,
    /// The parameter sets in avcC form - what VideoToolbox and WebCodecs want directly.
    pub avcc: &'a AvcC,
    /// Visible size after cropping.
    pub width: u32,
    /// Visible size after cropping.
    pub height: u32,
}

/// A platform decoder.
///
/// `Send` because the engines that embed this crate move a decoder between threads as a
/// matter of course (a cooperative scheduler resumes a fiber wherever it likes). Each
/// backend justifies its own `Send` where the platform types behind it are not obviously
/// so; none of them is used from two threads AT ONCE, which is what the platform APIs
/// actually require.
pub trait Backend: Send {
    /// Human-readable name, for diagnostics and for tests that assert which path ran.
    fn name(&self) -> &'static str;

    /// Ordering guarantee of [`Backend::poll`].
    fn output_order(&self) -> OutputOrder;

    /// Called before the first access unit, and again whenever the parameter sets change
    /// in a way that alters the stream's shape.
    fn configure(&mut self, config: StreamConfig<'_>) -> Result<()>;

    /// Submit one access unit. `timestamp` is the common layer's presentation-order key;
    /// a backend must carry it through to the frame it produces.
    fn send(&mut self, au: &AccessUnit, timestamp: i64) -> Result<()>;

    /// Collect whatever frames are ready, without blocking. `pool` supplies (and receives)
    /// pixel buffers.
    fn poll(&mut self, pool: &mut FramePool, out: &mut Vec<Frame>) -> Result<()>;

    /// End of stream: emit every frame still held back. After this the backend must accept
    /// new input again (the next thing a caller does is usually seek).
    fn drain(&mut self, pool: &mut FramePool, out: &mut Vec<Frame>) -> Result<()>;

    /// Discard all state for a seek: pending input, held frames, reference pictures.
    fn reset(&mut self) -> Result<()>;

    /// Whether decoding is reaching video hardware. Defaults to
    /// [`Acceleration::Unknown`]: a backend answers this only where it can prove it.
    fn acceleration(&self) -> Acceleration {
        Acceleration::Unknown
    }

    /// Anything this backend learned from RUNNING that a caller's one diagnostic line
    /// should carry - the layout a browser's decoder actually produced, say.
    ///
    /// It exists because the interesting facts about a platform decoder are not knowable
    /// before it decodes something, and a caller that has to ask for them separately will
    /// not have asked on the run that went wrong.
    fn detail(&self) -> Option<String> {
        None
    }

    /// A promise that settles when there may be more output, or `None` when the backend
    /// owes nothing and waiting cannot help.
    ///
    /// Only the browser needs this: WebCodecs delivers frames on a callback, so
    /// [`crate::Decoder::receive_async`] has to have something to await. The three native
    /// backends produce their output inside `poll`, so they never return one.
    #[cfg(target_arch = "wasm32")]
    fn pending_event(&self) -> Option<js_sys::Promise> {
        None
    }
}

/// Recycled pixel buffers.
///
/// Playback allocates one frame-sized block per frame otherwise, which at 1080p is 3 MB
/// sixty times a second through the general allocator. The pool is a plain free list: a
/// caller that never returns frames simply gets fresh allocations, so nothing breaks if
/// [`crate::Decoder::recycle`] is never called.
#[derive(Debug, Default)]
pub struct FramePool {
    free: Vec<Vec<u8>>,
    /// Never hold more than this many spare buffers, so a decoder that outlives a burst
    /// does not sit on the burst's memory.
    limit: usize,
}

impl FramePool {
    /// A pool holding at most `limit` spare buffers.
    pub fn new(limit: usize) -> Self {
        FramePool { free: Vec::new(), limit }
    }

    /// Take a buffer to build a frame in.
    pub fn take(&mut self) -> Vec<u8> {
        self.free.pop().unwrap_or_default()
    }

    /// Hand a buffer back.
    pub fn put(&mut self, mut buf: Vec<u8>) {
        if self.free.len() < self.limit {
            buf.clear();
            self.free.push(buf);
        }
    }

    /// Shape a frame of `format` at `width` x `height` out of a pooled buffer.
    pub fn frame(&mut self, format: PixelFormat, width: u32, height: u32) -> Frame {
        let buf = self.take();
        Frame::alloc(format, width, height, buf)
    }
}

/// Puts decode-order output back into presentation order.
///
/// Only used for backends that report [`OutputOrder::Decode`]. The depth comes from the
/// stream (`max_num_reorder_frames`, or the DPB size when the VUI does not say), so a
/// stream with no B-frames adds no latency at all.
#[derive(Debug, Default)]
pub struct Reorderer {
    held: Vec<Frame>,
    depth: usize,
}

impl Reorderer {
    /// Set the reorder depth in frames. Zero means "output immediately".
    pub fn set_depth(&mut self, depth: usize) {
        self.depth = depth;
    }

    /// Take a decoded frame in, and emit whatever is now safe to output.
    pub fn push(&mut self, frame: Frame, out: &mut Vec<Frame>) {
        let at = self.held.partition_point(|f| f.pts <= frame.pts);
        self.held.insert(at, frame);
        while self.held.len() > self.depth {
            out.push(self.held.remove(0));
        }
    }

    /// Emit everything held, in order.
    pub fn drain(&mut self, out: &mut Vec<Frame>) {
        out.append(&mut self.held);
    }

    /// Throw held frames away (a seek).
    pub fn clear(&mut self) {
        self.held.clear();
    }
}

/// How a backend should be built.
#[derive(Debug, Clone, Copy)]
pub struct BackendOptions {
    /// Whether to use video hardware.
    pub hardware: bool,
    /// Largest coded picture, in bytes, that the hardware path may be handed. Above it the
    /// backend reports rather than decodes - see [`crate::DecoderConfig::max_hardware_picture_bytes`].
    pub max_hardware_picture_bytes: usize,
    /// Emit each picture as soon as it is decoded - see [`crate::DecoderConfig::low_latency`].
    pub low_latency: bool,
}

/// Create the backend for this platform.
///
/// The only place a `cfg` decides which decoder runs. A platform with no implementation
/// gets [`crate::Error::NoDecoder`] rather than a compile error, so a caller can build for
/// it and fall back on something else.
///
/// `options.hardware` is a request, not a guarantee: a backend that cannot honour it says
/// so through [`Backend::acceleration`] rather than failing.
pub fn create(options: BackendOptions) -> Result<Box<dyn Backend>> {
    #[cfg(target_os = "windows")]
    {
        Ok(Box::new(windows::MediaFoundationBackend::new(options)?))
    }
    #[cfg(target_os = "macos")]
    {
        // Video Toolbox IS the hardware path on macOS, and a caller who wants software has
        // no reason to come through this crate for it - so the request is noted and the
        // limit, which is a DXVA-specific one, does not apply.
        let _ = options;
        Ok(Box::new(macos::VideoToolboxBackend::new()?))
    }
    #[cfg(target_os = "linux")]
    {
        // Likewise: VA-API's VLD entry point is the hardware decoder.
        let _ = options;
        Ok(Box::new(linux::VaapiBackend::new()?))
    }
    #[cfg(target_arch = "wasm32")]
    {
        Ok(Box::new(web::WebCodecsBackend::new(options.hardware, options.low_latency)?))
    }
    #[cfg(not(any(
        target_os = "windows",
        target_os = "macos",
        target_os = "linux",
        target_arch = "wasm32"
    )))]
    {
        Err(crate::error::Error::no_decoder(format!(
            "no H.264 backend is implemented for {}",
            std::env::consts::OS
        )))
    }
}
