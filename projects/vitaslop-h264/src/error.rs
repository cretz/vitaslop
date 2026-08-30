//! One error type for every backend.
//!
//! The rule this crate follows everywhere: an input it does not fully understand is an
//! `Unsupported` error, never a silently approximated decode. A caller that gets a frame
//! back can trust that the frame is what the bitstream says it is.

use core::fmt;

/// What went wrong.
#[derive(Debug)]
pub enum Error {
    /// The bitstream is malformed: a NAL that ends mid-syntax-element, a reserved value
    /// where the spec allows none, a parameter set that references one that was never sent.
    Bitstream(String),
    /// The bitstream is well formed but uses a tool this crate does not implement
    /// (interlaced coding on the VA-API path, a chroma format other than 4:2:0, a profile
    /// the platform decoder refuses). Never returned for something that was decoded
    /// approximately - nothing here approximates.
    Unsupported(String),
    /// No platform decoder could be created: no libva on the machine, no WebCodecs in the
    /// browser, Media Foundation's H.264 MFT missing (an "N" edition of Windows without the
    /// Media Feature Pack). The caller is expected to fall back on this, so it carries the
    /// reason as text rather than dying.
    NoDecoder(String),
    /// The platform decoder was created but failed a call. Carries the platform status code
    /// where there is one (`HRESULT`, `OSStatus`, `VAStatus`, a DOMException name).
    Platform {
        /// Which call failed, in the platform's own vocabulary.
        call: &'static str,
        /// The platform's status code, when it has a numeric one.
        code: i64,
        /// Whatever text the platform gave.
        detail: String,
    },
    /// The decoder was fed after it was flushed, or drained while it still owed input.
    /// A caller bug, not a stream problem.
    State(&'static str),
}

impl Error {
    pub(crate) fn bitstream(msg: impl Into<String>) -> Self {
        Error::Bitstream(msg.into())
    }
    pub(crate) fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
    #[allow(dead_code)]
    pub(crate) fn no_decoder(msg: impl Into<String>) -> Self {
        Error::NoDecoder(msg.into())
    }
    #[allow(dead_code)]
    pub(crate) fn platform(call: &'static str, code: i64, detail: impl Into<String>) -> Self {
        Error::Platform { call, code, detail: detail.into() }
    }

    /// True when the caller's sensible move is to try another decoder (or software) rather
    /// than to report a failure: the machine simply has no H.264 decoding to offer.
    pub fn is_missing_decoder(&self) -> bool {
        matches!(self, Error::NoDecoder(_))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Bitstream(m) => write!(f, "malformed H.264 bitstream: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported H.264 feature: {m}"),
            Error::NoDecoder(m) => write!(f, "no platform H.264 decoder available: {m}"),
            Error::Platform { call, code, detail } => {
                write!(f, "{call} failed (0x{:x}): {detail}", *code as u64)
            }
            Error::State(m) => write!(f, "decoder used out of order: {m}"),
        }
    }
}

impl std::error::Error for Error {}

/// This crate's result alias.
pub type Result<T> = core::result::Result<T, Error>;
