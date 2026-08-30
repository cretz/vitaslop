//! What can go wrong reaching a host's AAC decoder.

/// A decode error.
#[derive(Debug, Clone)]
pub enum Error {
    /// This host has no AAC decoder at all, or not one this crate can drive. A normal
    /// outcome, not a failure: a movie without sound is still a movie, and the caller is
    /// expected to say so and carry on.
    NoDecoder(String),
    /// The stream, or something in it, is not decodable here.
    Stream(String),
    /// A platform call failed. Carries what failed and the platform's own code, because
    /// "the decoder did not work" is not something anyone can act on.
    Platform { what: &'static str, code: i32, detail: String },
}

impl Error {
    /// Whether this is "no decoder here" rather than "this stream is bad", which decides
    /// whether a caller reports a missing feature or a broken file.
    pub fn is_missing_decoder(&self) -> bool {
        matches!(self, Error::NoDecoder(_))
    }

    pub(crate) fn platform(what: &'static str, code: i32, detail: impl Into<String>) -> Error {
        Error::Platform { what, code, detail: detail.into() }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::NoDecoder(m) => write!(f, "no AAC decoder on this host: {m}"),
            Error::Stream(m) => write!(f, "AAC stream cannot be decoded: {m}"),
            Error::Platform { what, code, detail } => {
                write!(f, "{what} failed ({code:#010x}): {detail}")
            }
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;
