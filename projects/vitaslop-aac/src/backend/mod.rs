//! One decoder, per platform, behind one trait.

use crate::error::{Error, Result};
use crate::{DecoderConfig, Pcm};

#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_arch = "wasm32")]
mod web;

/// What every platform's decoder has to do. Deliberately the smallest surface that can
/// serve an asynchronous decoder honestly: nothing here promises output for an input.
pub(crate) trait Backend: Send {
    fn submit(&mut self, es: &[u8], pts: i64) -> Result<()>;
    fn poll(&mut self) -> Result<Option<Pcm>>;
    fn reset(&mut self) -> Result<()>;
    fn describe(&self) -> String;
}

/// Open the decoder this platform has.
pub(crate) fn open(config: &DecoderConfig) -> Result<Box<dyn Backend>> {
    #[cfg(target_os = "windows")]
    {
        return Ok(Box::new(windows::MediaFoundationAac::new(config)?));
    }
    #[cfg(target_arch = "wasm32")]
    {
        return Ok(Box::new(web::WebCodecsAac::new(config)?));
    }
    #[allow(unreachable_code)]
    {
        let _ = config;
        // macOS (AudioConverter) and Linux (no single system decoder) are not written
        // yet. Reported as what it is - a platform this crate does not cover - rather
        // than as a stream problem, so the caller says "no sound on this host" and not
        // "this movie is broken".
        Err(Error::NoDecoder(format!(
            "this build has no AAC decoder for {}",
            std::env::consts::OS
        )))
    }
}
