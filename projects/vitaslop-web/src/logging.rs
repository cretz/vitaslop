//! The browser's `tracing` seam: the runtime's diagnostics, on the console.
//!
//! # Why the browser was silent
//! Every crate below this one reports through `tracing` (`vitaslop::io`, `vitaslop::sema`,
//! `vitaslop::thread`, ...), and the desktop binary installs a `tracing_subscriber::fmt`
//! over stderr filtered by `RUST_LOG`. The browser installed nothing, so `tracing` dropped
//! every event - the emulator ran completely mute in the one place a debugger, a stderr
//! pipe and a profiler are all hardest to reach. This module is the browser half of that
//! seam: same events, same filter syntax, written to `console.log`.
//!
//! The filter comes from the `VITASLOP_LOG` knob rather than `RUST_LOG`, because
//! `wasm32-unknown-unknown` has no environment (see [`vitaslop_platform::knobs`]).

use tracing_subscriber::fmt::MakeWriter;

/// The default filter when `VITASLOP_LOG` is unset: warnings and errors only.
///
/// A player's browser should be quiet. Everything below `warn` here is diagnostic - the
/// per-frame timing split, the host-call rate, the I/O and semaphore traces - and it is
/// for a run someone is investigating, not for a run someone is playing. The test
/// harness asks for it by name (`VITASLOP_LOG=warn,vitaslop::perf=info`), which is also
/// how a user can turn it on when reporting a problem.
///
/// Note this only silences OUTPUT. Anything that indicates the emulator is not being
/// faithful - an unimplemented NID, a renderer falling back, a software adapter - is a
/// hard failure or a `warn`, not an `info`, so no filter can hide it.
const DEFAULT_FILTER: &str = "warn";

/// A line-buffered writer that emits each completed line to the browser console.
///
/// `console.log` is per-message, not a stream, so the formatter's several small writes
/// per event have to be joined before they are worth emitting - otherwise one event
/// becomes half a dozen console entries broken mid-word.
struct ConsoleWriter {
    buf: Vec<u8>,
}

impl std::io::Write for ConsoleWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let text = String::from_utf8_lossy(&self.buf);
        web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(text.trim_end()));
        self.buf.clear();
        Ok(())
    }
}

impl Drop for ConsoleWriter {
    /// The fmt layer drops the writer at the end of each event rather than flushing it,
    /// so this is where a line actually reaches the console.
    fn drop(&mut self) {
        use std::io::Write;
        let _ = self.flush();
    }
}

struct MakeConsoleWriter;

impl<'a> MakeWriter<'a> for MakeConsoleWriter {
    type Writer = ConsoleWriter;
    fn make_writer(&'a self) -> ConsoleWriter {
        ConsoleWriter { buf: Vec::new() }
    }
}

/// Install the console subscriber. Idempotent - a second call is a no-op, so the
/// main-thread and worker entries can both call it without ordering rules.
///
/// Call AFTER the page's knobs are applied, so `VITASLOP_LOG` is visible.
pub fn init() {
    let filter = vitaslop_runtime::knobs::var("VITASLOP_LOG")
        .unwrap_or_else(|_| DEFAULT_FILTER.to_string());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(&filter))
        .with_writer(MakeConsoleWriter)
        // No ANSI (the console shows the escapes literally) and no timestamp
        // (`SystemTime` is not available on `wasm32-unknown-unknown`; the console
        // stamps every line itself anyway).
        .with_ansi(false)
        .without_time()
        .try_init();
}
