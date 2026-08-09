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

use std::collections::VecDeque;
use std::sync::Mutex;
use tracing_subscriber::fmt::MakeWriter;

/// How many DISTINCT WARN/ERROR lines the page mirror holds.
///
/// Distinct, not total: repeats are counted against the line already held (see
/// [`push_page_log`]), so this bounds the panel by how many different things went wrong
/// rather than by how often. The race screen produced 70+ occurrences of one warning shape
/// across dozens of program pairs and evicted six DIFFERENT warnings to fit them - including
/// the three that named real render defects. A ring that drops a unique warning to keep the
/// hundredth copy of another has its priority exactly backwards.
const PAGE_LOG_CAP: usize = 96;

/// The WARN/ERROR lines this run has emitted, for the page's diagnostics panel.
///
/// # Why the console is not where these can live
/// The console is unreachable on a phone without a cable and remote debugging, and the phone
/// is the only machine whose numbers are not a proxy. That argument is already written out at
/// the `diag` emit in `lib.rs` - and it was made for the COUNTERS while every warning and
/// error kept going to the console alone. So the one line that settles "did this draw fail, or
/// did the guest never submit it" - `WebGPU uncaptured error: ...`, a dropped draw, a renderer
/// falling back - was reachable only from a desktop. A session's notes recorded it as already
/// being in the page's diagnostics box; it was not, and a device capture taken on that belief
/// would have read a SILENT panel as "no error occurred".
///
/// This is a mirror, not a second sink: a line reaches it only if the filter already let it
/// through to the console, so `VITASLOP_LOG` still says what is emitted.
static PAGE_LOG: Mutex<Option<PageLog>> = Mutex::new(None);

#[derive(Default)]
struct PageLog {
    /// Each distinct line, with how many times it has been emitted.
    lines: VecDeque<(String, u64)>,
    /// Distinct lines pushed out of the ring. Reported rather than dropped silently - a panel
    /// showing the LAST N warnings while implying it shows all of them is the failure this
    /// project keeps meeting under other names.
    dropped: usize,
}

use vitaslop_platform::diag::dedupe_key;

/// The WARN/ERROR lines so far, oldest first, or `None` if there were none.
///
/// Non-draining on purpose: the panel is rebuilt from scratch each perf window, and a warning
/// that fired once must not vanish from it one window later.
pub fn page_log_report() -> Option<String> {
    let guard = PAGE_LOG.lock().ok()?;
    let log = guard.as_ref()?;
    if log.lines.is_empty() {
        return None;
    }
    let mut out = String::new();
    if log.dropped > 0 {
        out.push_str(&format!(
            "({} earlier DISTINCT line(s) dropped - this panel keeps {PAGE_LOG_CAP} of them)\n",
            log.dropped
        ));
    }
    for (l, n) in &log.lines {
        out.push_str(l);
        if *n > 1 {
            out.push_str(&format!("  [x{n}]"));
        }
        out.push('\n');
    }
    Some(out)
}

fn push_page_log(text: &str) {
    let Ok(mut guard) = PAGE_LOG.lock() else { return };
    let log = guard.get_or_insert_with(PageLog::default);
    let key = dedupe_key(text);
    // A repeat updates the line already held - keeping the LATEST text, so a diagnostic that
    // reports its own running count shows the newest one rather than the first.
    if let Some(slot) = log.lines.iter_mut().find(|(l, _)| dedupe_key(l) == key) {
        slot.0 = text.to_string();
        slot.1 += 1;
        return;
    }
    if log.lines.len() == PAGE_LOG_CAP {
        log.lines.pop_front();
        log.dropped += 1;
    }
    log.lines.push_back((text.to_string(), 1));
}


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
    /// Also mirror this event to the page panel (WARN and ERROR only).
    to_page: bool,
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
        let text = text.trim_end();
        web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(text));
        if self.to_page {
            push_page_log(text);
        }
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
        ConsoleWriter { buf: Vec::new(), to_page: false }
    }

    /// The per-event writer, which is where the LEVEL is knowable. `make_writer` above has no
    /// metadata, so a mirror decided there would either copy every line to the page (the perf
    /// windows are hundreds of lines a run) or none.
    fn make_writer_for(&'a self, meta: &tracing::Metadata<'_>) -> ConsoleWriter {
        ConsoleWriter {
            buf: Vec::new(),
            to_page: *meta.level() <= tracing::Level::WARN,
        }
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
