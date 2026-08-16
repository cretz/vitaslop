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
use std::sync::atomic::{AtomicBool, Ordering};
use tracing_subscriber::fmt::MakeWriter;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;

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

/// Set once, so a second entry point calling this is a no-op.
static PANIC_HOOK_INSTALLED: AtomicBool = AtomicBool::new(false);

/// The global function a host page/worker may expose to receive a panic report.
///
/// A string, not a channel: the panic hook runs on the edge of an abort and must not depend on
/// anything it might have poisoned.
const PANIC_SINK: &str = "__vitaslopPanic";

/// Frame-name fragments that belong to the PANIC MACHINERY rather than to the code that
/// panicked. Every one of these sits between the hook and the real fault, in a fixed order.
///
/// They are dropped from the FRONT of the stack only, so a legitimate later frame that happens
/// to contain one of these strings is kept.
const PANIC_MACHINERY: &[&str] = &[
    "js_sys::Error::new",
    "__wbg_new",
    "logging::install_panic_hook",
    "panicking::",
    "rust_begin_unwind",
    "panic_fmt",
    "__rust_end_short_backtrace",
];

/// The JS stack at the panic, with the hook's own frames removed.
///
/// # Why the trim is not cosmetic
/// The JS stack is what names the FRAMES; the Rust location names only the line that gave up.
/// A panic inside shared code (a slice index, an `unwrap` on a `None` a caller produced) is
/// attributed only by the frames above it.
///
/// V8 caps a stack at TEN frames by default, and getting here costs six of them - the hook, the
/// `Error` it constructs, and the four `std` panicking frames. MEASURED before this trim: the
/// captured stack ended one frame into real code. So the cap is raised and the machinery is
/// dropped, which turns ten frames of overhead-plus-nothing into thirty frames of caller.
///
/// `stack` and `stackTraceLimit` are both V8/SpiderMonkey extensions rather than standard, so
/// both are reached reflectively and an engine without them yields no stack rather than an error.
fn panic_stack() -> String {
    let error_ctor = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("Error")).ok();
    if let Some(ctor) = &error_ctor {
        let _ = js_sys::Reflect::set(
            ctor,
            &JsValue::from_str("stackTraceLimit"),
            &JsValue::from_f64(30.0),
        );
    }
    let raw = js_sys::Reflect::get(
        js_sys::Error::new("panic").as_ref(),
        &JsValue::from_str("stack"),
    )
    .ok()
    .and_then(|v| v.as_string())
    .unwrap_or_default();

    // The first line is the Error's own message ("Error: panic"), which is this hook talking to
    // itself; the frames follow. Drop that, then drop leading machinery frames.
    let mut lines = raw.lines();
    if raw.starts_with("Error") {
        lines.next();
    }
    let kept: Vec<&str> = lines
        .skip_while(|l| PANIC_MACHINERY.iter().any(|m| l.contains(m)))
        .collect();
    kept.join("\n")
}

/// Install the panic hook that puts the panic MESSAGE where a phone can read it.
///
/// # Why `console_error_panic_hook` is not enough
/// A Rust panic under `panic = "abort"` reaches the browser as
/// `Uncaught RuntimeError: unreachable at ...vitaslop_web_bg.wasm:1:3542933`, and that offset is
/// worthless: the wasm ships `lto = "fat"` with `codegen-units = 1`, so there is no symbol to
/// resolve it against. The only useful text - `panicked at src/....rs:NNN: <message>` - is
/// printed by the panic hook, and `console_error_panic_hook` prints it to the CONSOLE and
/// nowhere else.
///
/// **On a phone there is no console.** The device is the only machine whose numbers are not a
/// proxy, and it is the machine where the one line that names the fault was unreachable. A
/// device report therefore arrived as "it crashed, here is a wasm offset", which is a defect
/// that cannot be worked on. That is the same argument the counters and the WARN/ERROR mirror
/// above were already moved on; the panic - the single most valuable line the emulator can ever
/// emit - was the one thing left behind.
///
/// So the hook writes the panic THREE ways, and each covers a case the others do not:
/// 1. `console.error`, for a desktop run with devtools open (what we had).
/// 2. [`push_page_log`], so it appears in the on-page diagnostics panel and in every later
///    `/diag` dump - which is what reaches disk on the dev server.
/// 3. A call to `globalThis.__vitaslopPanic`, if the host defined one. The run worker wires that
///    to `postMessage`, so the page can show the text AT ONCE rather than waiting for a perf
///    window that a dead worker will never publish. The panel is rebuilt from reports, and after
///    a panic there are no more reports - so (2) alone would show the panic only if something
///    else happened to be still running.
///
/// The hook is deliberately total: no `unwrap`, no allocation it cannot afford to lose, and a
/// missing sink is silence rather than a second panic inside the panic hook.
pub fn install_panic_hook() {
    if PANIC_HOOK_INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    std::panic::set_hook(Box::new(|info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown location>".to_string());
        // `payload_as_str` covers the two payload shapes a `panic!` can produce (`&str` and
        // `String`); anything else is a payload nobody in this workspace creates.
        let message = info.payload_as_str().unwrap_or("<non-string panic payload>");
        let stack = panic_stack();
        let text = if stack.is_empty() {
            format!("PANIC at {location}: {message}")
        } else {
            format!("PANIC at {location}: {message}\n{stack}")
        };

        web_sys::console::error_1(&JsValue::from_str(&text));
        push_page_log(&text);

        let global = js_sys::global();
        if let Ok(sink) = js_sys::Reflect::get(&global, &JsValue::from_str(PANIC_SINK)) {
            if let Some(f) = sink.dyn_ref::<js_sys::Function>() {
                let _ = f.call1(&JsValue::NULL, &JsValue::from_str(&text));
            }
        }
    }));
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
