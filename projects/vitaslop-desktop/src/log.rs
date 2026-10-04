//! The desktop's `tracing` seam: where a diagnostic goes when nobody asked for one.
//!
//! # A player is not a rig, and stderr could not tell them apart
//! Every crate below this one reports through `tracing`, the default filter is `warn`, and
//! the binary wrote all of it to stderr. For the rigs that is exactly right - `--game` and
//! `--headless` runs are read from a terminal, and every recorded repro command depends on
//! their output being what it has always been. For the SHELL it is wrong: opening the
//! library and pressing Play printed a link-time inventory of unhandled imports, a decode-gap
//! summary, the transpiler's statement count, then MSAA, texture re-encode, presentation and
//! NGS warnings - a wall of text addressed to whoever is implementing the emulator, shown to
//! whoever is trying to play a game.
//!
//! # The fix is the AUDIENCE, not the level
//! Lowering these to `debug` would be silencing: a diagnostic nobody sees does not exist, and
//! this project has met that failure under several names. So nothing is dropped and nothing
//! is re-levelled. Every event is CAPTURED, in full, into the shared rings
//! (`vitaslop_platform::diag`) that the browser panel already reads and that the shell's own
//! Diagnostics view now reads too. Only the stderr MIRROR is conditional, and its condition
//! is the one thing that actually distinguishes the two audiences: whether a human named
//! `VITASLOP_LOG` or `RUST_LOG`.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use tracing_subscriber::fmt::MakeWriter;
use vitaslop_platform::diag::{self, Channel};

/// Whether captured events are also written to stderr.
///
/// Not a startup constant: the shell applies a title's knobs when the title starts, so a
/// `VITASLOP_LOG` typed into the advanced settings box has not been set yet when the
/// subscriber is installed. [`follow_env`] is called again once those knobs are in the
/// environment. (The FILTER cannot be rebuilt that late - `EnvFilter` is fixed at install -
/// so a named filter changes what is MIRRORED, and only a filter set before launch changes
/// what is emitted. Capture is at `warn` either way, which is the level every report of
/// something wrong already uses.)
static MIRROR: AtomicBool = AtomicBool::new(true);

/// Point the stderr mirror at whether a human named a filter. Idempotent.
pub fn follow_env() {
    let named = std::env::var_os("VITASLOP_LOG").is_some_and(|v| !v.is_empty())
        || std::env::var_os("RUST_LOG").is_some_and(|v| !v.is_empty());
    MIRROR.store(named, Ordering::Relaxed);
}

fn mirroring() -> bool {
    MIRROR.load(Ordering::Relaxed)
}

/// Install the subscriber the RIGS get: today's behaviour, unchanged, plus capture.
///
/// `--game`, `--headless`, `import`, `list`, `serve` and `--cube` all land here. Their output
/// is read from a terminal by whoever started them, and several recorded repro commands
/// depend on it, so the mirror is unconditional.
pub fn init_verbose() {
    install(true, false);
}

/// Install the subscriber the SHELL gets: capture always, stderr only if asked.
///
/// # Status is forced ON here, and leaving it off was a real hole
/// `knobs::log_filter` emits the status directive only when a filter was NAMED - correct for
/// a rig, whose stderr should stay silent on a clean run. But the shell has a PLACE to put
/// status now, and with the directive absent the whole channel was filtered out before it
/// reached the ring: the Diagnostics view's Status section would have been permanently empty
/// and the game-data line would have vanished on its way there. A channel the default filter
/// switches off is a channel that does not exist - the same trade that put this material at
/// `warn` in the first place. The web front end forces it on for exactly this reason; so
/// does this one. Appended last so it wins in `EnvFilter`.
pub fn init_quiet() {
    install(false, true);
    follow_env();
}

fn install(always_mirror: bool, force_status: bool) {
    MIRROR.store(always_mirror, Ordering::Relaxed);
    let mut filter = vitaslop_platform::knobs::log_filter();
    if force_status && !filter.contains(vitaslop_platform::knobs::STATUS_TARGET_DIRECTIVE) {
        filter = format!("{filter},{}", vitaslop_platform::knobs::STATUS_TARGET_DIRECTIVE);
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(filter))
        .with_writer(MakeCaptureWriter)
        .try_init();
}

/// `text` without its terminal colour codes (`ESC [ ... <letter>`). The fmt layer colours its
/// lines for a terminal; the ring is READ in the Diagnostics view and saved as a report, where
/// the codes are noise around every timestamp and level.
fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// A line-buffered writer: each completed event line goes to its ring, and to stderr when
/// the mirror is on.
struct CaptureWriter {
    buf: Vec<u8>,
    channel: Channel,
    /// Whether this event is worth keeping at all. The perf windows and I/O traces are
    /// hundreds of lines a run and belong to whoever asked for them, not to a bounded ring
    /// that a person reads.
    keep: bool,
}

impl std::io::Write for CaptureWriter {
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
        if self.keep {
            let clean = strip_ansi(text);
            file_line(&clean);
            diag::push(self.channel, &clean);
        }
        if mirroring() {
            let mut err = std::io::stderr();
            let _ = writeln!(err, "{text}");
        }
        self.buf.clear();
        Ok(())
    }
}

impl Drop for CaptureWriter {
    /// The fmt layer drops the writer at the end of each event rather than flushing it, so
    /// this is where a line actually goes anywhere.
    fn drop(&mut self) {
        use std::io::Write;
        let _ = self.flush();
    }
}

struct MakeCaptureWriter;

impl<'a> MakeWriter<'a> for MakeCaptureWriter {
    type Writer = CaptureWriter;

    fn make_writer(&'a self) -> CaptureWriter {
        CaptureWriter { buf: Vec::new(), channel: Channel::Warning, keep: false }
    }

    /// The per-event writer, which is where the LEVEL and TARGET are knowable. `make_writer`
    /// above has no metadata, so a decision taken there would either capture every line or
    /// none.
    fn make_writer_for(&'a self, meta: &tracing::Metadata<'_>) -> CaptureWriter {
        let status = meta.target() == diag::STATUS_TARGET;
        CaptureWriter {
            buf: Vec::new(),
            channel: if status { Channel::Status } else { Channel::Warning },
            keep: status || *meta.level() <= tracing::Level::WARN,
        }
    }
}

// ------------------------------- the log file -------------------------------
//
// # A double-clicked app had nowhere to leave a trace
// The rings above live in memory and stderr is not connected to anything when the shell is
// started from Explorer, so a crash took every line of its own explanation with it. The shell
// now also appends what the rings capture to a file under `<home>/logs`, one per run, the
// newest [`KEEP_LOGS`] kept - and a panic, on any thread, is written there with its backtrace
// before anything else happens.

/// How many run logs are kept; the oldest beyond this are deleted when a new one opens.
pub const KEEP_LOGS: usize = 10;

static FILE: Mutex<Option<(PathBuf, std::fs::File)>> = Mutex::new(None);

/// The open run log, if this run has one.
pub fn log_path() -> Option<PathBuf> {
    FILE.lock().ok()?.as_ref().map(|(p, _)| p.clone())
}

/// One line into the run log, if one is open. Never fails the caller: a log that cannot be
/// written is not a reason to stop a game.
pub fn file_line(text: &str) {
    if let Ok(mut g) = FILE.lock()
        && let Some((_, f)) = g.as_mut()
    {
        let _ = writeln!(f, "{text}");
    }
}

/// `YYYYMMDD-HHMMSS`, UTC, from Unix milliseconds - names that sort in the order they were made.
pub fn stamp(ms: u64) -> String {
    let secs = ms / 1000;
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(mo <= 2);
    format!("{y:04}{mo:02}{d:02}-{h:02}{m:02}{s:02}")
}

/// Delete the oldest `vitaslop-*.log` files in `dir` so that `keep` remain, counting the one
/// about to be made. Other files there are left alone.
pub fn prune_logs(dir: &Path, keep: usize) {
    let mut logs: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("vitaslop-") && n.ends_with(".log")))
        .collect();
    logs.sort();
    let excess = (logs.len() + 1).saturating_sub(keep);
    for p in logs.into_iter().take(excess) {
        let _ = std::fs::remove_file(p);
    }
}

/// Open this run's log under `dir` (pruning old ones first) and write its header.
pub fn open_log_file(dir: &Path) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    prune_logs(dir, KEEP_LOGS);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    let mut path = dir.join(format!("vitaslop-{}.log", stamp(now)));
    // Two runs started in the same second (a relaunch after a crash) must not share a file.
    if path.exists() {
        path = dir.join(format!("vitaslop-{}-{}.log", stamp(now), std::process::id()));
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
    writeln!(f, "vitaslop {} - run log, started {} UTC, pid {}", env!("CARGO_PKG_VERSION"), stamp(now), std::process::id())?;
    *FILE.lock().unwrap_or_else(|e| e.into_inner()) = Some((path.clone(), f));
    Ok(path)
}

/// The last panic this run took, for the shell to show: a panic on a worker thread (the
/// loader, the guest's) does not end the window, and must not end up only in a file.
static PANICKED: Mutex<Option<String>> = Mutex::new(None);

/// The message of a panic not yet shown, taken.
pub fn take_panic() -> Option<String> {
    PANICKED.lock().ok()?.take()
}

/// Write every panic, with its backtrace, to the run log before the default handling. When
/// `interactive` (the shell) and the panic is on the main thread - the window is going down
/// with it - also say so in a message box, naming the log, since a double-clicked app has no
/// terminal to print to. A panic on any other thread is left for the shell to show in its
/// own window (see [`take_panic`]).
pub fn install_panic_hook(interactive: bool) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("unnamed").to_string();
        let what = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "(no message)".into());
        let at = info.location().map(|l| format!(" at {}:{}", l.file(), l.line())).unwrap_or_default();
        let summary = format!("panic on thread '{name}': {what}{at}");
        let backtrace = std::backtrace::Backtrace::force_capture();
        file_line(&format!("\n{summary}\n{backtrace}"));
        diag::push(Channel::Warning, &summary);
        let log = log_path().map_or_else(|| "(no log file)".to_string(), |p| p.display().to_string());
        if interactive && name == "main" {
            let _ = rfd::MessageDialog::new()
                .set_level(rfd::MessageLevel::Error)
                .set_title("vitaslop stopped")
                .set_description(format!("vitaslop hit an internal error and has to close.\n\n{summary}\n\nThe details are in the log:\n{log}"))
                .set_buttons(rfd::MessageButtons::Ok)
                .show();
        } else if let Ok(mut g) = PANICKED.lock() {
            *g = Some(format!("{summary}\nThe details are in the log: {log}"));
        }
        previous(info);
    }));
}

/// One text report of everything this run has said: what a person attaches to a bug report.
///
/// Both sections are named even when empty, because "no warnings" is an answer and a report
/// that simply omits the section leaves the reader unable to tell it from a truncated file.
pub fn snapshot() -> String {
    let (held, total, dropped) = diag::counts(Channel::Warning);
    let mut out = format!(
        "vitaslop diagnostics\nversion: {}\nwarnings: {held} distinct, {total} total, \
         {dropped} dropped\n\n== WARNINGS ==\n",
        env!("CARGO_PKG_VERSION"),
    );
    out.push_str(diag::report(Channel::Warning).as_deref().unwrap_or("(none)\n"));
    out.push_str("\n== STATUS ==\n");
    out.push_str(diag::report(Channel::Status).as_deref().unwrap_or("(none)\n"));
    if let Some(p) = log_path() {
        out.push_str(&format!("\nrun log: {}\n", p.display()));
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn colour_codes_are_stripped_and_text_kept() {
        let line = "\u{1b}[2m2026-10-03T20:42:04Z\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m gxm depth: ok";
        assert_eq!(super::strip_ansi(line), "2026-10-03T20:42:04Z  INFO gxm depth: ok");
        assert_eq!(super::strip_ansi("plain [brackets] stay"), "plain [brackets] stay");
    }

    #[test]
    fn stamps_sort_as_time_does() {
        assert_eq!(super::stamp(0), "19700101-000000");
        assert_eq!(super::stamp(1_791_035_486_827), "20261003-135126");
        assert!(super::stamp(1_791_035_486_827) < super::stamp(1_791_035_546_827));
    }

    #[test]
    fn pruning_keeps_the_newest_logs_and_nothing_else_is_touched() {
        let dir = std::env::temp_dir().join(format!("vitaslop-logprune-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..12 {
            std::fs::write(dir.join(format!("vitaslop-20261001-0000{i:02}.log")), "x").unwrap();
        }
        std::fs::write(dir.join("notes.txt"), "keep me").unwrap();
        super::prune_logs(&dir, 10);
        let mut left: Vec<String> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        left.sort();
        assert_eq!(left.len(), 10, "nine logs, room for the new one, and the other file: {left:?}");
        assert!(left.contains(&"notes.txt".to_string()));
        assert!(!left.contains(&"vitaslop-20261001-000002.log".to_string()), "the oldest went first");
        assert!(left.contains(&"vitaslop-20261001-000003.log".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
