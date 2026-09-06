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
            diag::push(self.channel, text);
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
    out
}
