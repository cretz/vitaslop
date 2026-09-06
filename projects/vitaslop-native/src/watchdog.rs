//! A wall-clock stall watchdog for a run that is not driven interactively.
//!
//! # Why this exists
//! A headless run's loop is `while frames < target { guest.advance() }`. Every stop condition
//! it has is a condition the GUEST reaches: a trap, a thread exiting, a round budget running
//! out, the frame target being met. A guest that SPINS reaches none of them - `advance()` never
//! returns - so the run does not fail, it simply never ends, and every report the runner writes
//! at the end (the block histogram, the call-site tally, the scheduler report) is written after
//! a line that is never executed. The one case those instruments are built for, "the title got
//! to screen N and stopped", is the one case they could not be read in.
//!
//! The scheduler's own deadlock report does not fire either, and correctly so: a deadlock is
//! every thread parked with nothing to wake them, and a spin is the opposite - a thread that is
//! perfectly runnable and getting exactly what it asks for, forever.
//!
//! So the observer has to be OUTSIDE the guest and keyed on WALL CLOCK. This thread watches
//! [`crate::threaded::current_frame`], and when it has not moved for `VITASLOP_STALL_WATCHDOG`
//! seconds it prints what is knowable from outside and stops the process.
//!
//! # What it prints, and why it is a window and not a total
//! The useful reading is not "which NIDs has this run called" - on a hang that is mostly the
//! boot that got there. It is "which NIDs is the guest calling WHILE IT IS STUCK". Nothing
//! inside the run can close such a window, because closing it is the frame that never comes;
//! two snapshots a few seconds apart, taken from out here, can.
//!
//! An empty window is itself a reading and is reported as one: a spin that makes no host calls
//! is guest code with no NID in it, which the call-site tally is structurally unable to see and
//! `VITASLOP_TRACE_BLOCKS` + `VITASLOP_BLOCK_HIST` is the instrument for.
//!
//! # What it costs
//! Nothing when unset, and one thread polling an atomic twice a second when set. It is
//! deliberately not on by default: a legitimately slow frame (a shader-compiling first frame, a
//! streamed load) is indistinguishable from a spin without a threshold somebody chose.
//!
//! # The one thing it gives up
//! It ends the process with [`std::process::exit`], so nothing the runner would write on the
//! way out is written - a hung run has no clean exit to take, and the alternative (asking the
//! main loop to notice) is exactly the thing that cannot happen while `advance()` is inside the
//! guest. Everything the watchdog needs is therefore read from process-wide state.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Exit status for a watchdog stop. Distinct from 1 (a guest error, which the runner reports
/// itself) so a script can tell "this title is broken" from "this title never came back".
pub const STALL_EXIT_CODE: i32 = 3;

/// How long the watchdog waits between its two call-site snapshots. Long enough that a slow
/// loop still registers, short enough that a stopped run is not held open.
const SAMPLE_SECS: u64 = 5;

/// How many rows of each ranking to print.
const TOP: usize = 25;

/// Set once a stall has been declared, so two watchdogs cannot interleave their output.
static FIRED: AtomicU64 = AtomicU64::new(0);

/// A way to ask the running scheduler for its whole-machine sync dump from OUTSIDE the run.
///
/// This is the one reading the call-site window cannot give: the window says what the guest is
/// CALLING, and on a title whose sound thread free-runs that is entirely sound calls, which
/// says nothing about the thread that stopped producing frames. The dump says, for every live
/// thread, whether it is parked and on what - so "every game thread is waiting on a condition
/// variable nobody signals" and "one thread is spinning in guest code" are distinguishable,
/// and they are opposite bugs.
///
/// It is a registered closure rather than a reachable object because the host is owned by the
/// scheduler, the scheduler is owned by the main thread, and the main thread is inside the
/// guest - which is the whole situation. The closure holds its own handle and takes the lock
/// itself.
type SyncDump = Box<dyn Fn() -> Result<String, &'static str> + Send + Sync>;
static SYNC_DUMP: std::sync::OnceLock<SyncDump> = std::sync::OnceLock::new();

/// Register the sync-dump reader. Called by the scheduler as it is built; the first
/// registration wins, so a process that stands up two guests reports the first.
pub fn register_sync_dump(f: SyncDump) {
    let _ = SYNC_DUMP.set(f);
}

/// The scheduler's sync dump, retried for a couple of seconds because the host lock is held
/// for the duration of every host call and a free-running thread takes it constantly. A
/// non-blocking read is what keeps the watchdog from becoming the second thing that hangs.
fn sync_dump() -> Result<String, String> {
    let Some(f) = SYNC_DUMP.get() else {
        return Err("no scheduler registered a sync dump for this run".into());
    };
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut last = "not attempted";
    while Instant::now() < deadline {
        match f() {
            Ok(s) => return Ok(s),
            Err(e) => last = e,
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Err(format!("could not read the scheduler state in 2 s: {last}"))
}

/// The configured stall budget in seconds, from `VITASLOP_STALL_WATCHDOG`.
///
/// Errors are refused rather than defaulted: a run left overnight under a watchdog that
/// silently parsed to "off" is the failure this whole file exists to prevent.
pub fn budget_secs() -> Result<Option<u64>, String> {
    parse_budget(std::env::var("VITASLOP_STALL_WATCHDOG").ok().as_deref())
}

/// The parse, split out so it can be tested without mutating the process environment under a
/// parallel test run.
fn parse_budget(v: Option<&str>) -> Result<Option<u64>, String> {
    match v {
        None => Ok(None),
        Some(s) if s.trim().is_empty() => Ok(None),
        Some(s) => match s.trim().parse::<u64>() {
            Ok(0) => Ok(None),
            Ok(n) => Ok(Some(n)),
            Err(_) => Err(format!(
                "VITASLOP_STALL_WATCHDOG={s:?} is not a whole number of seconds \
                 (no unit suffix - write 120, not 120s)"
            )),
        },
    }
}

/// Start the watchdog if `VITASLOP_STALL_WATCHDOG` asks for one. Call once, before the run
/// loop. Returns whether one was started, so the caller can say so in its own preamble - a
/// watchdog nobody can tell is armed is one nobody relies on.
pub fn spawn(what: &str) -> Result<bool, String> {
    let Some(secs) = budget_secs()? else {
        return Ok(false);
    };
    let what = what.to_string();
    std::thread::Builder::new()
        .name("stall-watchdog".into())
        .spawn(move || watch(secs, &what))
        .map_err(|e| format!("spawn stall watchdog: {e}"))?;
    Ok(true)
}

fn watch(secs: u64, what: &str) {
    let budget = Duration::from_secs(secs);
    let mut last_frame = crate::threaded::current_frame();
    let mut last_move = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(500));
        let f = crate::threaded::current_frame();
        if f != last_frame {
            last_frame = f;
            last_move = Instant::now();
            continue;
        }
        let stuck = last_move.elapsed();
        if stuck < budget {
            continue;
        }
        if FIRED.swap(1, Ordering::SeqCst) != 0 {
            return;
        }
        fire(what, f, stuck);
    }
}

fn fire(what: &str, frame: u64, stuck: Duration) -> ! {
    eprintln!(
        "\n=== STALL WATCHDOG: no display frame for {:.1}s. The last flip was frame {frame}, \
         so frame {} started and never finished. Run: {what} ===",
        stuck.as_secs_f64(),
        frame + 1
    );
    // The window. Taken here rather than from boot, because a hang's cumulative tally is
    // dominated by the thousands of frames that worked.
    let profiling = vitaslop_runtime::vita::callsite_profiling_on();
    let before = profiling.then(vitaslop_runtime::vita::call_sites_snapshot);
    if profiling {
        eprintln!("watchdog: sampling host calls for {SAMPLE_SECS}s while it is stuck...");
    } else {
        eprintln!(
            "watchdog: VITASLOP_DBG_CALLSITES was not set, so no host calls were recorded and \
             the spin cannot be named by NID. Re-run with VITASLOP_DBG_CALLSITES=1."
        );
    }
    std::thread::sleep(Duration::from_secs(SAMPLE_SECS));
    if let Some(before) = before {
        match vitaslop_runtime::vita::call_sites_delta_report(&before, TOP) {
            Some(rep) => {
                eprintln!("watchdog: what the guest called during {SAMPLE_SECS}s of the stall:");
                eprint!("{rep}");
            }
            None => eprintln!(
                "watchdog: the guest made ZERO host calls in {SAMPLE_SECS}s of the stall. The \
                 spin is guest code with no NID in it, so the call-site tally cannot see it - \
                 VITASLOP_TRACE_BLOCKS=<lo>-<hi> plus VITASLOP_BLOCK_HIST=<top> is what names \
                 that, and this watchdog dumps it below."
            ),
        }
    }
    // Every live thread and what it is parked on. This is the half that names the STOPPED
    // thread, which the call-site window structurally cannot: a thread that is blocked makes
    // no calls, so it appears in the window as an absence.
    match sync_dump() {
        Ok(d) => eprintln!("watchdog: the scheduler's view of the stall:
{d}"),
        Err(e) => eprintln!("watchdog: NO scheduler state - {e}"),
    }
    // The block histogram, if the run was transpiled with the hooks. Silent when it was not.
    crate::dump_block_hist(TOP);
    eprintln!("=== STALL WATCHDOG: stopping the process (exit {STALL_EXIT_CODE}) ===");
    // Nothing the run would write on the way out gets written - see the module note.
    std::process::exit(STALL_EXIT_CODE)
}

#[cfg(test)]
mod tests {
    use super::parse_budget;

    /// A mis-typed budget must not read as "off". This is the whole safety property of the
    /// knob: an overnight run under a watchdog that silently disarmed is the hang it was
    /// supposed to catch, discovered a night later.
    #[test]
    fn a_bad_budget_is_refused_rather_than_treated_as_off() {
        assert!(parse_budget(Some("120s")).is_err(), "a unit suffix must not read as off");
        assert!(parse_budget(Some("yes")).is_err());
        assert!(parse_budget(Some("-1")).is_err());
        assert_eq!(parse_budget(Some("120")).unwrap(), Some(120));
        // The two spellings of "no watchdog": unset, and explicitly zero. Both are silent.
        assert_eq!(parse_budget(None).unwrap(), None);
        assert_eq!(parse_budget(Some("0")).unwrap(), None);
        assert_eq!(parse_budget(Some("")).unwrap(), None);
    }
}
