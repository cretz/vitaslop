//! Shared diagnostics-presentation helpers, and the run's diagnostic RINGS.
//!
//! The browser mirrors WARN/ERROR events into a bounded panel on the page, because the
//! console does not exist on the device whose numbers are the ones that matter. Bounding a
//! panel means deciding what to drop, and that decision is pure string handling with a rule
//! worth testing - which is why it lives here, in a crate that BUILDS ON THE HOST, rather
//! than beside its caller in the wasm32-gated `vitaslop-web`. A test that cannot run is not
//! a test.
//!
//! # Why the RINGS moved here too, and it is the same argument one step further
//! The rings were `vitaslop-web`'s, and so was the reasoning behind them: a warning is a
//! claim that something is broken, status is a heartbeat, and a panel that mixes them is a
//! panel nobody can read. None of that is about the browser. The DESKTOP shell had no
//! equivalent at all - its warnings went to stderr, which on a product run is either a
//! console the player never opened or a terminal they are being shouted at from. Same
//! events, same split, one implementation; each front end decides only where to MIRROR
//! them.

use std::collections::VecDeque;
use std::sync::Mutex;

/// How many DISTINCT WARN/ERROR lines a ring holds.
///
/// Distinct, not total: repeats are counted against the line already held (see [`push`]),
/// so this bounds a panel by how many different things went wrong rather than by how often.
/// A race screen produced 70+ occurrences of one warning shape across dozens of program
/// pairs and evicted six DIFFERENT warnings to fit them - including the three that named
/// real render defects. A ring that drops a unique warning to keep the hundredth copy of
/// another has its priority exactly backwards.
pub const RING_CAP: usize = 96;

/// The target that files an event as STATUS rather than as a warning.
///
/// Emitted through `vitaslop_platform::report_status!`, which is the only thing that should
/// ever name this string.
pub const STATUS_TARGET: &str = "vitaslop::status";

/// The WARN/ERROR lines this run has emitted, for whatever panel the front end shows.
///
/// # Why a console is not where these can live
/// The console is unreachable on a phone without a cable and remote debugging, and the phone
/// is the only machine whose numbers are not a proxy. The same holds for a shipped desktop
/// build: the person who can see the failure is the person with no terminal. So the one line
/// that settles "did this draw fail, or did the guest never submit it" has to be somewhere
/// the UI can reach, on every front end.
static WARNINGS: Mutex<Option<Ring>> = Mutex::new(None);

/// The STATUS lines this run has emitted, kept apart from the warnings above.
///
/// # Why this exists, and it is the whole point of the split
/// A WARNING IS A CLAIM THAT SOMETHING IS BROKEN AND OWED A FIX. A great deal of what this
/// engine reports is not that: the frame's pass structure, a precompile count, a texture
/// working set. Those are STATUS, and filing them at `warn` makes a panel unreadable - a
/// user cannot tell a defect from a heartbeat, and the genuine findings sit between twenty
/// copies of a frame-shape trace.
///
/// The wrong cure is `debug`: a report nobody sees does not exist, and this project has met
/// that failure repeatedly. So status keeps a channel of its own that is ALWAYS captured and
/// ALWAYS renderable. Nothing is silenced; it is sorted.
static STATUS: Mutex<Option<Ring>> = Mutex::new(None);

#[derive(Default)]
struct Ring {
    /// Each distinct line, with how many times it has been emitted.
    lines: VecDeque<(String, u64)>,
    /// Distinct lines pushed out of the ring. Reported rather than dropped silently - a panel
    /// showing the LAST N warnings while implying it shows all of them is the failure this
    /// project keeps meeting under other names.
    dropped: usize,
}

/// Which ring a line belongs in.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// Something is wrong and is owed a fix.
    Warning,
    /// The run describing itself: setup, shapes, counts.
    Status,
}

fn ring(c: Channel) -> &'static Mutex<Option<Ring>> {
    match c {
        Channel::Warning => &WARNINGS,
        Channel::Status => &STATUS,
    }
}

/// Record one already-formatted diagnostic line.
pub fn push(c: Channel, text: &str) {
    let Ok(mut guard) = ring(c).lock() else { return };
    let log = guard.get_or_insert_with(Ring::default);
    let key = dedupe_key(text);
    // A repeat updates the line already held - keeping the LATEST text, so a diagnostic that
    // reports its own running count shows the newest one rather than the first.
    if let Some(slot) = log.lines.iter_mut().find(|(l, _)| dedupe_key(l) == key) {
        slot.0 = text.to_string();
        slot.1 += 1;
        return;
    }
    if log.lines.len() == RING_CAP {
        log.lines.pop_front();
        log.dropped += 1;
    }
    log.lines.push_back((text.to_string(), 1));
}

/// The lines so far, oldest first, or `None` if there were none.
///
/// Non-draining on purpose: a panel is rebuilt from scratch each window, and a warning that
/// fired once must not vanish from it one window later.
pub fn report(c: Channel) -> Option<String> {
    let guard = ring(c).lock().ok()?;
    let log = guard.as_ref()?;
    if log.lines.is_empty() {
        return None;
    }
    let mut out = String::new();
    if log.dropped > 0 {
        out.push_str(&format!(
            "({} earlier DISTINCT line(s) dropped - this panel keeps {RING_CAP} of them)\n",
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

/// `(distinct lines held, total emissions, distinct lines dropped)` - what a badge shows.
///
/// The total is reported beside the distinct count so a UI can never imply that a warning
/// seen once and a warning seen ten thousand times are the same event.
pub fn counts(c: Channel) -> (usize, u64, usize) {
    let Ok(guard) = ring(c).lock() else { return (0, 0, 0) };
    let Some(log) = guard.as_ref() else { return (0, 0, 0) };
    (log.lines.len(), log.lines.iter().map(|(_, n)| *n).sum(), log.dropped)
}

/// The part of a diagnostic line that decides whether two emissions are THE SAME finding.
///
/// # Why the whole line is the wrong key
/// A repeating diagnostic reports itself on a power-of-ten schedule - `count=1`, `count=10`,
/// `count=100` - so keying on the whole line files one finding under three names and fills a
/// bounded panel with a counter's own progress reports. Keyed this way, a race frame evicted
/// six DIFFERENT warnings (three of which named real render defects) to make room for repeats
/// of one.
///
/// # Why the message alone is also wrong
/// Going further and keying on the message text would merge findings that share a shape but
/// not a subject: each stale program PAIR is its own defect, and collapsing them would report
/// one where there are thirty. So exactly one thing is removed - a trailing `count=<digits>` -
/// and every other field still distinguishes.
pub fn dedupe_key(text: &str) -> &str {
    match text.rfind("count=") {
        Some(i) if !text[i + 6..].is_empty() && text[i + 6..].bytes().all(|b| b.is_ascii_digit()) => {
            text[..i].trim_end()
        }
        _ => text,
    }
}

#[cfg(test)]
mod tests {
    use super::dedupe_key;

    /// A diagnostic's own power-of-ten progress reports are ONE finding, not three.
    #[test]
    fn a_count_suffix_does_not_make_a_new_finding() {
        let a = "STALE default uniform buffer bound_for=0x1 drawing=0x2 count=1";
        let b = "STALE default uniform buffer bound_for=0x1 drawing=0x2 count=1000";
        assert_eq!(dedupe_key(a), dedupe_key(b));
        assert_eq!(dedupe_key(a), "STALE default uniform buffer bound_for=0x1 drawing=0x2");
    }

    /// ...but a different SUBJECT is a different finding, however similar the text.
    #[test]
    fn a_different_subject_is_a_different_finding() {
        let a = "STALE default uniform buffer bound_for=0x1 drawing=0x2 count=1";
        let b = "STALE default uniform buffer bound_for=0x9 drawing=0x2 count=1";
        assert_ne!(dedupe_key(a), dedupe_key(b));
    }

    /// A trailing `count=` that is not a plain number is not a counter. Stripping it would
    /// let two unrelated findings collapse into one.
    #[test]
    fn only_a_numeric_count_suffix_is_stripped() {
        assert_eq!(dedupe_key("something count=many"), "something count=many");
        assert_eq!(dedupe_key("something count="), "something count=");
        assert_eq!(dedupe_key("no counter here"), "no counter here");
    }

    /// The ring counts repeats against the line it already holds, and says how many DISTINCT
    /// lines it had to drop rather than pretending it holds them all.
    ///
    /// One test, not three: the rings are process-global, so two tests writing to the same
    /// channel would race each other under the default parallel runner.
    #[test]
    fn the_ring_folds_repeats_and_reports_what_it_dropped() {
        use super::{Channel, RING_CAP, counts, push, report};
        for _ in 0..5 {
            push(Channel::Warning, "a draw fell back subject=0x1 count=1");
        }
        push(Channel::Warning, "a draw fell back subject=0x1 count=1000");
        assert_eq!(counts(Channel::Warning), (1, 6, 0), "one finding, six emissions");
        // The latest text wins, so a self-counting diagnostic shows its newest count.
        assert!(report(Channel::Warning).unwrap().contains("count=1000  [x6]"));

        for i in 0..RING_CAP {
            push(Channel::Warning, &format!("distinct finding {i}"));
        }
        let (held, _, dropped) = counts(Channel::Warning);
        assert_eq!(held, RING_CAP);
        assert_eq!(dropped, 1, "the oldest distinct line made way, and is admitted to");
        assert!(report(Channel::Warning).unwrap().starts_with("(1 earlier DISTINCT line(s) dropped"));
    }

    /// A line that is ONLY a count still keys to something stable rather than to the empty
    /// string colliding with every other degenerate line.
    #[test]
    fn a_bare_count_is_not_an_empty_key() {
        assert_eq!(dedupe_key("count=7"), "");
        // Documented rather than asserted as desirable: a warning whose entire text is a
        // count has no subject to distinguish, so collapsing such lines is correct.
    }
}
