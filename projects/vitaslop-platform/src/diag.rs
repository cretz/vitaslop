//! Shared diagnostics-presentation helpers.
//!
//! The browser mirrors WARN/ERROR events into a bounded panel on the page, because the
//! console does not exist on the device whose numbers are the ones that matter. Bounding a
//! panel means deciding what to drop, and that decision is pure string handling with a rule
//! worth testing - which is why it lives here, in a crate that BUILDS ON THE HOST, rather
//! than beside its caller in the wasm32-gated `vitaslop-web`. A test that cannot run is not
//! a test.

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

    /// A line that is ONLY a count still keys to something stable rather than to the empty
    /// string colliding with every other degenerate line.
    #[test]
    fn a_bare_count_is_not_an_empty_key() {
        assert_eq!(dedupe_key("count=7"), "");
        // Documented rather than asserted as desirable: a warning whose entire text is a
        // count has no subject to distinguish, so collapsing such lines is correct.
    }
}
