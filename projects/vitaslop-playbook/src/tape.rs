//! Tapes: the exact input of one run, as text.
//!
//! A run of a playbook records a tape; replaying the tape presses exactly the same input on
//! exactly the same input frames, with no Lua and no cues. On one build, in deterministic
//! mode, that reproduces the run byte for byte. The format is in the crate README.
use std::fmt::Write as _;

use vitaslop_runtime::recipe::BUTTON_NAMES;
use vitaslop_runtime::{CtrlFrame, TouchFrame};

/// The first line of every tape, and its version.
pub const MAGIC: &str = "vitaslop-tape 1";

/// The input in effect from `input` (an input frame) until the next change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Change {
    pub input: u64,
    /// The display frame that input frame landed on when this tape was recorded.
    pub display: u64,
    pub ctrl: CtrlFrame,
    pub touch: Option<(u16, u16)>,
}

/// A timeline line: a split, a phase boundary, a log line - informative, not replayed.
#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    pub input: u64,
    pub display: u64,
    pub seconds: f64,
    pub text: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tape {
    /// `key value` header lines, in order (`title`, `playbook`, `mode`, ...).
    pub header: Vec<(String, String)>,
    pub changes: Vec<Change>,
    pub notes: Vec<Note>,
    /// The last input frame recorded, and its display frame.
    pub end: Option<(u64, u64)>,
}

/// Button names joined by `+`, or `-` for none.
pub fn buttons_text(bits: u32) -> String {
    let names: Vec<&str> = BUTTON_NAMES.iter().filter(|(_, b)| bits & b != 0).map(|(n, _)| *n).collect();
    if names.is_empty() { "-".into() } else { names.join("+") }
}

/// Parse `buttons_text`'s output (also accepts the aliases `x`, `l`, `r`).
pub fn parse_buttons(s: &str) -> Result<u32, String> {
    if s == "-" {
        return Ok(0);
    }
    s.split('+').try_fold(0u32, |bits, n| Ok(bits | button_bit(n).ok_or_else(|| format!("unknown button {n:?}"))?))
}

/// One button's bit, by its canonical name or alias.
pub fn button_bit(name: &str) -> Option<u32> {
    let canonical = match name {
        "x" => "cross",
        "l" => "ltrigger",
        "r" => "rtrigger",
        other => other,
    };
    BUTTON_NAMES.iter().find(|(n, _)| *n == canonical).map(|(_, b)| *b)
}

impl Tape {
    pub fn header_value(&self, key: &str) -> Option<&str> {
        self.header.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// The input in effect at input frame `input`: the last change at or before it.
    pub fn input_at(&self, input: u64) -> (CtrlFrame, TouchFrame) {
        // Changes are in increasing input-frame order (`parse` refuses anything else).
        let i = self.changes.partition_point(|c| c.input <= input);
        match i.checked_sub(1).map(|i| &self.changes[i]) {
            Some(c) => (c.ctrl, c.touch.map(|(x, y)| TouchFrame::single(x, y)).unwrap_or_default()),
            None => (CtrlFrame::default(), TouchFrame::default()),
        }
    }

    /// The change recorded exactly at `input`, if any.
    pub fn change_at(&self, input: u64) -> Option<&Change> {
        let i = self.changes.partition_point(|c| c.input < input);
        self.changes.get(i).filter(|c| c.input == input)
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(MAGIC);
        s.push('\n');
        for (k, v) in &self.header {
            let _ = writeln!(s, "{k} {v}");
        }
        s.push_str("# <input frame> <display frame> <buttons> <lx> <ly> <rx> <ry> <touch>  - a line per CHANGE\n");
        // Changes and notes interleave in input-frame order, a note after the change it sits on.
        let mut notes = self.notes.iter().peekable();
        for c in &self.changes {
            while let Some(n) = notes.peek().filter(|n| n.input < c.input) {
                let _ = writeln!(s, "# {} f{} {:.3}s {}", n.input, n.display, n.seconds, n.text);
                notes.next();
            }
            let touch = match c.touch {
                Some((x, y)) => format!("{x},{y}"),
                None => "-".into(),
            };
            let _ = writeln!(
                s,
                "{} f{} {} {} {} {} {} {touch}",
                c.input,
                c.display,
                buttons_text(c.ctrl.buttons),
                c.ctrl.lx,
                c.ctrl.ly,
                c.ctrl.rx,
                c.ctrl.ry
            );
        }
        for n in notes {
            let _ = writeln!(s, "# {} f{} {:.3}s {}", n.input, n.display, n.seconds, n.text);
        }
        if let Some((input, display)) = self.end {
            let _ = writeln!(s, "end {input} f{display}");
        }
        s
    }

    pub fn parse(text: &str) -> Result<Tape, String> {
        let mut lines = text.lines().enumerate();
        match lines.next() {
            Some((_, l)) if l.trim() == MAGIC => {}
            other => return Err(format!("not a tape: the first line must be {MAGIC:?}, got {:?}", other.map(|(_, l)| l))),
        }
        let mut t = Tape::default();
        let mut in_body = false;
        for (i, raw) in lines {
            let line = raw.trim();
            let at = |e: String| format!("tape line {}: {e}", i + 1);
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("# ") {
                // A note is `# <input> f<display> <seconds>s <text>`; any other comment is skipped.
                let mut it = rest.splitn(4, ' ');
                let (Some(a), Some(b), Some(c), Some(text)) = (it.next(), it.next(), it.next(), it.next()) else { continue };
                if let (Ok(input), Some(Ok(display)), Some(Ok(seconds))) = (
                    a.parse::<u64>(),
                    b.strip_prefix('f').map(str::parse::<u64>),
                    c.strip_suffix('s').map(str::parse::<f64>),
                ) {
                    t.notes.push(Note { input, display, seconds, text: text.to_string() });
                }
                continue;
            }
            if line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            if f[0] == "end" {
                let (Some(input), Some(display)) = (f.get(1).and_then(|v| v.parse().ok()), f.get(2).and_then(|v| v.strip_prefix('f')).and_then(|v| v.parse().ok())) else {
                    return Err(at("`end <input frame> f<display frame>`".into()));
                };
                t.end = Some((input, display));
                continue;
            }
            if !in_body && !f[0].chars().all(|c| c.is_ascii_digit()) {
                let (k, v) = line.split_once(' ').unwrap_or((line, ""));
                t.header.push((k.to_string(), v.trim().to_string()));
                continue;
            }
            in_body = true;
            if f.len() != 8 {
                return Err(at(format!("a change has 8 fields, this has {}", f.len())));
            }
            let num = |s: &str| s.parse::<u8>().map_err(|_| at(format!("bad axis {s:?}")));
            let input = f[0].parse().map_err(|_| at(format!("bad input frame {:?}", f[0])))?;
            let display = f[1].strip_prefix('f').and_then(|v| v.parse().ok()).ok_or_else(|| at(format!("bad display frame {:?}", f[1])))?;
            let touch = if f[7] == "-" {
                None
            } else {
                let (x, y) = f[7].split_once(',').ok_or_else(|| at(format!("bad touch {:?}", f[7])))?;
                Some((x.parse().map_err(|_| at("bad touch x".into()))?, y.parse().map_err(|_| at("bad touch y".into()))?))
            };
            t.changes.push(Change {
                input,
                display,
                ctrl: CtrlFrame { buttons: parse_buttons(f[2]).map_err(at)?, lx: num(f[3])?, ly: num(f[4])?, rx: num(f[5])?, ry: num(f[6])? },
                touch,
            });
        }
        if t.changes.windows(2).any(|w| w[1].input <= w[0].input) {
            return Err("tape changes are not in increasing input-frame order".into());
        }
        Ok(t)
    }
}

/// Compare two tapes' timelines: every note text the two share, with how far it moved.
/// This is the cross-build regression read - "the kickoff split is 40 input frames later".
pub fn diff_timelines(a: &Tape, b: &Tape) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "{:<40} {:>12} {:>12} {:>8} {:>9}", "note", "a input/f", "b input/f", "d input", "d secs");
    let mut used = vec![false; b.notes.len()];
    for na in &a.notes {
        // The same text, matched in order, so a repeated split pairs with its own repeat.
        let hit = b.notes.iter().enumerate().find(|(i, nb)| !used[*i] && nb.text == na.text);
        match hit {
            Some((i, nb)) => {
                used[i] = true;
                let _ = writeln!(
                    s,
                    "{:<40} {:>12} {:>12} {:>+8} {:>+9.3}",
                    truncate(&na.text, 40),
                    format!("{}/f{}", na.input, na.display),
                    format!("{}/f{}", nb.input, nb.display),
                    nb.input as i64 - na.input as i64,
                    nb.seconds - na.seconds
                );
            }
            None => {
                let _ = writeln!(s, "{:<40} {:>12} {:>12}", truncate(&na.text, 40), format!("{}/f{}", na.input, na.display), "(missing)");
            }
        }
    }
    for (i, nb) in b.notes.iter().enumerate() {
        if !used[i] {
            let _ = writeln!(s, "{:<40} {:>12} {:>12}", truncate(&nb.text, 40), "(missing)", format!("{}/f{}", nb.input, nb.display));
        }
    }
    s
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { s.chars().take(n - 1).collect::<String>() + "~" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tape_round_trips() {
        let t = Tape {
            header: vec![("title".into(), "ABCD00001".into()), ("mode".into(), "deterministic".into())],
            changes: vec![
                Change { input: 0, display: 100, ctrl: CtrlFrame::default(), touch: None },
                Change { input: 4, display: 104, ctrl: CtrlFrame { buttons: 0x4008, lx: 0, ..CtrlFrame::default() }, touch: Some((10, 20)) },
            ],
            notes: vec![Note { input: 2, display: 102, seconds: 1.5, text: "split title".into() }],
            end: Some((9, 109)),
        };
        let text = t.render();
        assert_eq!(Tape::parse(&text).unwrap(), t, "{text}");
        assert_eq!(t.input_at(3).0, CtrlFrame::default());
        assert_eq!(t.input_at(7).0.buttons, 0x4008);
    }

    #[test]
    fn a_timeline_diff_reports_how_far_each_note_moved() {
        let n = |input, text: &str| Note { input, display: input + 100, seconds: input as f64 / 30.0, text: text.into() };
        let a = Tape { notes: vec![n(10, "split kickoff")], ..Tape::default() };
        let b = Tape { notes: vec![n(50, "split kickoff")], ..Tape::default() };
        assert!(diff_timelines(&a, &b).contains("+40"));
    }
}
