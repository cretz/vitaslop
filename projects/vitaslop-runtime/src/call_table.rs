//! `VITASLOP_CALL_TABLE=<from>-<to>` (display frames, inclusive): every DISTINCT host call the
//! guest makes in that window - keyed by (function, thread, return address) - with how many
//! times it was made, the frames it was first and last seen, and its first argument. Printed as
//! the diagnostics panel's CALL TABLE, in first-seen order.
//!
//! # Why a table and not a log
//! A thread that is stuck is usually stuck in a LOOP: MEASURED on one title's intro movie, two
//! threads made ~490,000 condition-variable calls in 650 frames between them, which drowns any
//! per-call log and cannot be forwarded off a phone at all. What a hang question needs is the
//! SHAPE: which calls each thread makes, in what order they first appeared, which of them are
//! still happening at the end (last-seen frame) and which stopped. Deduplicated, a whole boot
//! is a few hundred rows - small enough for the phone runner's result - and two runs (a device
//! that hangs, a desktop that does not) can be diffed row by row.
//!
//! Calls the transpiler inlines (lightweight mutexes, some fast paths) never reach dispatch and
//! are not here. Off unless the knob is set; the cost when set is one map lookup per call.

use std::collections::HashMap;
use std::sync::Mutex;

/// Distinct rows kept. Past it new keys are counted in `dropped`, never recorded.
pub const CAP: usize = 4000;

struct Row {
    seq: u64,
    first_frame: u64,
    last_frame: u64,
    count: u64,
    arg0: u32,
    arg0_varies: bool,
    /// r0..r12 at the FIRST occurrence, printed under `VITASLOP_CALL_TABLE_REGS=1` - the
    /// caller's object pointers, which is what turns "this thread took the other branch" into
    /// an address a store watchpoint can be put on.
    regs: [u32; 13],
}

#[derive(Default)]
struct Table {
    rows: HashMap<(u32, i32, u32), Row>,
    seq: u64,
    dropped: u64,
}

static TABLE: Mutex<Option<Table>> = Mutex::new(None);
static SPEC: std::sync::OnceLock<Option<(u64, u64)>> = std::sync::OnceLock::new();

/// The frame window, or `None` when the table is off.
pub fn spec() -> Option<(u64, u64)> {
    *SPEC.get_or_init(|| {
        let raw = crate::knobs::var("VITASLOP_CALL_TABLE").ok()?;
        let raw = raw.trim();
        if raw.is_empty() || raw == "0" {
            return None;
        }
        match raw.split_once('-') {
            Some((a, b)) => Some((a.trim().parse().ok()?, b.trim().parse().ok()?)),
            None => Some((raw.parse().ok()?, u64::MAX)),
        }
    })
}

/// Record one call. `lr` is the guest's return address, which is what tells two call sites
/// of the same function on the same thread apart.
pub fn record(frame: u64, func_nid: u32, thid: i32, lr: u32, regs: &[u32]) {
    let arg0 = regs[0];
    let Some((from, to)) = spec() else { return };
    if frame < from || frame > to {
        return;
    }
    let Ok(mut g) = TABLE.lock() else { return };
    let t = g.get_or_insert_with(Table::default);
    let key = (func_nid, thid, lr);
    if let Some(r) = t.rows.get_mut(&key) {
        r.count += 1;
        r.last_frame = frame;
        if r.arg0 != arg0 {
            r.arg0_varies = true;
        }
        return;
    }
    if t.rows.len() >= CAP {
        t.dropped += 1;
        return;
    }
    t.seq += 1;
    let seq = t.seq;
    let mut first = [0u32; 13];
    for (d, s) in first.iter_mut().zip(regs) {
        *d = *s;
    }
    t.rows.insert(key, Row { seq, first_frame: frame, last_frame: frame, count: 1, arg0, arg0_varies: false, regs: first });
}

/// `[trace]` hits (block tracer, store watch log) kept for the diagnostics panel, so they can
/// be read off a device whose worker consoles are not forwarded. Capped; the first ones kept.
pub const TRACE_HITS_CAP: usize = 600;
static TRACE_HITS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static TRACE_HITS_DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn note_trace_hit(line: &str) {
    if let Ok(mut v) = TRACE_HITS.lock()
        && v.len() < TRACE_HITS_CAP {
            // Runs of padding in the register dump carry nothing.
            v.push(line.split_whitespace().collect::<Vec<_>>().join(" "));
            return;
        }
    TRACE_HITS_DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// The kept trace hits as panel text, or empty when there are none.
pub fn trace_hits_report() -> String {
    let Ok(v) = TRACE_HITS.lock() else { return String::new() };
    if v.is_empty() {
        return String::new();
    }
    let mut s = format!(
        "{} trace hit(s) (cap {TRACE_HITS_CAP}; {} more not kept), in order:\n",
        v.len(),
        TRACE_HITS_DROPPED.load(std::sync::atomic::Ordering::Relaxed)
    );
    for l in v.iter() {
        s.push_str(l);
        s.push('\n');
    }
    s
}

/// The table as panel text, first-seen order, or empty when nothing was recorded.
pub fn report() -> String {
    let Ok(g) = TABLE.lock() else { return String::new() };
    let Some(t) = g.as_ref() else { return String::new() };
    let mut rows: Vec<_> = t.rows.iter().collect();
    rows.sort_by_key(|(_, r)| r.seq);
    let mut s = format!(
        "{} distinct (call, thread, return address) rows in the window (cap {CAP}; {} calls past it \
         not recorded). `fA-B xN` = first/last frame and count; `a0` = first argument (`*` = it \
         varied). A row whose last frame is the end of the run is a call still being made - a loop.\n",
        rows.len(),
        t.dropped
    );
    let regs_on = crate::knobs::flag("VITASLOP_CALL_TABLE_REGS");
    for ((nid, thid, lr), r) in rows {
        s.push_str(&format!(
            "f{}-{} x{} t{:#x} {} lr={:#010x} a0={:#x}{}",
            r.first_frame,
            r.last_frame,
            r.count,
            thid,
            crate::nid::name(*nid),
            lr,
            r.arg0,
            if r.arg0_varies { "*" } else { "" }
        ));
        if regs_on {
            s.push_str(" regs=");
            for (i, v) in r.regs.iter().enumerate() {
                s.push_str(&format!("{}{v:x}", if i > 0 { "," } else { "" }));
            }
        }
        s.push('\n');
    }
    s
}
