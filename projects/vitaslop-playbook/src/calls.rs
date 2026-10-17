//! Counting the host calls a playbook's call cues name, from whichever guest thread makes them.
//!
//! >>> THIS RUNS ON EVERY HOST CALL, ON EVERY GUEST THREAD, so it takes no lock. A call cue
//! owns a SLOT: an append-only array of atomics read without locking, scanned per call (a
//! playbook declares a handful). A slot learns its NID the first time a call with its NAME is
//! made - the runtime's NID table names NIDs but cannot be searched by name - and from then on
//! a call costs one integer compare per slot.
//!
//! Also here: the DISCOVERY recorder (`--discover`), which is how cues get found in the first
//! place. It does lock, and it is a tool mode.
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use vitaslop_runtime::observer::ObservedCall;
use vitaslop_runtime::recipe_eval::GuestRead;

pub const MAX_SLOTS: usize = 64;

/// What a slot counts. Immutable once published.
pub struct SlotMeta {
    pub name: String,
    pub filter: Option<(u8, String)>,
}

struct Slot {
    /// 0 until the first call with the slot's name resolves it.
    nid: AtomicU32,
    count: AtomicU64,
    meta: AtomicPtr<SlotMeta>,
}

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY: Slot = Slot { nid: AtomicU32::new(0), count: AtomicU64::new(0), meta: AtomicPtr::new(std::ptr::null_mut()) };
static SLOTS: [Slot; MAX_SLOTS] = [EMPTY; MAX_SLOTS];
static USED: AtomicUsize = AtomicUsize::new(0);
static UNRESOLVED: AtomicUsize = AtomicUsize::new(0);

/// Forget every slot - the start of a run. Only called while no run is delivering calls.
pub fn reset() {
    USED.store(0, Ordering::Release);
    UNRESOLVED.store(0, Ordering::Release);
}

/// Claim a slot for a call cue. The metadata is leaked: a run declares a few dozen at most,
/// and a slot read on another thread must never see it freed.
pub fn add(name: &str, filter: Option<(u8, String)>) -> Result<usize, String> {
    let i = USED.load(Ordering::Acquire);
    if i >= MAX_SLOTS {
        return Err(format!("a playbook can declare at most {MAX_SLOTS} call cues"));
    }
    let meta = Box::into_raw(Box::new(SlotMeta { name: name.to_string(), filter }));
    let s = &SLOTS[i];
    s.nid.store(0, Ordering::Relaxed);
    s.count.store(0, Ordering::Relaxed);
    s.meta.store(meta, Ordering::Relaxed);
    UNRESOLVED.fetch_add(1, Ordering::AcqRel);
    // Published last: a reader that sees the new length sees the slot's fields.
    USED.store(i + 1, Ordering::Release);
    Ok(i)
}

fn meta(i: usize) -> &'static SlotMeta {
    // SAFETY: set before `USED` covered the slot and never freed (see `add`).
    unsafe { &*SLOTS[i].meta.load(Ordering::Acquire) }
}

pub fn count(i: usize) -> u64 {
    SLOTS[i].count.load(Ordering::Acquire)
}

/// Whether slot `i` ever matched a call's name.
pub fn resolved(i: usize) -> bool {
    SLOTS[i].nid.load(Ordering::Acquire) != 0
}

pub fn slot_name(i: usize) -> &'static str {
    &meta(i).name
}

/// A NUL-terminated string at `addr`, if it is printable ASCII of 1..=`max` bytes.
pub fn c_string(mem: &dyn GuestRead, addr: u32, max: usize) -> Option<String> {
    if addr < 0x1000 {
        return None;
    }
    let mut out = Vec::new();
    let mut chunk = [0u8; 32];
    while out.len() < max {
        if !mem.read_into(addr.wrapping_add(out.len() as u32), &mut chunk) {
            // The tail of mapped memory: retry a byte at a time.
            let mut b = [0u8; 1];
            if !mem.read_into(addr.wrapping_add(out.len() as u32), &mut b) {
                return None;
            }
            chunk[0] = b[0];
            if b[0] == 0 {
                break;
            }
            out.push(b[0]);
            continue;
        }
        match chunk.iter().position(|&b| b == 0) {
            Some(end) => {
                out.extend_from_slice(&chunk[..end]);
                break;
            }
            None => out.extend_from_slice(&chunk),
        }
    }
    if out.is_empty() || out.len() > max || !out.iter().all(|&b| (0x20..0x7f).contains(&b)) {
        return None;
    }
    String::from_utf8(out).ok()
}

/// Count `call` against every slot it matches.
pub fn note(call: &ObservedCall<'_>) {
    let used = USED.load(Ordering::Acquire);
    if used == 0 {
        return;
    }
    let mut name: Option<&'static str> = None;
    for (i, s) in SLOTS.iter().enumerate().take(used) {
        let nid = s.nid.load(Ordering::Relaxed);
        if nid == 0 {
            let n = *name.get_or_insert_with(|| vitaslop_runtime::nid::name(call.nid));
            if n != meta(i).name {
                continue;
            }
            if s.nid.compare_exchange(0, call.nid, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                UNRESOLVED.fetch_sub(1, Ordering::AcqRel);
            }
        } else if nid != call.nid {
            continue;
        }
        if let Some((arg, has)) = &meta(i).filter {
            let reg = call.regs.get(*arg as usize).copied().unwrap_or(0);
            match c_string(call.mem, reg, 512) {
                Some(text) if text.contains(has.as_str()) => {}
                _ => continue,
            }
        }
        s.count.fetch_add(1, Ordering::AcqRel);
    }
}

/// The discovery recorder: every NID the game calls, when it first and last called it, and
/// every distinct string argument it passed, with the display frame it first appeared on.
#[derive(Default)]
pub struct Discovery {
    calls: HashMap<u32, (u64, u64, u64)>,
    strings: BTreeMap<(u64, u32, u8), String>,
    seen: std::collections::HashSet<(u32, u8, String)>,
}

/// More distinct strings than this and the recorder stops adding (a title that formats a new
/// string every frame would otherwise grow without bound).
const MAX_STRINGS: usize = 4000;

pub static DISCOVERY: Mutex<Option<Discovery>> = Mutex::new(None);

pub fn discover(call: &ObservedCall<'_>) {
    let Ok(mut d) = DISCOVERY.lock() else { return };
    let Some(d) = d.as_mut() else { return };
    let e = d.calls.entry(call.nid).or_insert((call.frame, call.frame, 0));
    e.1 = call.frame;
    e.2 += 1;
    if d.strings.len() >= MAX_STRINGS {
        return;
    }
    for (r, &reg) in call.regs.iter().take(4).enumerate() {
        if let Some(text) = c_string(call.mem, reg, 160).filter(|t| looks_like_text(t)) {
            let key = (call.nid, r as u8, text);
            if !d.seen.contains(&key) {
                d.strings.insert((call.frame, call.nid, r as u8), key.2.clone());
                d.seen.insert(key);
            }
        }
    }
}

/// Whether a printable byte run is worth reporting as a string argument. Decoded video and
/// texture rows are printable ASCII more often than not; a name or a path is mostly letters,
/// digits and path punctuation.
fn looks_like_text(t: &str) -> bool {
    let wordy = t.chars().filter(|c| c.is_ascii_alphanumeric() || "/\\._-: ".contains(*c)).count();
    t.len() >= 4 && wordy * 10 >= t.len() * 9 && t.chars().any(|c| c.is_ascii_alphabetic())
}

impl Discovery {
    /// The report: calls by the display frame they were FIRST made on (a new screen is new
    /// calls), then the string arguments in the order they appeared.
    pub fn render(&self) -> String {
        let mut s = String::from("DISCOVER calls, by first display frame (first..last, count):\n");
        let mut calls: Vec<_> = self.calls.iter().collect();
        calls.sort_by_key(|(nid, (first, _, _))| (*first, **nid));
        for (nid, (first, last, n)) in calls {
            s.push_str(&format!(
                "  f{first:<6} f{last:<6} x{n:<8} {}\n",
                match vitaslop_runtime::nid::name(*nid) {
                    vitaslop_runtime::nid::UNKNOWN_NAME => format!("nid {nid:#010x}"),
                    name => name.to_string(),
                }
            ));
        }
        s.push_str("DISCOVER string arguments, by first display frame:\n");
        for ((frame, nid, reg), text) in &self.strings {
            s.push_str(&format!("  f{frame:<6} {} r{reg} {text:?}\n", vitaslop_runtime::nid::name(*nid)));
        }
        s
    }
}
