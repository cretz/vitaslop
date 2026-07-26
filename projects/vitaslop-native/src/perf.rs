//! Opt-in accounting for where a guest frame's wall-clock actually goes.
//!
//! "Guest CPU" - the time between two display flips - is not one thing. It is
//! translated guest code running in the JIT, plus every host call the guest made,
//! plus the scheduler's own bookkeeping between resumes. Those call for completely
//! different fixes, and a single number cannot rank them, so this module splits the
//! frame into buckets that sum to it:
//!
//! - **import** - the whole `env.import` closure: marshalling the register file in
//!   and out of the wasm globals, plus the handler itself.
//!   - **dispatch** - the handler alone ([`ImportDispatch::dispatch`]).
//!   - the difference is MARSHALLING, which is host overhead the guest never asked
//!     for and is therefore pure waste worth attacking on its own.
//! - **everything else** - translated code and the scheduler, by subtraction.
//!
//! Per-selector counts and times come free with that, which is what turns "host
//! calls cost 40 ms" into "one NID costs 40 ms".
//!
//! # Why it is opt-in
//! Two `Instant::now()` calls per host call are tens of nanoseconds each, and a
//! title makes millions of host calls. Measuring must not become the thing being
//! measured, so everything here is behind [`enabled`], a `VITASLOP_PERF` read
//! cached in a `OnceLock`, and compiles to one relaxed atomic load when off.
//!
//! Counters are plain atomics rather than a lock because the scheduler is
//! cooperative (one OS thread runs every guest fiber), so they are uncontended;
//! atomics only keep the module sound if that ever stops being true.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;

/// Highest import selector tracked per-NID. Selectors are dense loader indices and a
/// real title imports a few hundred, so this is slack, not a limit to tune. Calls
/// above it still land in the totals - only their per-NID attribution is dropped.
const MAX_SELECTOR: usize = 4096;

static IMPORT_NS: AtomicU64 = AtomicU64::new(0);
static DISPATCH_NS: AtomicU64 = AtomicU64::new(0);
static IMPORT_CALLS: AtomicU64 = AtomicU64::new(0);
static PER_SEL_NS: [AtomicU64; MAX_SELECTOR] = [const { AtomicU64::new(0) }; MAX_SELECTOR];
static PER_SEL_CALLS: [AtomicU64; MAX_SELECTOR] = [const { AtomicU64::new(0) }; MAX_SELECTOR];

/// Is perf accounting on (`VITASLOP_PERF` set)? Read once and cached.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VITASLOP_PERF").is_some())
}

/// Record one serviced host call: `total_ns` across the whole import closure, of
/// which `dispatch_ns` was the handler. No-op unless [`enabled`].
pub fn note_import(selector: u32, total_ns: u64, dispatch_ns: u64) {
    IMPORT_NS.fetch_add(total_ns, Relaxed);
    DISPATCH_NS.fetch_add(dispatch_ns, Relaxed);
    IMPORT_CALLS.fetch_add(1, Relaxed);
    if let Some(slot) = PER_SEL_NS.get(selector as usize) {
        slot.fetch_add(total_ns, Relaxed);
        PER_SEL_CALLS[selector as usize].fetch_add(1, Relaxed);
    }
}

/// One import selector's share of the measured time.
#[derive(Clone, Copy, Debug)]
pub struct SelectorCost {
    pub selector: u32,
    pub calls: u64,
    pub ns: u64,
}

/// The accumulated split since the last [`reset`].
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    /// Wall-clock inside the `env.import` closure, in nanoseconds.
    pub import_ns: u64,
    /// Of that, wall-clock inside the handler itself.
    pub dispatch_ns: u64,
    /// Number of host calls serviced.
    pub calls: u64,
    /// Per-selector cost, descending by time. Only selectors actually called.
    pub by_selector: Vec<SelectorCost>,
}

impl Snapshot {
    /// Time spent marshalling the guest register file across the host boundary -
    /// the part of a host call that services no guest request at all.
    pub fn marshal_ns(&self) -> u64 {
        self.import_ns.saturating_sub(self.dispatch_ns)
    }
}

/// Read the counters.
pub fn snapshot() -> Snapshot {
    let mut by_selector: Vec<SelectorCost> = (0..MAX_SELECTOR)
        .filter_map(|i| {
            let calls = PER_SEL_CALLS[i].load(Relaxed);
            (calls != 0).then(|| SelectorCost {
                selector: i as u32,
                calls,
                ns: PER_SEL_NS[i].load(Relaxed),
            })
        })
        .collect();
    by_selector.sort_by_key(|s| std::cmp::Reverse(s.ns));
    Snapshot {
        import_ns: IMPORT_NS.load(Relaxed),
        dispatch_ns: DISPATCH_NS.load(Relaxed),
        calls: IMPORT_CALLS.load(Relaxed),
        by_selector,
    }
}

/// Zero every counter. A benchmark calls this after its warm-up prefix so the
/// measured window is the steady state and not the boot.
pub fn reset() {
    IMPORT_NS.store(0, Relaxed);
    DISPATCH_NS.store(0, Relaxed);
    IMPORT_CALLS.store(0, Relaxed);
    for i in 0..MAX_SELECTOR {
        PER_SEL_NS[i].store(0, Relaxed);
        PER_SEL_CALLS[i].store(0, Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Totals accumulate and the per-selector view keeps only what was called,
    /// ordered by cost. Serialized with the reset test through one lock because the
    /// counters are process-global.
    #[test]
    fn accumulates_and_orders_by_cost() {
        let _g = crate::perf::tests::lock();
        reset();
        note_import(3, 100, 60);
        note_import(3, 50, 20);
        note_import(7, 400, 400);
        let s = snapshot();
        assert_eq!(s.calls, 3);
        assert_eq!(s.import_ns, 550);
        assert_eq!(s.dispatch_ns, 480);
        assert_eq!(s.marshal_ns(), 70);
        let sels: Vec<u32> = s.by_selector.iter().map(|c| c.selector).collect();
        assert_eq!(sels, vec![7, 3], "ordered by time, cheapest last");
        assert_eq!(s.by_selector[1].calls, 2);
    }

    /// A selector past the tracked range still counts toward the totals rather than
    /// being silently dropped from them.
    #[test]
    fn out_of_range_selector_still_totals() {
        let _g = lock();
        reset();
        note_import(MAX_SELECTOR as u32 + 5, 90, 10);
        let s = snapshot();
        assert_eq!(s.calls, 1);
        assert_eq!(s.import_ns, 90);
        assert!(s.by_selector.is_empty());
    }

    pub fn lock() -> std::sync::MutexGuard<'static, ()> {
        static L: std::sync::Mutex<()> = std::sync::Mutex::new(());
        L.lock().unwrap_or_else(|e| e.into_inner())
    }
}
