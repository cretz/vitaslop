//! What the RUST HEAP is holding, in bytes, at any moment.
//!
//! # Why this exists
//! The one number that decides whether a title can run in a browser at all is the size of the
//! emulator's linear memory, and nothing could attribute it. `memory_size(0)` says how many
//! pages the module has taken; it never shrinks, so by the time it is large the allocation that
//! made it large is long gone. MEASURED on one retail title: the panel read `emulator wasm heap
//! 68 MB` for nine thousand frames and `3412 MB` thirty frames later, with wasm32's 4,096 MB
//! ceiling 684 MB above that - which is a tab that dies, and the user's report of one. A page
//! count cannot say what happened in those thirty frames; a LIVE byte count can, because it goes
//! down again.
//!
//! Two atomics on every allocation and free, `Relaxed`, no ordering with anything. That price is
//! paid on a path a profile of this engine ranks at 2.3% of its busy time, and it buys the only
//! residency number in the project that is not an estimate.
//!
//! [`live_bytes`] is what is held RIGHT NOW - it falls when a cache is evicted, so a rise that
//! does not fall is a leak and a rise that falls is a working set. [`peak_bytes`] is the
//! high-water mark, which is the number the wasm heap actually took pages for and can never give
//! back. Reading them apart is the whole point: a 2 GB peak with a 200 MB live figure is a
//! TRANSIENT that costs a phone 2 GB of address space forever.

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Bytes the Rust allocator is holding right now.
pub fn live_bytes() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// The largest [`live_bytes`] ever reached. The wasm heap took pages for this and, on wasm32,
/// can never hand them back - so this, not the live figure, is what a device pays.
pub fn peak_bytes() -> usize {
    PEAK.load(Ordering::Relaxed)
}

/// Drop the high-water mark to the current live figure, so the next phase's peak is its own.
/// Used at boot checkpoints; a running frame loop must not call it or the peak means nothing.
pub fn reset_peak() {
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// `live / peak` in megabytes, for a report line.
pub fn live_peak_mb() -> (usize, usize) {
    (live_bytes() / (1024 * 1024), peak_bytes() / (1024 * 1024))
}

/// # The large-allocation ledger (`VITASLOP_HEAP_TRACE=<min MB>`, native only)
///
/// The two counters say HOW MUCH is held and never WHO holds it, and a renderer whose named
/// caches add up to 200 MB under a 2,400 MB live figure has 2,200 MB of holders nobody can
/// name from a cache line. MEASURED on a baseball title's stadium: the heap climbed 70-90 MB
/// every four frames, dropped 1,658 MB in one frame with no cache reporting a clear, and
/// climbed again - a shape no counter can attribute.
///
/// When armed (`trace_large`), every allocation of at least the threshold keeps its
/// backtrace for as long as it is live, and `large_live_report` groups what is live by
/// allocation site. Only allocations that large pay for a backtrace, which keeps it off the
/// hot path.
///
/// The ledger's own allocations go through the same allocator: a thread-local guard keeps
/// them out of the ledger (and out of a recursion).
mod ledger {
    use std::alloc::Layout;
    use std::cell::Cell;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// Minimum size that is traced; 0 = off.
    pub static THRESHOLD: AtomicUsize = AtomicUsize::new(0);

    struct Entry {
        size: usize,
        trace: std::backtrace::Backtrace,
    }

    static LIVE: Mutex<Option<HashMap<usize, Entry>>> = Mutex::new(None);

    thread_local! {
        static INSIDE: Cell<bool> = const { Cell::new(false) };
    }

    /// Run `f` with the ledger's reentrancy guard held; `None` if it was already held.
    fn guarded<R>(f: impl FnOnce() -> R) -> Option<R> {
        INSIDE.with(|g| {
            if g.get() {
                return None;
            }
            g.set(true);
            let r = f();
            g.set(false);
            Some(r)
        })
    }

    pub fn record(ptr: *mut u8, size: usize) {
        let min = THRESHOLD.load(Ordering::Relaxed);
        if min == 0 || size < min {
            return;
        }
        guarded(|| {
            let trace = std::backtrace::Backtrace::force_capture();
            let mut g = LIVE.lock().unwrap_or_else(|e| e.into_inner());
            g.get_or_insert_with(HashMap::new).insert(ptr as usize, Entry { size, trace });
        });
    }

    pub fn forget(ptr: *mut u8, layout: Layout) {
        let min = THRESHOLD.load(Ordering::Relaxed);
        if min == 0 || layout.size() < min {
            return;
        }
        guarded(|| {
            let mut g = LIVE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(m) = g.as_mut() {
                m.remove(&(ptr as usize));
            }
        });
    }

    /// The live allocations at or above the threshold, grouped by allocation site, largest
    /// total first: `(bytes, count, site)` where `site` is the first few workspace frames
    /// of the backtrace.
    pub fn report(top: usize) -> Vec<(usize, usize, String)> {
        guarded(|| {
            let g = LIVE.lock().unwrap_or_else(|e| e.into_inner());
            let Some(m) = g.as_ref() else { return Vec::new() };
            let mut by_site: HashMap<String, (usize, usize)> = HashMap::new();
            for e in m.values() {
                let site = site_of(&e.trace);
                let s = by_site.entry(site).or_insert((0, 0));
                s.0 += e.size;
                s.1 += 1;
            }
            let mut v: Vec<(usize, usize, String)> =
                by_site.into_iter().map(|(k, (b, c))| (b, c, k)).collect();
            v.sort_by_key(|x| std::cmp::Reverse(x.0));
            v.truncate(top);
            v
        })
        .unwrap_or_default()
    }

    /// The workspace frames of a backtrace, innermost first, up to six of them - the frames
    /// of the allocator, of `alloc::` and of this module are noise on every trace.
    fn site_of(trace: &std::backtrace::Backtrace) -> String {
        let text = format!("{trace}");
        let mut frames = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            // `N: symbol` lines carry the function; the `at file:line` lines follow them.
            if !line.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                continue;
            }
            let Some((_, sym)) = line.split_once(": ") else { continue };
            if !sym.starts_with("vitaslop") || sym.contains("heap::") {
                continue;
            }
            let short: String = sym.chars().take(96).collect();
            frames.push(short);
            if frames.len() >= 6 {
                break;
            }
        }
        if frames.is_empty() { "(no workspace frame)".to_string() } else { frames.join(" <- ") }
    }
}

/// Arm the large-allocation ledger: every allocation of `min_bytes` or more keeps its
/// backtrace while live. See [`ledger`]. `0` disarms it.
pub fn trace_large(min_bytes: usize) {
    ledger::THRESHOLD.store(min_bytes, Ordering::Relaxed);
}

/// The ledger's report as lines: the `top` sites holding the most live bytes at or above
/// the threshold, as `"<MB> MB in <n> allocation(s): <site>"`. Empty when not armed.
pub fn large_live_report(top: usize) -> Vec<String> {
    ledger::report(top)
        .into_iter()
        .map(|(b, c, site)| format!("{} MB in {c} allocation(s): {site}", b / (1024 * 1024)))
        .collect()
}

#[inline]
fn took(n: usize) {
    // `fetch_max` rather than a compare-exchange loop: one instruction's worth of contention on
    // the hot path, and a peak that is never wrong under threads.
    let live = LIVE.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

#[inline]
fn gave(n: usize) {
    LIVE.fetch_sub(n, Ordering::Relaxed);
}

/// A [`GlobalAlloc`] that counts. Wrap the allocator a front end would have used anyway:
///
/// ```ignore
/// #[global_allocator]
/// static ALLOC: vitaslop_platform::heap::Counting<std::alloc::System> =
///     vitaslop_platform::heap::Counting(std::alloc::System);
/// ```
pub struct Counting<A>(pub A);

// SAFETY: every method forwards to the wrapped allocator unchanged; the counters are the only
// addition and they touch no memory the allocator owns.
unsafe impl<A: GlobalAlloc> GlobalAlloc for Counting<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { self.0.alloc(layout) };
        if !p.is_null() {
            took(layout.size());
            ledger::record(p, layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        gave(layout.size());
        ledger::forget(ptr, layout);
        unsafe { self.0.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { self.0.alloc_zeroed(layout) };
        if !p.is_null() {
            took(layout.size());
            ledger::record(p, layout.size());
        }
        p
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ledger::forget(ptr, layout);
        let p = unsafe { self.0.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            ledger::record(p, new_size);
            // Count the DELTA, and count a grow as a peak event: a realloc that grows may have
            // copied, in which case both blocks were live for the length of the copy - but the
            // allocator has already released the old one by the time it returns, so the honest
            // figure here is the new size.
            if new_size >= layout.size() {
                took(new_size - layout.size());
            } else {
                gave(layout.size() - new_size);
            }
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A live count that goes UP on an allocation and back DOWN on its free - which is the
    /// property that separates this from `memory_size`, and the reason the module exists.
    ///
    /// Driven through an EXPLICIT `Counting` instance rather than by allocating a `Vec`: only a
    /// front end installs this as `#[global_allocator]`, so inside this crate's own test binary
    /// an ordinary allocation goes to the untracked system allocator and the counters never
    /// move. Testing the wrapper directly is also the honest subject - the accounting is what
    /// this module contributes, and `System` is not ours to test.
    #[test]
    fn live_bytes_rise_and_fall() {
        let alloc = Counting(std::alloc::System);
        let layout = Layout::from_size_align(8 * 1024 * 1024, 8).unwrap();
        // Not asserted against an absolute figure: other threads in the test binary may be
        // driving the counters too. The DELTA across a known allocation is the claim.
        let before = live_bytes();
        // SAFETY: the layout is non-zero and well-formed, and the pointer is freed below with
        // the same layout it was allocated with.
        let p = unsafe { alloc.alloc(layout) };
        assert!(!p.is_null(), "the system allocator refused 8 MB");
        let held = live_bytes();
        assert!(held >= before + layout.size(), "an 8 MB allocation must show as at least 8 MB held");
        assert!(peak_bytes() >= held, "the peak must cover the live figure");
        // SAFETY: `p` came from the call above with this exact layout.
        unsafe { alloc.dealloc(p, layout) };
        assert!(
            live_bytes() <= held - layout.size(),
            "freeing it must give the bytes back - a count that only goes up is memory_size again"
        );
    }

    /// A REALLOC counts its delta, in both directions. A grow that counted the whole new size
    /// would double-count the block, and a shrink that counted nothing would leak the
    /// difference for the life of the process.
    #[test]
    fn a_realloc_counts_only_the_delta() {
        let alloc = Counting(std::alloc::System);
        let small = Layout::from_size_align(1024 * 1024, 8).unwrap();
        let before = live_bytes();
        // SAFETY: well-formed layout; the block is reallocated and then freed below.
        let p = unsafe { alloc.alloc(small) };
        assert!(!p.is_null());
        // SAFETY: `p` was allocated with `small`; 4 MB is a valid new size.
        let grown = unsafe { alloc.realloc(p, small, 4 * 1024 * 1024) };
        assert!(!grown.is_null());
        assert!(
            live_bytes() >= before + 4 * 1024 * 1024,
            "a grow must add the difference, not nothing"
        );
        let big = Layout::from_size_align(4 * 1024 * 1024, 8).unwrap();
        // SAFETY: `grown` is live with `big`'s size and alignment.
        unsafe { alloc.dealloc(grown, big) };
        assert!(live_bytes() <= before, "the whole grown block must come back");
    }
}
