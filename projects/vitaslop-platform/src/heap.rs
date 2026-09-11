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
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        gave(layout.size());
        unsafe { self.0.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { self.0.alloc_zeroed(layout) };
        if !p.is_null() {
            took(layout.size());
        }
        p
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { self.0.realloc(ptr, layout, new_size) };
        if !p.is_null() {
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
