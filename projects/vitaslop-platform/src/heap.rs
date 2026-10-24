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
    /// of the allocator, of `alloc::` and of this module are noise on every trace. Led by the
    /// innermost frame OUTSIDE the workspace and the standard library when one sits below the
    /// first workspace frame (`[wgpu_core::...]`): a football title's largest holder read only
    /// "194 MB in 1191 allocation(s): headless_check" - a dependency allocated it, and the
    /// workspace frames alone could not say which.
    fn site_of(trace: &std::backtrace::Backtrace) -> String {
        let text = format!("{trace}");
        let mut frames = Vec::new();
        let mut external: Option<String> = None;
        for line in text.lines() {
            let line = line.trim();
            // `N: symbol` lines carry the function; the `at file:line` lines follow them.
            if !line.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                continue;
            }
            let Some((_, sym)) = line.split_once(": ") else { continue };
            if sym.contains("heap::") {
                continue;
            }
            if !sym.starts_with("vitaslop") && !sym.starts_with("<vitaslop") {
                const NOISE: [&str; 8] =
                    ["alloc::", "std::", "core::", "<alloc::", "<std::", "<core::", "__rust", "backtrace"];
                if frames.is_empty() && external.is_none() && !NOISE.iter().any(|n| sym.starts_with(n)) {
                    external = Some(format!("[{}]", sym.chars().take(96).collect::<String>()));
                }
                continue;
            }
            if frames.is_empty()
                && let Some(ext) = external.take()
            {
                frames.push(ext);
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

/// >>> THE THREADS BUNDLE'S HEAP: dlmalloc, GROWING THE WASM MEMORY 16 MB AT A TIME.
///
/// std's wasm32 allocator is dlmalloc too, but it grows the memory one 64 KB `memory.grow` at a
/// time, and in the threads bundle a grow of the SHARED memory is expensive. MEASURED (desktop
/// browser, V8 sampling profiles of every worker over a baseball title's load into the game):
/// dlmalloc's page provider was 1.2-1.3 s of self time on each of the render and resolver
/// workers per 35 s, and 81% of a shader translation's samples - the same pairs translated in
/// 213-248 ms with the guest workers running and 17-24 ms without. With this heap: that title's
/// in-frame translation 2194 -> 529 ms, the load 88% -> 92% speed, worst freeze 3.8 -> 1.9 s; a
/// football title's translation 376 -> 126 ms, worst freeze 812 -> 278 ms.
///
/// New blocks come from this heap from the first allocation ([`route_all_allocations`];
/// `VITASLOP_HEAP_BIG_GROW=0` is the arm back to std's). Its pages are marked in a page bitmap,
/// and a free or realloc is routed by WHICH heap owns the pointer's page - so blocks std's heap
/// made before a switch are still freed there. Outside the threads bundle (and natively) this
/// forwards unchanged.
pub struct PrivateHeap<A>(pub A);

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
mod private {
    use std::cell::UnsafeCell;
    use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

    /// One bit per 64 KB wasm page (4 GiB of them): set when the second heap took that page.
    static OWNED: [AtomicU32; 2048] = [const { AtomicU32::new(0) }; 2048];
    /// Bytes of pages the second heap has taken, for a report line.
    pub static PAGES_BYTES: AtomicU64 = AtomicU64::new(0);

    pub struct Pages;

    /// The smallest `memory.grow` the second heap makes, in bytes.
    const GROW_STEP: usize = 16 << 20;

    // SAFETY: hands out whole fresh pages from `memory.grow` and never returns any, exactly like
    // dlmalloc's own wasm `System`; marking them is the only addition.
    unsafe impl dlmalloc::Allocator for Pages {
        fn alloc(&self, size: usize) -> (*mut u8, usize, u32) {
            // At least GROW_STEP at a time: dlmalloc asks for 64 KB segments, and a
            // `memory.grow` of a SHARED memory is expensive in the browser - MEASURED, 81% of a
            // shader translation's samples sat in this call when it grew 64 KB at a time.
            // Near wasm32's 4 GiB ceiling the step can fail where the request itself fits: then
            // take exactly what was asked, so this heap runs out no sooner than std's would.
            let mut pages = size.max(GROW_STEP).div_ceil(65536);
            let mut prev = core::arch::wasm32::memory_grow(0, pages);
            if prev == usize::MAX && size < GROW_STEP {
                pages = size.div_ceil(65536);
                prev = core::arch::wasm32::memory_grow(0, pages);
            }
            if prev == usize::MAX {
                return (std::ptr::null_mut(), 0, 0);
            }
            for p in prev..prev + pages {
                OWNED[p / 32].fetch_or(1 << (p % 32), Ordering::Release);
            }
            PAGES_BYTES.fetch_add((pages * 65536) as u64, Ordering::Relaxed);
            ((prev * 65536) as *mut u8, pages * 65536, 0)
        }
        fn remap(&self, _: *mut u8, _: usize, _: usize, _: bool) -> *mut u8 {
            std::ptr::null_mut()
        }
        fn free_part(&self, _: *mut u8, _: usize, _: usize) -> bool {
            false
        }
        fn free(&self, _: *mut u8, _: usize) -> bool {
            false
        }
        fn can_release_part(&self, _: u32) -> bool {
            false
        }
        fn allocates_zeros(&self) -> bool {
            true
        }
        fn page_size(&self) -> usize {
            65536
        }
    }

    struct Locked {
        busy: AtomicBool,
        heap: UnsafeCell<dlmalloc::Dlmalloc<Pages>>,
    }
    // SAFETY: `heap` is only touched while `busy` is held (see `with`).
    unsafe impl Sync for Locked {}

    static HEAP: Locked =
        Locked { busy: AtomicBool::new(false), heap: UnsafeCell::new(dlmalloc::Dlmalloc::new_with_allocator(Pages)) };

    pub fn with<R>(f: impl FnOnce(&mut dlmalloc::Dlmalloc<Pages>) -> R) -> R {
        while HEAP.busy.swap(true, Ordering::Acquire) {
            std::hint::spin_loop();
        }
        // SAFETY: the lock above is held until the store below.
        let r = f(unsafe { &mut *HEAP.heap.get() });
        HEAP.busy.store(false, Ordering::Release);
        r
    }

    pub fn owns(p: *mut u8) -> bool {
        let page = p as usize >> 16;
        OWNED[page / 32].load(Ordering::Acquire) & (1 << (page % 32)) != 0
    }

    /// Whether new blocks come from this heap - `route_all_allocations`. ON from the first
    /// allocation; `VITASLOP_HEAP_BIG_GROW=0` turns it off before the run starts.
    pub static ALL: AtomicBool = AtomicBool::new(true);

    #[inline]
    pub fn here() -> bool {
        ALL.load(Ordering::Relaxed)
    }
}

/// Whether new blocks come from the 16 MB-step heap ([`PrivateHeap`]) - ON from the start;
/// `VITASLOP_HEAP_BIG_GROW=0` (thrown in `set_knob`, before the run starts) goes back to std's.
/// Blocks made before a switch stay in the heap that made them and are freed there - routing is
/// by page. A no-op outside the threads bundle.
pub fn route_all_allocations(on: bool) {
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    private::ALL.store(on, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
    let _ = on;
}

/// Bytes of pages the second heap has taken from the wasm memory (0 where there is none).
pub fn private_heap_bytes() -> u64 {
    #[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
    {
        private::PAGES_BYTES.load(Ordering::Relaxed)
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
    {
        0
    }
}

// SAFETY: a block is freed or resized by the heap that owns its page (`private::owns`), which is
// the heap that allocated it; everything outside a scope forwards to the inner allocator.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
unsafe impl<A: GlobalAlloc> GlobalAlloc for PrivateHeap<A> {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if private::here() {
            private::with(|h| unsafe { h.malloc(l.size(), l.align()) })
        } else {
            unsafe { self.0.alloc(l) }
        }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if private::here() {
            private::with(|h| unsafe { h.calloc(l.size(), l.align()) })
        } else {
            unsafe { self.0.alloc_zeroed(l) }
        }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if private::owns(p) {
            private::with(|h| unsafe { h.free(p, l.size(), l.align()) })
        } else {
            unsafe { self.0.dealloc(p, l) }
        }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let inside = private::here();
        match (private::owns(p), inside) {
            (true, true) => private::with(|h| unsafe { h.realloc(p, l.size(), l.align(), n) }),
            (false, false) => unsafe { self.0.realloc(p, l, n) },
            // Crossing heaps: allocate where this thread allocates now, copy, free where it was.
            _ => {
                // SAFETY: `GlobalAlloc::realloc`'s contract makes `n` with `l.align()` a valid layout.
                let nl = unsafe { Layout::from_size_align_unchecked(n, l.align()) };
                let q = unsafe { self.alloc(nl) };
                if !q.is_null() {
                    unsafe { std::ptr::copy_nonoverlapping(p, q, l.size().min(n)) };
                    unsafe { self.dealloc(p, l) };
                }
                q
            }
        }
    }
}

// SAFETY: forwards every call unchanged.
#[cfg(not(all(target_arch = "wasm32", target_feature = "atomics")))]
unsafe impl<A: GlobalAlloc> GlobalAlloc for PrivateHeap<A> {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        unsafe { self.0.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        unsafe { self.0.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { self.0.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        unsafe { self.0.realloc(p, l, n) }
    }
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
