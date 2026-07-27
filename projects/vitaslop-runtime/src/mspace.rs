//! `sceClibMspace*`: the SCE C library's memory spaces.
//!
//! A title hands the system a block of its own guest memory and asks for a general
//! allocator over it (`sceClibMspaceCreate`), then does its own `malloc`/`free` inside
//! that space. On hardware the allocator's bookkeeping lives in the space itself
//! (dlmalloc's `mspace`); here it lives host-side, keyed by the space's base address,
//! which is what the API's opaque handle is anyway. The guest never inspects the
//! metadata - it only holds the handle and the pointers it was given - so keeping the
//! bookkeeping out of guest memory costs nothing and leaves the entire block available
//! to the title.
//!
//! This is a real allocator, not a bump: a title that runs for minutes will free and
//! reallocate constantly, and a bump would exhaust the space. Blocks are best-fit with
//! neighbour coalescing on free.

use std::collections::{BTreeMap, BTreeSet, HashMap};

/// The alignment every allocation is rounded to. `malloc` must be aligned for any type
/// the C ABI has; on ARM that is 8 bytes.
const MIN_ALIGN: u32 = 8;

/// One memory space: a guest byte range plus what is allocated inside it.
#[derive(Debug)]
pub struct Mspace {
    base: u32,
    capacity: u32,
    /// Free runs by start address, for coalescing with neighbours on free.
    free_by_addr: BTreeMap<u32, u32>,
    /// The same runs keyed `(len, start)`, so a best fit is a range query.
    free_by_size: BTreeSet<(u32, u32)>,
    /// Live allocations: pointer -> the length of the run backing it. The run may be
    /// larger than the requested size (alignment padding is folded into it) so that
    /// freeing gives every byte back.
    used: HashMap<u32, u32>,
}

impl Mspace {
    /// A space over `[base, base + capacity)`.
    pub fn new(base: u32, capacity: u32) -> Mspace {
        let mut m = Mspace {
            base,
            capacity,
            free_by_addr: BTreeMap::new(),
            free_by_size: BTreeSet::new(),
            used: HashMap::new(),
        };
        if capacity > 0 {
            m.insert_free(base, capacity);
        }
        m
    }

    pub fn base(&self) -> u32 {
        self.base
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Bytes currently handed out (not counting alignment padding folded into runs).
    pub fn used_bytes(&self) -> u32 {
        self.used.values().sum()
    }

    fn insert_free(&mut self, start: u32, len: u32) {
        if len == 0 {
            return;
        }
        self.free_by_addr.insert(start, len);
        self.free_by_size.insert((len, start));
    }

    fn remove_free(&mut self, start: u32, len: u32) {
        self.free_by_addr.remove(&start);
        self.free_by_size.remove(&(len, start));
    }

    /// Allocate `size` bytes aligned to `align` (rounded up to at least [`MIN_ALIGN`]).
    /// Returns the guest pointer, or `None` when the space cannot satisfy it - which is
    /// `malloc` returning NULL, a result the caller is expected to handle.
    pub fn alloc(&mut self, size: u32, align: u32) -> Option<u32> {
        let align = align.max(MIN_ALIGN).next_power_of_two();
        let size = size.max(1).next_multiple_of(MIN_ALIGN);

        // Best fit first: the smallest run that fits with no alignment padding. An
        // aligned request may still need padding, so the candidate is re-checked.
        let mut chosen: Option<(u32, u32, u32)> = None; // (run start, run len, aligned ptr)
        for &(len, start) in self.free_by_size.range((size, 0)..) {
            let ptr = start.next_multiple_of(align);
            let pad = ptr - start;
            if pad.checked_add(size).is_some_and(|need| need <= len) {
                chosen = Some((start, len, ptr));
                break;
            }
        }
        // No unpadded-size run worked: an aligned request can still fit in a bigger run.
        // Scan the rest by size ascending, which keeps the fit as tight as alignment
        // allows.
        if chosen.is_none() {
            for &(len, start) in self.free_by_size.iter() {
                let ptr = start.next_multiple_of(align);
                let pad = ptr - start;
                if pad.checked_add(size).is_some_and(|need| need <= len) {
                    chosen = Some((start, len, ptr));
                    break;
                }
            }
        }
        let (start, len, ptr) = chosen?;

        self.remove_free(start, len);
        // Alignment padding before the block goes back on the free list rather than
        // being lost; the tail after it does too.
        self.insert_free(start, ptr - start);
        let end = ptr + size;
        self.insert_free(end, start + len - end);
        self.used.insert(ptr, size);
        Some(ptr)
    }

    /// Free a pointer this space handed out. Returns false for a pointer it did not -
    /// a double free or a foreign pointer, which the caller reports rather than
    /// silently absorbing.
    pub fn free(&mut self, ptr: u32) -> bool {
        let Some(len) = self.used.remove(&ptr) else {
            return false;
        };
        let mut start = ptr;
        let mut len = len;
        // Coalesce with the run ending here.
        if let Some((&prev_start, &prev_len)) = self.free_by_addr.range(..start).next_back() {
            if prev_start + prev_len == start {
                self.remove_free(prev_start, prev_len);
                start = prev_start;
                len += prev_len;
            }
        }
        // Coalesce with the run starting at our end.
        if let Some((&next_start, &next_len)) = self.free_by_addr.range(start + len..).next() {
            if next_start == start + len {
                self.remove_free(next_start, next_len);
                len += next_len;
            }
        }
        self.insert_free(start, len);
        true
    }

    /// Whether `ptr` is a live allocation from this space.
    pub fn owns(&self, ptr: u32) -> bool {
        self.used.contains_key(&ptr)
    }
}

/// Every live memory space, keyed by the handle the guest holds.
///
/// The handle IS the space's base address, which is what `sceClibMspaceCreate` returns on
/// hardware (dlmalloc puts its state at the start of the block). Making the handle
/// identity-resolvable that way means a space stays findable even if the guest copies the
/// handle around - the failure mode a side table keyed by an arbitrary id would have.
#[derive(Debug, Default)]
pub struct MspaceStore {
    spaces: BTreeMap<u32, Mspace>,
}

impl MspaceStore {
    /// Create a space over `[base, base + capacity)` and return its handle, or `None` if
    /// the request is degenerate (the guest handed over nothing to allocate from).
    pub fn create(&mut self, base: u32, capacity: u32) -> Option<u32> {
        if base == 0 || capacity < MIN_ALIGN {
            return None;
        }
        self.spaces.insert(base, Mspace::new(base, capacity));
        Some(base)
    }

    pub fn destroy(&mut self, handle: u32) -> bool {
        self.spaces.remove(&handle).is_some()
    }

    pub fn get_mut(&mut self, handle: u32) -> Option<&mut Mspace> {
        self.spaces.get_mut(&handle)
    }

    pub fn get(&self, handle: u32) -> Option<&Mspace> {
        self.spaces.get(&handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_aligned_and_in_bounds() {
        let mut m = Mspace::new(0x1000, 0x1000);
        let a = m.alloc(16, 0).unwrap();
        let b = m.alloc(16, 0).unwrap();
        assert_eq!(a % MIN_ALIGN, 0);
        assert_eq!(b % MIN_ALIGN, 0);
        assert_ne!(a, b);
        assert!(a >= 0x1000 && b + 16 <= 0x2000);
    }

    #[test]
    fn honours_a_larger_alignment() {
        let mut m = Mspace::new(0x1004, 0x1000);
        // Force a non-aligned start, then demand 256-byte alignment.
        let _ = m.alloc(4, 0).unwrap();
        let p = m.alloc(32, 256).unwrap();
        assert_eq!(p % 256, 0, "memalign did not align");
        assert!(p >= 0x1004 && p + 32 <= 0x2004);
    }

    #[test]
    fn reuses_freed_space_rather_than_growing() {
        let mut m = Mspace::new(0, 0x100);
        let mut ptrs = Vec::new();
        for _ in 0..0x100 / 16 {
            ptrs.push(m.alloc(16, 0).unwrap());
        }
        assert!(m.alloc(16, 0).is_none(), "space should be full");
        for p in &ptrs {
            assert!(m.free(*p));
        }
        // Everything came back, so the whole space is one run again.
        assert_eq!(m.alloc(0x100, 0), Some(0), "free did not coalesce the space back");
    }

    #[test]
    fn coalesces_neighbours_on_free() {
        let mut m = Mspace::new(0, 0x40);
        let a = m.alloc(16, 0).unwrap();
        let b = m.alloc(16, 0).unwrap();
        let c = m.alloc(16, 0).unwrap();
        m.free(a);
        m.free(c);
        // The middle free joins both sides into one 48-byte run.
        m.free(b);
        assert_eq!(m.alloc(0x30, 0), Some(0));
    }

    #[test]
    fn a_full_space_reports_null_rather_than_overrunning() {
        let mut m = Mspace::new(0x2000, 0x20);
        assert!(m.alloc(0x100, 0).is_none());
        let p = m.alloc(0x20, 0).unwrap();
        assert_eq!(p, 0x2000);
        assert!(m.alloc(8, 0).is_none());
        assert_eq!(m.used_bytes(), 0x20);
    }

    #[test]
    fn a_foreign_or_double_free_is_reported() {
        let mut m = Mspace::new(0, 0x40);
        let a = m.alloc(16, 0).unwrap();
        assert!(m.free(a));
        assert!(!m.free(a), "double free reported as ok");
        assert!(!m.free(0x9999), "foreign pointer reported as ok");
    }

    #[test]
    fn the_store_keys_a_space_by_its_base() {
        let mut s = MspaceStore::default();
        let h = s.create(0x8000, 0x1000).unwrap();
        assert_eq!(h, 0x8000);
        let p = s.get_mut(h).unwrap().alloc(64, 0).unwrap();
        assert!(s.get(h).unwrap().owns(p));
        assert!(s.destroy(h));
        assert!(!s.destroy(h));
        assert!(s.create(0, 0x1000).is_none());
    }

    #[test]
    fn a_churning_workload_stays_inside_the_space() {
        let mut m = Mspace::new(0x1_0000, 0x8000);
        let mut live: Vec<u32> = Vec::new();
        // Deterministic pseudo-random sizes: no clock, no rng dependency.
        let mut seed = 12345u32;
        for i in 0..4000 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let size = 8 + (seed >> 20) % 200;
            if i % 3 == 2 && !live.is_empty() {
                let victim = live.swap_remove((seed as usize) % live.len());
                assert!(m.free(victim));
                continue;
            }
            if let Some(p) = m.alloc(size, 0) {
                assert!(p >= 0x1_0000 && p + size <= 0x1_8000, "allocation left the space");
                live.push(p);
            }
        }
        for p in live {
            assert!(m.free(p));
        }
        assert_eq!(m.used_bytes(), 0);
        assert_eq!(m.alloc(0x8000, 0), Some(0x1_0000), "the space did not come back whole");
    }
}
