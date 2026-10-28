use super::*;

/// The number of VFP single-precision registers (s0..s15) that carry floating-
/// point arguments and the float return under AAPCS-VFP (the Vita is hardfloat).
/// The host reads these from the guest at a NID trap alongside the core registers
/// so [`GuestCtx`] can marshal float args and returns.
pub const VFP_ARG_COUNT: usize = 16;

/// A guest address crossing the host-call boundary (an in- or out-parameter). A
/// newtype so `#[hostcall]` classifies it as integer class (core register / stack)
/// while keeping the intent - "this is a pointer" - visible in the signature. The
/// handler dereferences it through [`GuestCtx`] (`read_u32`, `write_bytes`, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ptr(pub u32);

impl Ptr {
    /// The raw guest address.
    pub fn addr(self) -> u32 {
        self.0
    }
    /// True for a null guest pointer.
    pub fn is_null(self) -> bool {
        self.0 == 0
    }
}

/// Random-access to the guest's linear memory during a host call, rebased so
/// guest address `A` is byte `A - base`. The abstraction exists because the two
/// engines reach guest memory differently: on native wasmtime the runtime and
/// guest share one address space, so a plain `&mut [u8]` slice works and is
/// zero-copy; in the browser the guest runs as its own `WebAssembly` instance
/// with a linear memory the runtime's (wasm-bindgen) module cannot borrow as a
/// Rust slice, so its impl copies through the guest `ArrayBuffer`. Host calls
/// happen only at kernel/GXM boundaries (tens per frame), never in the hot CPU
/// loop, so the per-access indirection here costs nothing on the fast path.
///
/// Callers (via [`GuestCtx`]) validate `off` against [`len`](GuestMemory::len)
/// and clamp before calling, so implementors may assume `off` and
/// `off + buf.len()` are in range.
/// Read and write single guest WORDS by guest address, for state that lives in guest
/// memory and must be reachable from outside a host call.
///
/// [`GuestCtx`] is the ordinary way to touch guest memory, but it exists only for the
/// duration of a dispatch. Some state the guest owns has to be maintained at points where
/// no guest call is in flight - a lightweight mutex handed to a waiter whose cond wait
/// TIMED OUT is decided in `advance_time_to`, on the scheduler's idle path, with no ctx
/// anywhere. This is the narrow accessor those paths take instead, so the state can have
/// exactly one home (see [`crate::vita::lwwork`]) rather than a guest copy and a host copy
/// that agree until they do not.
///
/// An address outside the provisioned region reads zero and swallows its write, matching
/// what [`GuestCtx`] does. That is a real answer for the callers here: a work area outside
/// guest memory is not a mutex, and every fast path tests its identity stamp first.
pub trait GuestWords {
    fn word(&self, addr: u32) -> u32;
    fn set_word(&mut self, addr: u32, value: u32);
    /// Store `new` at `addr` if it reads `expect`; return what it read. Atomic where guest
    /// code on other workers can write the word concurrently - see
    /// [`GuestMemory::cas_u32`]. The default is exact for a single-baton backing.
    fn cas_word(&mut self, addr: u32, expect: u32, new: u32) -> u32 {
        let cur = self.word(addr);
        if cur == expect {
            self.set_word(addr, new);
        }
        cur
    }
}

/// A [`GuestMemory`] at its `base`, read by guest address with a bounds check.
pub(super) struct MemRead<'a> {
    pub(super) mem: &'a dyn GuestMemory,
    pub(super) base: u32,
}

impl crate::guest_read::GuestRead for MemRead<'_> {
    fn read_into(&self, addr: u32, out: &mut [u8]) -> bool {
        let Some(off) = addr.checked_sub(self.base).map(|o| o as usize) else {
            return false;
        };
        if off + out.len() > self.mem.len() {
            return false;
        }
        self.mem.read(off, out);
        true
    }
}

pub trait GuestMemory {
    /// The size of the provisioned guest region, in bytes from `base`.
    fn len(&self) -> usize;
    /// True when a read of this memory is a plain load - the bytes are in this process's
    /// own address space and [`borrow`](Self::borrow) hands them out without a copy. False
    /// for a memory that lives across a boundary (the browser's typed-array view), where a
    /// caller that can batch its reads should ([`PrefetchedMemory`]). The batching is pure
    /// overhead on a direct memory, so it asks this first.
    fn direct(&self) -> bool {
        false
    }
    /// True when no guest memory is provisioned.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Read `buf.len()` bytes starting at rebased offset `off` into `buf`.
    fn read(&self, off: usize, buf: &mut [u8]);
    /// Write `bytes` at rebased offset `off`.
    fn write(&mut self, off: usize, bytes: &[u8]);
    /// Read the little-endian u32 at rebased offset `off`, which the caller has already
    /// bounds-checked.
    ///
    /// # Why a backing gets to override this
    /// The default is [`read`](Self::read) into a four-byte buffer, which is right for an
    /// in-process backing where a block read is a `memcpy`. It is NOT right for the
    /// browser, where guest memory is a `SharedArrayBuffer` reached through a typed array:
    /// there a block read is a `subarray` (a boundary crossing that ALLOCATES a JS object)
    /// followed by a `copy_to` (a second crossing), where an aligned word is one
    /// `get_index` into a `Uint32Array` over the same buffer - one crossing, no allocation.
    ///
    /// That matters because word reads are not rare. MEASURED on a racing title's on-track
    /// frame in the browser: ~22,000 single-word reads per presented frame, after the four largest
    /// per-word readers had already been converted to bulk reads
    /// ([[vitaslop-count-calls-not-bytes-across-the-guest-boundary]]). Converting a reader
    /// to a bulk read removes its words; this makes the words that REMAIN - in every
    /// handler, on every path, including ones nobody has profiled - cost a third as much.
    fn read_u32(&self, off: usize) -> u32 {
        let mut b = [0u8; 4];
        self.read(off, &mut b);
        u32::from_le_bytes(b)
    }

    /// Read the little-endian u16 at rebased offset `off`. See [`read_u32`](Self::read_u32).
    fn read_u16(&self, off: usize) -> u16 {
        let mut b = [0u8; 2];
        self.read(off, &mut b);
        u16::from_le_bytes(b)
    }

    /// Write the little-endian u32 `v` at rebased offset `off`. See
    /// [`read_u32`](Self::read_u32) - a write pays the same two crossings a read does, and
    /// the scalar GXM setters and the scheduler's guest-visible mirror are all writes.
    ///
    /// An overriding implementation owes the same side effects [`write`](Self::write) has,
    /// which for the browser means stamping the guest-store dirty map.
    fn write_u32(&mut self, off: usize, v: u32) {
        self.write(off, &v.to_le_bytes());
    }

    /// Compare-and-swap the little-endian u32 at rebased offset `off`: store `new` if it
    /// reads `expect`, and return what it read either way.
    ///
    /// The default is a read and a conditional write, which is exact for a backing only one
    /// guest thread touches at a time. A backing that guest code on OTHER workers writes
    /// concurrently (the browser's parallel run) must make it one atomic operation - the
    /// lightweight mutex's inline SMP form races the host on exactly this word (see
    /// `vita::lwwork`). An overriding implementation owes `write`'s side effects on a store.
    fn cas_u32(&mut self, off: usize, expect: u32, new: u32) -> u32 {
        let cur = self.read_u32(off);
        if cur == expect {
            self.write_u32(off, new);
        }
        cur
    }

    /// Borrow `len` bytes at rebased offset `off` directly, without copying, if this
    /// backing can lend them.
    ///
    /// Every in-process host can: guest memory is a byte buffer it owns. The point is
    /// to COMPARE a large region against a host-side copy without materialising a
    /// second one - which is how the texture-snapshot cache stays exact (it re-reads
    /// only what actually changed) instead of re-reading everything every frame.
    ///
    /// The default is `None`, so a backing that cannot lend simply loses the
    /// optimisation rather than the correctness.
    fn borrow(&self, off: usize, len: usize) -> Option<&[u8]> {
        let _ = (off, len);
        None
    }

    /// Has the guest STORED anywhere in rebased `[off, off + len)` at or after epoch
    /// `stamp`? `Some(true)` if it may have, `Some(false)` if it provably did not,
    /// `None` if this backing tracks no stores at all.
    ///
    /// `None` is not "clean" - it is "no answer", and a caller that cannot tell the
    /// difference would serve stale pixels. Every caller must fall back to reading the
    /// bytes.
    ///
    /// The map is a byte per 4 KB page holding the epoch of the last store into it; the
    /// transpiler writes it (`emit_dirty_mark`, which explains why it is an epoch and
    /// not a flag) and this reads it. Nothing is ever cleared, so two callers asking
    /// about overlapping pages cannot spoil each other's answer.
    ///
    /// # The one-page overhang
    /// The transpiler stamps the page a store STARTS in, not every page it touches -
    /// widening it costs as much again on the hottest path in the module, and the
    /// reader gets it for free. The largest translated store is 8 bytes, so a store
    /// reaching into a page can only have started in that page or the one below it.
    /// An implementation therefore examines one page BELOW `off` as well, which makes
    /// the answer exact.
    fn dirty_since(&self, off: usize, len: usize, stamp: u8) -> Option<bool> {
        let _ = (off, len, stamp);
        None
    }

    /// Read every `(off, len)` range of `ranges`, concatenated in order, into `out` (cleared
    /// first; a range past the end reads as zeros). The default is one `read` per range; a
    /// memory that lives across a boundary overrides it with ONE crossing for the whole
    /// table - see the browser's `SharedView`, and [`PrefetchedMemory`] for who asks.
    fn read_many(&self, ranges: &[(usize, usize)], out: &mut Vec<u8>) {
        out.clear();
        for &(off, len) in ranges {
            let start = out.len();
            out.resize(start + len, 0);
            let n = len.min(self.len().saturating_sub(off));
            if n > 0 {
                self.read(off, &mut out[start..start + n]);
            }
        }
    }

    /// A COPY of the whole guest-store dirty map - one stamp byte per page - with the page
    /// shift, when this memory keeps one. Taken once, it answers every `dirty_since` of a
    /// host call from Rust instead of one crossing each. `None` where there is no map.
    fn dirty_stamps(&self) -> Option<(Vec<u8>, u32)> {
        None
    }

    /// >>> WHICH PARTS of `[off, off + len)` the guest may have stored into since `stamp`,
    /// > > > as byte ranges RELATIVE TO `off`, appended to `out`. `None` from a backing that
    /// > > > tracks no stores - which is not "clean" and not "all dirty", it is no answer.
    ///
    /// [`dirty_since`](Self::dirty_since) answers one bit for a whole range, so a texture is
    /// re-read in full whenever any page of it was written. CENSUSED on a browser run of one
    /// title (`VITASLOP_TEX_DIRTY_CENSUS`): across 1,600 re-read ranges covering 22,801 pages,
    /// **1,611 pages were dirty - 7.1%, and under 2% for the large textures** that dominate
    /// the bytes. The rest of every one of those reads was work done to re-establish bytes the
    /// map already knew had not changed.
    ///
    /// The same one-page overhang applies as in `dirty_since`: a store is stamped against the
    /// page it STARTS in and the largest translated store is 8 bytes, so an implementation
    /// must treat this range's first page as dirty whenever the page below it is.
    fn dirty_runs_since(
        &self,
        off: usize,
        len: usize,
        stamp: u8,
        out: &mut Vec<(usize, usize)>,
    ) -> Option<()> {
        let _ = (off, len, stamp, out);
        None
    }

    /// Advance the guest-store epoch and return `(the stamp to record, whether the
    /// epoch WRAPPED)`. `None` when this backing tracks no stores.
    ///
    /// A caller records the returned stamp against bytes it has just read out of guest
    /// memory; a later [`dirty_since`](Self::dirty_since) with that stamp then answers
    /// "has the guest written here since I read it?". Advancing BEFORE the caller
    /// records is what makes it exact: the guest cannot run during a host call, so no
    /// store can already carry the new epoch.
    ///
    /// The epoch is one byte, so it runs out; on wrap the map is zeroed and `true` is
    /// returned, and every stamp recorded before that moment is meaningless. A caller
    /// that sees `true` must discard its stamps and re-establish them the slow way.
    ///
    /// # Why `&self` for a method that writes
    /// The map is not guest state and not Rust-owned memory: the only backing that has
    /// one reaches it through a JS typed array over a `SharedArrayBuffer`, where no
    /// `&mut [u8]` exists to alias. Taking `&mut self` would instead force
    /// [`GuestCtx::borrow_bytes`]'s caller to hold a mutable borrow of guest memory
    /// across the very compare it is trying to avoid.
    fn bump_dirty_epoch(&self) -> Option<(u8, bool)> {
        None
    }

    /// The epoch a store made RIGHT NOW would be stamped with, or `None` from a backing
    /// that tracks no stores. The number [`rebase_dirty_epoch`](Self::rebase_dirty_epoch)
    /// exists to keep away from its ceiling.
    fn dirty_epoch(&self) -> Option<u8> {
        None
    }

    /// >>> RENUMBER THE EPOCH INSTEAD OF SPENDING IT, so a wrap does not throw the whole
    /// > > > texture working set away.
    ///
    /// The epoch is ONE BYTE and advances once per SCENE. A race frame is eleven scenes and
    /// the browser runs more than one guest frame per present, so the 253 usable values are
    /// gone in about ten presented frames - and [`bump_dirty_epoch`](Self::bump_dirty_epoch)
    /// answers that by zeroing the map, which invalidates every stamp its caller holds.
    /// MEASURED on a racing title's on-track frame in the browser: **5.40 MB per present**
    /// copied
    /// across the guest boundary and compared, purely to re-establish snapshots a wrap had
    /// just disowned - and every one of those compares found the bytes IDENTICAL.
    ///
    /// The values in use are not spread over the range: a snapshot is re-stamped whenever it
    /// is proved current, so live stamps cluster just below the epoch and everything below
    /// `floor` is free. Subtracting `floor - 1` from the map, from the epoch and from every
    /// stamp the caller holds reclaims all of it and preserves the `page >= stamp` predicate
    /// exactly - a page below `floor` becomes 0, which is below every renumbered stamp, and it
    /// was below every live stamp before.
    ///
    /// `floor` MUST be the lowest stamp the caller still holds; a stamp below it would be
    /// renumbered into nonsense. Returns the new current epoch, and `None` from a backing
    /// with no map (whose caller then keeps the wrap it always had).
    fn rebase_dirty_epoch(&self, floor: u8) -> Option<u8> {
        let _ = floor;
        None
    }

    /// Stamp every page of rebased `[off, off + len)` with the CURRENT epoch, as a guest
    /// store into each of them would. A backing with no map does nothing.
    ///
    /// # Why this is the host's job for the STACKS
    /// The transpiler does not mark a store whose base register is SP (`emit::plan_dirty_run`):
    /// that was three quarters of every mark a frame paid (a football title's main thread ran
    /// +8.6% without them). The pages it would have marked are the guest thread stacks, so the
    /// host stamps them instead, at every epoch it hands out ([`TextureSnapshots::stamp_stacks`]),
    /// and a stack page therefore reads as written-since for every stamp anyone holds. That is
    /// exactly as conservative as a guest that stored into the page every scene - so a vertex
    /// array a title builds on its stack and draws is re-read, as it always was.
    ///
    /// REQUIRED, with no default: a backing (or a wrapper that forgets to forward) that did
    /// nothing here would silently serve a stale stack buffer. A census of 13 titles found no
    /// GXM snapshot reading guest data on a stack, but the guarantee is this method, not that.
    fn stamp_dirty_pages(&self, off: usize, len: usize);
}

/// >>> THE PAGE SCAN BEHIND [`GuestMemory::dirty_runs_since`], written ONCE.
///
/// Both engines answer this question and they had a copy of the loop each - which is how they
/// came to differ: one of them applied the page-below overhang to the range's FIRST page only.
/// The rule is stated here, and `stamp_of` is the only thing an engine supplies.
///
/// `stamp_of` is called for pages `first ..= last` and for `first - 1` when there is one, so a
/// caller that copies the map must copy the page below the range as well.
///
/// # The overhang, and why EVERY page owes it
/// A mark stamps the page its extent STARTS in, and an extent may run to
/// [`vitaslop_transpiler::DIRTY_PAGE_BYTES`] (see `emit::plan_dirty_run`), so a write recorded
/// against page `p` can reach into `p + 1`. A page is therefore dirty when its OWN stamp
/// qualifies or its predecessor's does. Applying that to the first page alone leaves a hole in
/// the middle of any range: a write straddling a boundary INSIDE it stamps the lower page, the
/// upper one reads clean, and the returned run stops short of bytes the guest has overwritten.
/// The one-bit [`GuestMemory::dirty_since`] never had that hole - it folds the whole range,
/// overhang included, into one answer - which is why this only ever went wrong in the
/// finer-grained form.
pub fn dirty_runs_from_pages(
    first: usize,
    last: usize,
    off: usize,
    len: usize,
    stamp: u8,
    stamp_of: impl Fn(usize) -> u8,
    out: &mut Vec<(usize, usize)>,
) {
    let shift = vitaslop_transpiler::DIRTY_SHIFT;
    let page_bytes = 1usize << shift;
    let mut run: Option<(usize, usize)> = None;
    for page in first..=last {
        let below = page.checked_sub(1).is_some_and(|p| stamp_of(p) >= stamp);
        if stamp_of(page) < stamp && !below {
            if let Some(r) = run.take() {
                out.push(r);
            }
            continue;
        }
        // This page, clipped to the range: the first starts part way in and the last ends
        // part way through.
        let start = (page << shift).max(off) - off;
        let end = ((page << shift) + page_bytes).min(off + len) - off;
        match run.as_mut() {
            Some(r) => r.1 = end,
            None => run = Some((start, end)),
        }
    }
    if let Some(r) = run {
        out.push(r);
    }
}

/// The native/zero-copy backing: guest memory the runtime can borrow directly
/// (wasmtime linear memory, or any in-process buffer). A `Sized` newtype so it
/// can coerce to `&mut dyn GuestMemory` (an unsized `[u8]` cannot back a trait
/// object). Wrap the engine's rebased slice: `SliceMemory(bytes)`.
pub struct SliceMemory<'a>(pub &'a mut [u8]);

impl GuestMemory for SliceMemory<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
    /// No map, so nothing to stamp.
    fn stamp_dirty_pages(&self, _off: usize, _len: usize) {}
    fn read(&self, off: usize, buf: &mut [u8]) {
        buf.copy_from_slice(&self.0[off..off + buf.len()]);
    }
    fn write(&mut self, off: usize, bytes: &[u8]) {
        self.0[off..off + bytes.len()].copy_from_slice(bytes);
    }
    fn borrow(&self, off: usize, len: usize) -> Option<&[u8]> {
        self.0.get(off..off.checked_add(len)?)
    }
}

/// A read-through overlay over a guest memory: a table of ranges fetched in ONE crossing and
/// a copy of the dirty map, so a host path that reads thousands of guest ranges pays two
/// boundary crossings instead of thousands.
///
/// # Why
/// MEASURED on a phone (the target device): `sceGxmDisplayQueueAddEntry` at 3.5-10.4 ms ONCE
/// A FRAME with 143-2,847 bulk guest reads inside it - `resolve_deferred_geometry` reading
/// every deferred draw's indices, vertex rows and uniform windows through `copy_range`, one
/// JavaScript call each, on a device where a crossing costs ~20x the desktop's. The desktop
/// hides the same count inside 12 us. This is the GENERIC answer the per-call inline forms
/// are not: it is per-frame infrastructure, and every title's every draw goes through it.
///
/// # What it is not
/// Not a cache: the ranges are whatever the caller asked for, served by borrow, and the
/// overlay is dropped at the end of the call. Reads outside the table fall through to the
/// underlying memory exactly as before, so a range the caller did not predict costs what it
/// always cost and nothing reads wrong. Writes are refused (the paths that use this only
/// read), loudly, so a future caller that writes through it cannot lose a store.
pub struct PrefetchedMemory<'m> {
    base: &'m dyn GuestMemory,
    /// `(off, len, start in data)`, sorted by `off`.
    table: Vec<(usize, usize, usize)>,
    data: Vec<u8>,
    stamps: Option<(Vec<u8>, u32)>,
}

impl<'m> PrefetchedMemory<'m> {
    /// Fetch `ranges` (any order, overlaps allowed) in one crossing and copy the dirty map.
    pub fn new(base: &'m dyn GuestMemory, ranges: &[(usize, usize)]) -> Self {
        let mut me = PrefetchedMemory {
            base,
            table: Vec::new(),
            data: Vec::new(),
            stamps: base.dirty_stamps(),
        };
        me.extend(ranges);
        me
    }

    /// Fetch `ranges` in one crossing and take NO copy of the dirty map: for a caller that only
    /// reads bytes (one draw's uniform banks), where copying the whole map per call would cost
    /// far more than the crossings it saves. `dirty_since` then asks the base, as before.
    pub fn reads_only(base: &'m dyn GuestMemory, ranges: &[(usize, usize)]) -> Self {
        let mut me = PrefetchedMemory {
            base,
            table: Vec::new(),
            data: Vec::new(),
            stamps: None,
        };
        me.extend(ranges);
        me
    }

    /// Fetch more ranges (a second stage whose addresses depend on the first), one crossing.
    /// Nothing is fetched over a DIRECT memory (see [`GuestMemory::direct`]): a read there is
    /// already a load, and the table would only copy bytes that `borrow` hands out for free.
    pub fn extend(&mut self, ranges: &[(usize, usize)]) {
        if self.base.direct() {
            return;
        }
        let ranges: Vec<(usize, usize)> = ranges.iter().copied().filter(|&(_, l)| l > 0).collect();
        if ranges.is_empty() {
            return;
        }
        let mut more = Vec::new();
        self.base.read_many(&ranges, &mut more);
        let mut start = self.data.len();
        self.data.extend_from_slice(&more);
        for &(off, len) in &ranges {
            self.table.push((off, len, start));
            start += len;
        }
        self.table
            .sort_unstable_by_key(|&(off, len, _)| (off, usize::MAX - len));
    }

    /// The prefetched bytes covering `[off, off + len)` entirely, if any range does.
    fn find(&self, off: usize, len: usize) -> Option<&[u8]> {
        // The last table entry starting at or before `off`, then walk back over entries
        // that start at the same or earlier offsets until one covers the whole span.
        let mut i = self.table.partition_point(|&(o, _, _)| o <= off);
        while i > 0 {
            i -= 1;
            let (o, l, st) = self.table[i];
            if o + l >= off + len {
                return Some(&self.data[st + (off - o)..st + (off - o) + len]);
            }
            // Entries are sorted by offset; once an entry starts more than the longest
            // plausible range before `off` there is nothing to find. Keep it simple: scan.
        }
        None
    }

    /// How many bytes the table holds - for the caller's own accounting.
    pub fn bytes(&self) -> usize {
        self.data.len()
    }
}

impl GuestMemory for PrefetchedMemory<'_> {
    fn len(&self) -> usize {
        self.base.len()
    }
    fn direct(&self) -> bool {
        self.base.direct()
    }
    fn read(&self, off: usize, buf: &mut [u8]) {
        match self.find(off, buf.len()) {
            Some(s) => buf.copy_from_slice(s),
            None => self.base.read(off, buf),
        }
    }
    fn write(&mut self, _off: usize, _bytes: &[u8]) {
        panic!(
            "PrefetchedMemory is read-only: a host path that writes guest memory must not run under the prefetch overlay"
        );
    }
    fn borrow(&self, off: usize, len: usize) -> Option<&[u8]> {
        self.find(off, len).or_else(|| self.base.borrow(off, len))
    }
    fn dirty_since(&self, off: usize, len: usize, stamp: u8) -> Option<bool> {
        let Some((stamps, shift)) = &self.stamps else {
            return self.base.dirty_since(off, len, stamp);
        };
        if len == 0 {
            return Some(false);
        }
        // The same window the browser's own check reads: one page BELOW the range too, because
        // a translated store stamps only the page it starts in and may reach 8 bytes into the
        // next.
        let first = (off >> shift).saturating_sub(1);
        let last = ((off + len - 1) >> shift).min(stamps.len().saturating_sub(1));
        Some(stamps[first.min(last)..=last].iter().any(|&s| s >= stamp))
    }
    fn dirty_runs_since(
        &self,
        off: usize,
        len: usize,
        stamp: u8,
        out: &mut Vec<(usize, usize)>,
    ) -> Option<()> {
        self.base.dirty_runs_since(off, len, stamp, out)
    }
    fn bump_dirty_epoch(&self) -> Option<(u8, bool)> {
        self.base.bump_dirty_epoch()
    }
    /// Reaches the BASE map only, not this overlay's copy of it. That is sound because the one
    /// caller arms the scene's epoch - which is where stacks are stamped
    /// ([`TextureSnapshots::arm_scene_stamp`]) - BEFORE it takes the copy, and a scene arms once.
    fn stamp_dirty_pages(&self, off: usize, len: usize) {
        self.base.stamp_dirty_pages(off, len)
    }
    fn dirty_epoch(&self) -> Option<u8> {
        self.base.dirty_epoch()
    }
    fn rebase_dirty_epoch(&self, floor: u8) -> Option<u8> {
        self.base.rebase_dirty_epoch(floor)
    }
}

#[cfg(test)]
mod prefetch_tests {
    use super::*;
    use std::cell::Cell;

    /// A memory across a boundary: every `read` and every `read_many` is one crossing, and it
    /// counts whether anyone copied its dirty map.
    struct Remote {
        bytes: Vec<u8>,
        crossings: Cell<u32>,
        stamp_copies: Cell<u32>,
    }

    impl GuestMemory for Remote {
        fn len(&self) -> usize {
            self.bytes.len()
        }
        fn read(&self, off: usize, buf: &mut [u8]) {
            self.crossings.set(self.crossings.get() + 1);
            buf.copy_from_slice(&self.bytes[off..off + buf.len()]);
        }
        fn write(&mut self, _: usize, _: &[u8]) {}
        fn stamp_dirty_pages(&self, _: usize, _: usize) {}
        fn read_many(&self, ranges: &[(usize, usize)], out: &mut Vec<u8>) {
            self.crossings.set(self.crossings.get() + 1);
            out.clear();
            for &(o, l) in ranges {
                out.extend_from_slice(&self.bytes[o..o + l]);
            }
        }
        fn dirty_stamps(&self) -> Option<(Vec<u8>, u32)> {
            self.stamp_copies.set(self.stamp_copies.get() + 1);
            Some((vec![0; 4], 12))
        }
    }

    /// A draw's uniform banks: fetched in ONE crossing, read back from it exactly (whole ranges
    /// and pieces of them), anything not fetched still read from the memory - and no copy of
    /// the dirty map, which per draw would cost more than the crossings saved.
    #[test]
    fn a_reads_only_prefetch_serves_its_ranges_from_one_crossing() {
        let mem = Remote {
            bytes: (0..=255u8).cycle().take(4096).collect(),
            crossings: Cell::new(0),
            stamp_copies: Cell::new(0),
        };
        let pm = PrefetchedMemory::reads_only(&mem, &[(100, 64), (1000, 32)]);
        assert_eq!((mem.crossings.get(), mem.stamp_copies.get()), (1, 0));
        let mut a = [0u8; 64];
        pm.read(100, &mut a);
        assert_eq!(a[..], mem.bytes[100..164]);
        let mut b = [0u8; 8];
        pm.read(1010, &mut b);
        assert_eq!(b[..], mem.bytes[1010..1018]);
        assert_eq!(pm.borrow(1000, 32), Some(&mem.bytes[1000..1032]));
        assert_eq!(
            mem.crossings.get(),
            1,
            "every fetched byte came out of the one crossing"
        );
        pm.read(2000, &mut b);
        assert_eq!(b[..], mem.bytes[2000..2008]);
        assert_eq!(
            mem.crossings.get(),
            2,
            "a range not fetched is read from the memory"
        );
    }
}

/// A borrowed view of guest state for the duration of one host call: the
/// register file, guest memory (rebased so guest address `A` is byte `A - base`),
/// and a sequential AAPCS argument cursor.
pub struct GuestCtx<'a> {
    pub regs: &'a mut [u32; REG_COUNT],
    /// Raw bits of the VFP single-precision registers s0..s15 (the float argument
    /// and return file). d`n` is (s[2n], s[2n+1]).
    pub vfp: &'a mut [u32; VFP_ARG_COUNT],
    pub mem: &'a mut dyn GuestMemory,
    pub base: u32,
    /// Next core (integer/pointer) argument slot: args 0..3 are r0..r3, args >=4
    /// are on the stack at sp + (n-4)*4.
    next_core: usize,
    /// Next VFP single-register slot for a float argument (AAPCS-VFP keeps a
    /// separate counter from the core one; doubles align to an even slot).
    next_vfp: usize,
}

impl<'a> GuestCtx<'a> {
    pub(crate) fn new(
        regs: &'a mut [u32; REG_COUNT],
        vfp: &'a mut [u32; VFP_ARG_COUNT],
        mem: &'a mut dyn GuestMemory,
        base: u32,
    ) -> Self {
        GuestCtx {
            regs,
            vfp,
            mem,
            base,
            next_core: 0,
            next_vfp: 0,
        }
    }

    /// Read positional integer/pointer argument `n` (AAPCS: r0..r3 then stack).
    pub fn arg(&self, n: usize) -> u32 {
        if n < 4 {
            self.regs[n]
        } else {
            let sp = self.regs[SP];
            self.read_u32(sp.wrapping_add(((n - 4) * 4) as u32))
        }
    }

    /// Read the next core integer/pointer argument and advance the core cursor.
    pub fn next_u32(&mut self) -> u32 {
        let v = self.arg(self.next_core);
        self.next_core += 1;
        v
    }

    /// Read the next float argument (a single VFP register) and advance the VFP
    /// cursor. Used by `#[hostcall]` for `f32` parameters.
    pub fn next_f32(&mut self) -> f32 {
        let i = self.next_vfp;
        self.next_vfp += 1;
        let bits = self.vfp.get(i).copied().unwrap_or(0);
        f32::from_bits(bits)
    }

    /// Read the next double argument (an even-aligned VFP register pair) and
    /// advance the VFP cursor past it. Used by `#[hostcall]` for `f64` parameters.
    pub fn next_f64(&mut self) -> f64 {
        // Doubles occupy an even-aligned single-register pair (d`k` = s[2k],
        // s[2k+1]); align up to the next even slot, no back-fill.
        let i = (self.next_vfp + 1) & !1;
        self.next_vfp = i + 2;
        let lo = self.vfp.get(i).copied().unwrap_or(0) as u64;
        let hi = self.vfp.get(i + 1).copied().unwrap_or(0) as u64;
        f64::from_bits(lo | (hi << 32))
    }

    /// Set the call's float return value in s0.
    pub fn ret_f32(&mut self, v: f32) {
        self.vfp[0] = v.to_bits();
    }

    /// Set the call's double return value in d0 (s0 low, s1 high).
    pub fn ret_f64(&mut self, v: f64) {
        let bits = v.to_bits();
        self.vfp[0] = bits as u32;
        self.vfp[1] = (bits >> 32) as u32;
    }

    /// The offset into guest memory for guest address `addr`, or None if outside
    /// the provisioned region.
    pub(super) fn offset(&self, addr: u32) -> Option<usize> {
        let off = addr.checked_sub(self.base)? as usize;
        if off <= self.mem.len() {
            Some(off)
        } else {
            None
        }
    }

    /// Does `addr` fall inside the guest's linear address space at all?
    ///
    /// Not "is the whole range readable" - [`Self::read_into`] already clamps and zero-fills
    /// for that. This answers the one question a caller merging several reads into one has to
    /// ask: an address BELOW the base maps to nothing, and a merged read starting there would
    /// hand back zeros where the individual reads would have found data. See
    /// `TextureSnapshots::gather_into_scratch`.
    pub fn maps(&self, addr: u32) -> bool {
        self.offset(addr).is_some()
    }

    /// Read a little-endian u32 at guest address `addr` (0 if out of range).
    pub fn read_u32(&self, addr: u32) -> u32 {
        crate::perf::note_word_read();
        match self.offset(addr) {
            Some(o) if o + 4 <= self.mem.len() => self.mem.read_u32(o),
            _ => 0,
        }
    }

    /// Write a little-endian u32 at guest address `addr` (ignored if out of range).
    pub fn write_u32(&mut self, addr: u32, v: u32) {
        report_host_write(self, addr, 4, &v.to_le_bytes());
        if let Some(o) = self.offset(addr)
            && o + 4 <= self.mem.len()
        {
            self.mem.write_u32(o, v);
        }
    }

    /// Read `len` bytes at guest address `addr` (short read clamped to range).
    pub fn read_bytes(&self, addr: u32, len: usize) -> Vec<u8> {
        crate::perf::note_bulk_read();
        match self.offset(addr) {
            Some(o) => {
                let n = (o + len).min(self.mem.len()) - o;
                // Borrow-and-clone where the backing allows it: `to_vec` allocates
                // and copies once, where allocating a zeroed buffer and reading into
                // it writes every byte TWICE. This path moves megabytes a frame
                // (vertex, index and texture snapshots), so the second pass is real.
                if let Some(s) = self.mem.borrow(o, n) {
                    return s.to_vec();
                }
                let mut buf = vec![0u8; n];
                self.mem.read(o, &mut buf);
                buf
            }
            None => Vec::new(),
        }
    }

    /// Read `len` bytes at guest address `addr` straight into a shared buffer (short read
    /// clamped to range, exactly as [`read_bytes`](Self::read_bytes)).
    ///
    /// # Why this exists beside `read_bytes`
    /// Every snapshot path used to do `read_bytes` (guest bytes -> `Vec`) and then
    /// `Arc::from(vec)` - which COPIES again, because an `Arc<[u8]>`'s layout is not a
    /// `Vec`'s. On the engine that can lend its bytes that was two passes where one
    /// suffices; on the browser - which cannot lend, so the read itself is a crossing into
    /// a zeroed staging buffer - it was three. The snapshot paths move megabytes a frame
    /// (`draw: read+interleave vertices` alone is ~6 MB on a retail sports title), so the
    /// extra pass is real. This reads once into the `Arc`'s own allocation.
    pub fn read_bytes_arc(&self, addr: u32, len: usize) -> Arc<[u8]> {
        crate::perf::note_bulk_read();
        let Some(o) = self.offset(addr) else {
            return Arc::from(&[][..]);
        };
        let n = (o + len).min(self.mem.len()) - o;
        if let Some(s) = self.mem.borrow(o, n) {
            // One copy: straight from the lent slice into the Arc's allocation.
            return Arc::from(s);
        }
        // Zeroed rather than uninit so `assume_init` below is unconditionally sound for
        // `u8`; the memset is a fraction of the boundary copy it precedes.
        let arc = unsafe {
            // SAFETY: a zeroed byte is an initialised byte.
            std::sync::Arc::<[std::mem::MaybeUninit<u8>]>::assume_init(
                std::sync::Arc::new_zeroed_slice(n),
            )
        };
        let mut arc = arc;
        let buf = Arc::get_mut(&mut arc).expect("freshly created, uniquely owned");
        self.mem.read(o, buf);
        arc
    }

    /// Read guest memory at `addr` into `buf`, filling the tail with zeros if the
    /// range runs past the provisioned region. The allocation-free counterpart of
    /// [`read_bytes`](Self::read_bytes), for a caller with a buffer to reuse - which is
    /// what a per-draw or per-parameter read must be, since the allocator is otherwise
    /// the dominant cost of reading a few dozen bytes.
    pub fn read_into(&self, addr: u32, buf: &mut [u8]) {
        crate::perf::note_bulk_read();
        let n = match self.offset(addr) {
            Some(o) => {
                let n = buf.len().min(self.mem.len().saturating_sub(o));
                self.mem.read(o, &mut buf[..n]);
                n
            }
            None => 0,
        };
        // Only the TAIL that guest memory did not cover is zeroed - zeroing the whole
        // buffer first would write every byte twice on the common in-range path.
        buf[n..].fill(0);
    }

    /// Read `count` consecutive little-endian f32s starting at `addr`.
    ///
    /// One bounds check and one block copy, rather than `count` calls through the
    /// `dyn GuestMemory` vtable. A default uniform buffer is hundreds of floats and is
    /// read on every draw, so the per-element version showed up as a per-draw cost.
    pub fn read_f32s(&self, addr: u32, count: usize) -> Vec<f32> {
        // Borrow and convert straight into the result: no staging buffer, and the
        // result is written once. A default uniform buffer is hundreds of floats and
        // is read on every draw.
        if let Some(o) = self.offset(addr)
            && let Some(s) = self.mem.borrow(o, count * 4)
        {
            return s
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
        }
        let mut raw = vec![0u8; count * 4];
        self.read_into(addr, &mut raw);
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Borrow `len` bytes of guest memory at `addr` in place, without copying.
    /// `None` when the range is out of bounds or the backing cannot lend
    /// ([`GuestMemory::borrow`]).
    pub fn borrow_bytes(&self, addr: u32, len: usize) -> Option<&[u8]> {
        crate::perf::note_bulk_read();
        self.mem.borrow(self.offset(addr)?, len)
    }

    /// Has the guest stored anywhere in `[addr, addr + len)` at or after epoch
    /// `stamp`? `None` when the engine tracks no stores, which is not an answer -
    /// see [`GuestMemory::dirty_since`], which this forwards to.
    pub fn dirty_since(&self, addr: u32, len: usize, stamp: u8) -> Option<bool> {
        self.mem.dirty_since(self.offset(addr)?, len, stamp)
    }

    /// Which parts of `[addr, addr + len)` the guest may have stored into since `stamp`,
    /// as byte offsets relative to `addr` - see [`GuestMemory::dirty_runs_since`].
    pub fn dirty_runs_since(
        &self,
        addr: u32,
        len: usize,
        stamp: u8,
        out: &mut Vec<(usize, usize)>,
    ) -> Option<()> {
        self.mem
            .dirty_runs_since(self.offset(addr)?, len, stamp, out)
    }

    /// Advance the guest-store epoch - see [`GuestMemory::bump_dirty_epoch`].
    pub fn bump_dirty_epoch(&self) -> Option<(u8, bool)> {
        self.mem.bump_dirty_epoch()
    }

    /// Stamp `[addr, addr + len)` with the current epoch - see [`GuestMemory::stamp_dirty_pages`].
    pub fn stamp_dirty_pages(&self, addr: u32, len: usize) {
        if let Some(o) = self.offset(addr) {
            self.mem.stamp_dirty_pages(o, len);
        }
    }

    /// The epoch a store made now would carry - see [`GuestMemory::dirty_epoch`].
    pub fn dirty_epoch(&self) -> Option<u8> {
        self.mem.dirty_epoch()
    }

    /// Renumber the epoch against `floor` - see [`GuestMemory::rebase_dirty_epoch`]. The
    /// caller owes the same subtraction on every stamp it holds.
    pub fn rebase_dirty_epoch(&self, floor: u8) -> Option<u8> {
        self.mem.rebase_dirty_epoch(floor)
    }

    /// Write `bytes` at guest address `addr` (clamped to range).
    pub fn write_bytes(&mut self, addr: u32, bytes: &[u8]) {
        report_host_write(self, addr, bytes.len() as u32, bytes);
        if let Some(o) = self.offset(addr) {
            let end = (o + bytes.len()).min(self.mem.len());
            self.mem.write(o, &bytes[..end - o]);
        }
    }

    /// Read a little-endian f32 at guest address `addr`.
    pub fn read_f32(&self, addr: u32) -> f32 {
        f32::from_bits(self.read_u32(addr))
    }

    /// Read a little-endian u16 at guest address `addr` (0 if out of range).
    pub fn read_u16(&self, addr: u32) -> u16 {
        crate::perf::note_word_read();
        match self.offset(addr) {
            Some(o) if o + 2 <= self.mem.len() => self.mem.read_u16(o),
            _ => 0,
        }
    }

    /// Read a NUL-terminated ASCII string at guest address `addr` (bounded).
    pub fn read_cstr(&self, addr: u32, max: usize) -> String {
        let bytes = self.read_bytes(addr, max);
        let n = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        String::from_utf8_lossy(&bytes[..n]).into_owned()
    }

    /// Set the call's return value (r0).
    pub fn ret(&mut self, v: u32) {
        self.regs[0] = v;
    }
}

/// The ordinary way to reach guest-resident state: through the call in flight, so the
/// write watches and the range clamp all apply exactly as they do to any other host write.
impl GuestWords for GuestCtx<'_> {
    fn word(&self, addr: u32) -> u32 {
        self.read_u32(addr)
    }
    fn set_word(&mut self, addr: u32, value: u32) {
        self.write_u32(addr, value);
    }
    fn cas_word(&mut self, addr: u32, expect: u32, new: u32) -> u32 {
        match self.offset(addr) {
            Some(o) if o + 4 <= self.mem.len() => {
                let cur = self.mem.cas_u32(o, expect, new);
                if cur == expect {
                    report_host_write(self, addr, 4, &new.to_le_bytes());
                }
                cur
            }
            _ => 0,
        }
    }
}

/// `VITASLOP_HOST_WRITE_WATCH=<hex addr>[,...]`: report every write a HOST CALL makes to one
/// of those guest addresses, with the bytes and the guest return address that asked for it.
///
/// This is the counterpart `VITASLOP_WATCH_STORE` cannot be. That watchpoint traps the guest's
/// own stores; a host call writing a guest buffer on the guest's behalf - a file read, an
/// out-parameter, `sceGxmSetUniformDataF` into a game-owned block - leaves no store to trap.
/// "No guest store ever writes that address" then reads as "it must be static data", and a
/// whole session can go looking for a file that does not exist. Both watchpoints exist because
/// each is blind exactly where the other sees (memory `vitaslop-host-call-reference-semantics`).
/// An entry may also be `v:<hex word>`, which matches on the VALUE written anywhere rather
/// than on an address. When the question is "where did this number come from" the address is
/// often the thing you do not have: a material's uniform block is heap, so its address moves
/// between runs, while the suspicious 32-bit pattern in it is the same every time.
/// An entry may also be a RANGE, `<lo hex>-<hi hex>` (inclusive): a corrupted record inside a
/// guest table is usually one of thousands, and which one is the thing not yet known.
fn report_host_write(ctx: &GuestCtx, addr: u32, len: u32, bytes: &[u8]) {
    use std::sync::OnceLock;
    let parse = |hex: &str| u32::from_str_radix(hex.trim().trim_start_matches("0x"), 16).ok();
    static WATCH: OnceLock<(Vec<(u32, u32)>, Vec<u32>)> = OnceLock::new();
    let (addrs, vals) = WATCH.get_or_init(|| {
        let (mut a, mut v) = (Vec::new(), Vec::new());
        for t in crate::knobs::var("VITASLOP_HOST_WRITE_WATCH")
            .unwrap_or_default()
            .split(',')
        {
            let t = t.trim();
            if let Some(rest) = t.strip_prefix("v:") {
                v.extend(parse(rest));
            } else if let Some((lo, hi)) = t.split_once('-') {
                if let (Some(lo), Some(hi)) = (parse(lo), parse(hi)) {
                    a.push((lo.min(hi), lo.max(hi)));
                }
            } else if let Some(n) = parse(t) {
                a.push((n, n));
            }
        }
        (a, v)
    });
    if addrs.is_empty() && vals.is_empty() {
        return;
    }
    // SAY THAT IT ARMED. A silent run is this watchpoint's output both when no host call ever
    // writes the address AND when the knob never parsed - and those two readings send an
    // investigation in opposite directions. The store watchpoint on the other side of this
    // pair has said so since it was written; this one relied on its silence being trusted,
    // which is the failure mode the whole family exists to avoid.
    // [[vitaslop-instrument-failure-imitating-its-subject]]
    {
        use std::sync::OnceLock;
        static SAID: OnceLock<()> = OnceLock::new();
        SAID.get_or_init(|| {
            // STATUS, not stderr: a browser has no stderr (`eprintln!` there reaches nothing).
            tracing::info!(
                target: "vitaslop::status",
                "host write watch: ARMED on {} address(es) and {} value(s) - a run with no \
                 further `host write watch` line means NO HOST CALL wrote them. That says \
                 nothing about the guest's own stores; VITASLOP_WATCH_STORE is that half.",
                addrs.len(),
                vals.len()
            );
        });
    }
    let end = addr.wrapping_add(len.max(1)).wrapping_sub(1);
    let hit = addrs
        .iter()
        .copied()
        .find(|&(lo, hi)| lo <= end && hi >= addr)
        .map(|(lo, _)| lo.max(addr))
        .or_else(|| {
            (!vals.is_empty())
                .then(|| {
                    bytes
                        .chunks_exact(4)
                        .position(|c| vals.contains(&u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                })
                .flatten()
                .map(|i| addr + i as u32 * 4)
        });
    let Some(hit) = hit else { return };
    // The bytes AT the watched word, not the whole (possibly megabyte) write: what the question
    // is about is the value that ends up there.
    let off = (hit.wrapping_sub(addr)) as usize & !3;
    let word = bytes
        .get(off..off + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .unwrap_or(0);
    let head: String = bytes.iter().take(32).map(|b| format!("{b:02x}")).collect();
    tracing::info!(
        target: "vitaslop::status",
        "host write watch: a HOST CALL wrote {len} bytes at {addr:#x} covering {hit:#x}, leaving          {word:#010x} there, with the guest at lr={:#010x} pc={:#010x} f{} bytes {head}",
        ctx.regs[14], ctx.regs[15], crate::sched::current_frame()
    );
}
