use super::*;

impl VitaState {
    /// Mint a fresh SceUID (for a mutex, semaphore, event flag, ...).
    pub fn new_uid(&mut self) -> i32 {
        let uid = self.next_uid;
        self.next_uid += 1;
        uid
    }

    /// Create a timer, returning its SceUID. Timers are created STOPPED with a count of
    /// zero, which is what `sceKernelCreateTimer` does: `sceKernelStartTimer` is a
    /// separate call and a title that reads the time without starting must see nothing.
    pub fn create_timer(&mut self, name: &str, attr: u32) -> i32 {
        let uid = self.new_uid();
        self.timers.create(uid, name, attr);
        uid
    }

    /// The SceUID of an existing timer with this name, for `sceKernelOpenTimer`.
    #[inline]
    pub fn timer_by_name(&self, name: &str) -> Option<i32> {
        self.timers.by_name(name)
    }

    /// Start a timer counting. Returns whether it was found, and whether it had ALREADY
    /// been started - the caller reports that rather than silently restarting, since a
    /// double start would otherwise lose however long the timer had already run.
    #[inline]
    pub fn timer_start(&mut self, uid: i32) -> Option<bool> {
        self.timers.start(uid, self.virtual_us)
    }

    /// Stop a timer, banking what it has counted so far.
    #[inline]
    pub fn timer_stop(&mut self, uid: i32) -> Option<bool> {
        self.timers.stop(uid, self.virtual_us)
    }

    /// A timer's current count in microseconds: what it banked before it was last
    /// stopped, plus the time since it was last started if it is running.
    #[inline]
    pub fn timer_time_us(&self, uid: i32) -> Option<u64> {
        self.timers.time_us(uid, self.virtual_us)
    }

    /// Set a timer's count, keeping it running if it was.
    #[inline]
    pub fn timer_set_time_us(&mut self, uid: i32, us: u64) -> Option<u64> {
        self.timers.set_time_us(uid, us, self.virtual_us)
    }

    /// Forget a timer (`sceKernelDeleteTimer`). Returns whether it existed.
    #[inline]
    pub fn timer_delete(&mut self, uid: i32) -> bool {
        self.timers.delete(uid)
    }

    /// The guest's name for a timer, for reports.
    #[inline]
    pub fn timer_name(&self, uid: i32) -> &str {
        self.timers.name(uid)
    }

    /// A timer's attribute word, as it was created with.
    #[inline]
    pub fn timer_attr(&self, uid: i32) -> u32 {
        self.timers.attr(uid)
    }

    /// Whether a timer is currently counting.
    #[inline]
    pub fn timer_running(&self, uid: i32) -> bool {
        self.timers.running(uid)
    }

    /// A shader-patcher program was created: it now holds one reference. See
    /// [`ProgramRefs`] for why references are counted.
    #[inline]
    pub fn note_program_created(&mut self, handle: u32) {
        self.program_refs.created(handle)
    }

    /// A program was released: drop one reference, forgetting it at zero. Returns
    /// whether the handle was known, so a release of something never created can be
    /// told apart from an ordinary one.
    #[inline]
    pub fn note_program_released(&mut self, handle: u32) -> bool {
        self.program_refs.released(handle)
    }

    /// A live program created with exactly these parameters, taking one more reference to it.
    /// See `gxm::create_vertex_program`.
    #[inline]
    pub fn share_program(&mut self, key: &(u32, Vec<u8>)) -> Option<u32> {
        self.program_refs.share(key)
    }

    /// Record a freshly created program under its creation parameters. See [`Self::share_program`].
    #[inline]
    pub fn remember_shared_program(&mut self, key: (u32, Vec<u8>), handle: u32) {
        self.program_refs.remember(key, handle)
    }

    /// How many references a live program holds, or `None` if it is not one of ours.
    #[inline]
    pub fn program_ref_count(&self, handle: u32) -> Option<u32> {
        self.program_refs.count(handle)
    }

    /// Guest-region bytes reserved at the top for the MAIN THREAD'S STACK, which the
    /// heap may never hand out.
    ///
    /// The main thread's stack pointer starts near the top of the region and grows
    /// DOWN, while this bump allocator grows UP; nothing but this reservation keeps
    /// them apart. Capping only at the region ceiling is not enough, and the failure is
    /// vicious: the heap marches into live stack frames and the next perfectly ordinary
    /// guest write - a `memcpy` into a buffer we just handed out - silently overwrites a
    /// caller's saved registers. The crash then lands in an innocent function, frames
    /// and often minutes later, with a callee-saved register holding a float.
    ///
    /// The reserve covers the host's startup headroom above the initial SP plus room
    /// for real stack depth beneath it.
    pub(super) const MAIN_STACK_RESERVE_BYTES: u32 = 5 * 1024 * 1024;

    /// Allocate `size` bytes of real guest memory aligned to `align`, returning
    /// the guest address (0 on exhaustion). Deterministic: a pure function of the
    /// allocation order. The bump cursor is capped below the guest region ceiling
    /// (`base + mem_bytes`) by [`Self::MAIN_STACK_RESERVE_BYTES`), for two reasons: the
    /// indirect-dispatch address table lives immediately above the ceiling (a stray
    /// write there corrupts the table and faults every later indirect call), and the
    /// main thread's stack descends through the top of the region itself. An allocation
    /// that would reach either is a real out-of-memory: return 0 - loudly - rather than
    /// a pointer that aliases something live.
    #[inline]
    pub fn galloc(&mut self, size: u32, align: u32) -> u32 {
        let ceiling = self.heap_ceiling();
        self.arena.galloc(size, align, ceiling)
    }

    /// The address the heap may not reach: the guest region ceiling (`base + mem_bytes`)
    /// less [`Self::MAIN_STACK_RESERVE_BYTES`] - see [`Self::galloc`].
    #[inline]
    pub(super) fn heap_ceiling(&self) -> u32 {
        self.base
            .wrapping_add(self.mem_bytes)
            .wrapping_sub(Self::MAIN_STACK_RESERVE_BYTES)
    }

    /// The UID of the memory block that fully contains `[addr, addr + size)`, or
    /// `None`. A zero `size` asks about the single byte at `addr`, which is how a
    /// caller with only a pointer phrases the question.
    #[inline]
    pub fn memblock_containing(&self, addr: u32, size: u32) -> Option<i32> {
        self.arena.memblock_containing(addr, size)
    }

    /// Allocate a memory block of `size` and record it, returning its SceUID, or 0 when
    /// the arena cannot satisfy it.
    ///
    /// A freed block is reused before the bump cursor moves (first fit over the
    /// release-ordered free list, so the choice is deterministic). Without that, a title
    /// that cycles screens - freeing one screen's buffers and allocating the next one's -
    /// grows the arena forever and eventually gets a null allocation for a request the
    /// console would have satisfied from the memory it just handed back.
    ///
    /// Returning 0 on exhaustion matters as much as the reuse: the caller turns it into a
    /// real `sceKernelAllocMemBlock` error, where handing back a live SceUID whose base is
    /// 0 is a hollow success the guest cannot detect until it writes through the pointer.
    pub fn alloc_memblock(&mut self, size: u32, align: u32, ty: u32) -> i32 {
        let ceiling = self.heap_ceiling();
        self.arena
            .alloc_memblock(size, align, ty, ceiling, &mut self.next_uid)
    }

    /// The block containing `addr`, as `(base, size, type)`, for
    /// `sceKernelGetMemBlockInfoByAddr`. The address may be anywhere inside the block -
    /// that is the whole point of the query, which a title reaches with a pointer it was
    /// handed rather than with a base.
    #[inline]
    pub fn memblock_info_at(&self, addr: u32) -> Option<(u32, u32, u32)> {
        self.arena.memblock_info_at(addr)
    }

    /// The base address of the block with SceUID `uid`, if known.
    #[inline]
    pub fn memblock_base(&self, uid: i32) -> Option<u32> {
        self.arena.memblock_base(uid)
    }

    /// Release a memory block by SceUID (`sceKernelFreeMemBlock`). Returns true if a
    /// block was registered under `uid`. Its bytes go back to the arena's free list
    /// (coalesced with any neighbouring hole), so a later allocation can reuse them, and
    /// a later `sceKernelGetMemBlockBase(uid)` no longer resolves it, matching the guest-
    /// visible contract that the id is now invalid.
    pub fn free_memblock(&mut self, uid: i32) -> bool {
        let Some((base, size)) = self.arena.free_memblock(uid) else {
            return false;
        };
        // Drop any texture snapshot over the released memory. Now that the block's
        // address CAN come back from [`Self::alloc_memblock`], this invalidation is
        // load-bearing rather than merely tidy: a stale snapshot over reused memory
        // would render the previous screen's pixels into the next one's texture.
        snaps!(self).invalidate_range(base, size as usize);
        true
    }

    /// Bytes outstanding in the CDRAM partition. Read by `sceAppMgrGetBudgetInfo`, so the
    /// budget it reports is the one the allocator enforces.
    #[inline]
    pub fn cdram_used(&self) -> u32 {
        self.arena.cdram_used()
    }

    /// Bytes outstanding in the physically-contiguous partition. See [`Self::cdram_used`].
    #[inline]
    pub fn phycont_used(&self) -> u32 {
        self.arena.phycont_used()
    }

    /// Bytes the guest arena can still hand out: the headroom between the bump cursor and
    /// [`Self::galloc`]'s ceiling, plus every released hole the free list would reuse. This
    /// is the ONE pool `sceKernelAllocMemBlock` draws main RAM from, so it is what a
    /// free-memory query has to report for it - a figure the allocator would not honour is
    /// how a title sizes a heap it is then refused.
    #[inline]
    pub fn main_free_bytes(&self) -> u32 {
        self.arena.main_free_bytes(self.heap_ceiling())
    }

    /// Where the bump cursor stands. For diagnostics only - `vitaslop::mem` prints it beside
    /// every block, which is what separates "the arena grew" from "a hole was reused".
    #[inline]
    pub fn alloc_cursor_addr(&self) -> u32 {
        self.arena.cursor()
    }

    /// How many released holes the free list holds, and their total bytes. A large byte
    /// count beside an exhausted arena is fragmentation, not a leak - the two are the same
    /// symptom and need different fixes.
    #[inline]
    pub fn freed_memblock_count(&self) -> usize {
        self.arena.freed_count()
    }

    /// Total bytes held in released holes. See [`Self::freed_memblock_count`].
    #[inline]
    pub fn freed_memblock_bytes(&self) -> u64 {
        self.arena.freed_bytes()
    }
}
