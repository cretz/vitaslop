//! The guest arena: the bump allocator behind every host-side allocation of guest memory,
//! and the `sceKernelAllocMemBlock` registry on top of it - live blocks, the free list that
//! reuses and coalesces released ones, and the CDRAM / physically-contiguous partition budgets.
//!
//! [`GuestArena`] owns all of it; its fields are private to this file. What it needs from the
//! rest of the host is the address the heap may not reach (`ceiling`, computed by
//! `VitaState::heap_ceiling`) and, for a block, the next SceUID. Every answer is a pure
//! function of the guest's own allocation sequence, so two runs of one input lay memory out
//! identically.

/// A GPU memory block the guest allocated: its SceUID, guest base address, and
/// size. Backed by real guest memory so the guest can read and write it.
pub struct MemBlock {
    pub uid: i32,
    pub base: u32,
    pub size: u32,
    /// The `SceKernelMemBlockType` the guest asked for. Kept because
    /// `sceKernelGetMemBlockInfoByAddr` reports it back along with the cacheability and
    /// access rights ENCODED IN IT - deriving those from anything else would let the
    /// query disagree with the request that created the block.
    pub ty: u32,
}

/// The guest arena. See the module docs.
pub(super) struct GuestArena {
    /// The bump cursor: the lowest address no allocation has handed out yet.
    cursor: u32,
    /// Whether `VITASLOP_GALLOC_SKEW` has been applied - see [`GuestArena::galloc`].
    skewed: bool,
    blocks: Vec<MemBlock>,
    /// Bytes currently outstanding in the CDRAM and PHYCONT partitions - the two the
    /// console bounds separately from main RAM. See [`crate::vita::sysmem::partition_of`];
    /// main RAM is not tracked here because it is bounded by the arena itself.
    cdram_used: u32,
    phycont_used: u32,
    /// `(base, size)` of every released memory block, in release order, available for
    /// reuse by [`GuestArena::alloc_memblock`]. Not coalesced on reuse: adjacency in the arena
    /// is not adjacency in usefulness here (blocks are whole buffers a title allocates and
    /// releases as units), and leaving the list in release order keeps the choice of hole
    /// a pure function of the guest's own allocation sequence.
    freed: Vec<(u32, u32)>,
}

impl GuestArena {
    /// An empty arena whose first allocation starts at `cursor`.
    pub(super) fn new(cursor: u32) -> Self {
        GuestArena {
            cursor,
            skewed: false,
            blocks: Vec::new(),
            cdram_used: 0,
            phycont_used: 0,
            freed: Vec::new(),
        }
    }

    /// Allocate `size` bytes aligned to `align` below `ceiling`, returning the guest address
    /// (0 on exhaustion). See `VitaState::galloc`, which supplies the ceiling.
    pub(super) fn galloc(&mut self, size: u32, align: u32, ceiling: u32) -> u32 {
        // `VITASLOP_GALLOC_SKEW=<bytes>`: start the arena that many bytes later, once. Every
        // block the guest is handed - and so its whole heap - moves by that much while not one
        // guest instruction changes, which is what separates a defect that depends on WHERE
        // things are from one that depends on WHEN they happen (an instrumented run changes
        // both at once).
        static SKEW: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        let skew = *SKEW.get_or_init(|| {
            crate::knobs::var("VITASLOP_GALLOC_SKEW")
                .ok()
                .and_then(|v| {
                    let v = v.trim();
                    v.strip_prefix("0x")
                        .map_or_else(|| v.parse().ok(), |h| u32::from_str_radix(h, 16).ok())
                })
                .unwrap_or(0)
        });
        if skew != 0 && !self.skewed {
            self.skewed = true;
            self.cursor = self.cursor.wrapping_add(skew);
        }
        let a = align.max(4);
        let p = (self.cursor + a - 1) & !(a - 1);
        let end = p.wrapping_add(size.max(4));
        if end > ceiling || end < p {
            // Silence here is what turns a leak into memory corruption somewhere else
            // entirely, so an exhausted heap always says so.
            tracing::error!(
                target: "vitaslop::err",
                size,
                align,
                cursor = format_args!("{:#x}", self.cursor),
                ceiling = format_args!("{ceiling:#x}"),
                "guest heap EXHAUSTED - returning a null allocation rather than encroaching \
                 on the main thread stack or the dispatch table"
            );
            return 0;
        }
        self.cursor = end;
        p
    }

    /// Allocate and record a memory block, returning its SceUID - minted from `next_uid`
    /// only on success - or 0. See `VitaState::alloc_memblock`.
    pub(super) fn alloc_memblock(
        &mut self,
        size: u32,
        align: u32,
        ty: u32,
        ceiling: u32,
        next_uid: &mut i32,
    ) -> i32 {
        let want = size.max(4);
        let a = align.max(4);
        // The PARTITION budget first: a CDRAM request that would take more video memory
        // than the console has must fail HERE, whatever the arena has room for. A title
        // sizes its pools by allocating until refused, so an unbounded partition is not a
        // spare resource - it is a wrong answer the title then builds on.
        use crate::vita::sysmem::{
            CDRAM_BUDGET_BYTES, PHYCONT_BUDGET_BYTES, Partition, partition_of,
        };
        let part = partition_of(ty);
        let (used, budget) = match part {
            Partition::Cdram => (self.cdram_used, CDRAM_BUDGET_BYTES),
            Partition::Phycont => (self.phycont_used, PHYCONT_BUDGET_BYTES),
            Partition::Main => (0, u32::MAX),
        };
        if used.saturating_add(want) > budget {
            return 0;
        }
        let reused = self
            .freed
            .iter()
            .position(|&(base, sz)| sz >= want && base & (a - 1) == 0);
        let base = match reused {
            Some(i) => {
                let (base, sz) = self.freed.remove(i);
                // Keep the remainder available rather than rounding the whole hole up to
                // this request; a 4 KiB reuse of a 2 MiB hole must not lose 2 MiB.
                let rest = sz - want;
                if rest >= a {
                    let split = base + want;
                    let aligned = (split + a - 1) & !(a - 1);
                    if aligned < base + sz {
                        self.freed.push((aligned, base + sz - aligned));
                    }
                }
                base
            }
            None => self.galloc(size, align, ceiling),
        };
        if base == 0 {
            return 0;
        }
        match part {
            Partition::Cdram => self.cdram_used += want,
            Partition::Phycont => self.phycont_used += want,
            Partition::Main => {}
        }
        let uid = *next_uid;
        *next_uid += 1;
        self.blocks.push(MemBlock {
            uid,
            base,
            size,
            ty,
        });
        uid
    }

    /// Forget block `uid` and give its bytes back, returning its `(base, size)` - `None` if
    /// no block has that id. The caller owns invalidating anything cached over the range.
    pub(super) fn free_memblock(&mut self, uid: i32) -> Option<(u32, u32)> {
        let i = self.blocks.iter().position(|b| b.uid == uid)?;
        let b = self.blocks.remove(i);
        {
            use crate::vita::sysmem::{Partition, partition_of};
            let want = b.size.max(4);
            match partition_of(b.ty) {
                Partition::Cdram => self.cdram_used = self.cdram_used.saturating_sub(want),
                Partition::Phycont => self.phycont_used = self.phycont_used.saturating_sub(want),
                Partition::Main => {}
            }
        }
        if b.base != 0 {
            // COALESCE with any hole this one touches, repeatedly - a merge can make the
            // result adjacent to a third hole. Without it, a title that frees several
            // neighbouring blocks and then asks for one block their combined size gets a
            // NO_MEMORY the console would never report: MEASURED on a title whose startup
            // probes available RAM by allocating and freeing 128 MB, then 192 MB, then
            // asking for 224 MB - the two holes are ADJACENT and their sum is 320 MB, and
            // the probe still failed at every size, which tells the title it has no memory
            // at all. Merging is what makes a free actually give the bytes back.
            let mut hole = (b.base, b.size.max(4));
            loop {
                let touching = self.freed.iter().position(|&(base, sz)| {
                    base.wrapping_add(sz) == hole.0 || hole.0.wrapping_add(hole.1) == base
                });
                let Some(i) = touching else { break };
                let (base, sz) = self.freed.remove(i);
                let lo = base.min(hole.0);
                hole = (lo, sz + hole.1);
            }
            self.freed.push(hole);
        }
        Some((b.base, b.size))
    }

    pub(super) fn memblock_containing(&self, addr: u32, size: u32) -> Option<i32> {
        let end = addr.wrapping_add(size.max(1));
        self.blocks
            .iter()
            .find(|b| addr >= b.base && end <= b.base.wrapping_add(b.size))
            .map(|b| b.uid)
    }

    pub(super) fn memblock_info_at(&self, addr: u32) -> Option<(u32, u32, u32)> {
        self.blocks
            .iter()
            .find(|b| addr >= b.base && addr < b.base.wrapping_add(b.size))
            .map(|b| (b.base, b.size, b.ty))
    }

    pub(super) fn memblock_base(&self, uid: i32) -> Option<u32> {
        self.blocks.iter().find(|b| b.uid == uid).map(|b| b.base)
    }

    pub(super) fn cdram_used(&self) -> u32 {
        self.cdram_used
    }

    pub(super) fn phycont_used(&self) -> u32 {
        self.phycont_used
    }

    /// Headroom between the cursor and `ceiling`, plus every released hole.
    pub(super) fn main_free_bytes(&self, ceiling: u32) -> u32 {
        let headroom = ceiling.saturating_sub(self.cursor);
        let holes = self.freed_bytes().min(u64::from(u32::MAX)) as u32;
        headroom.saturating_add(holes)
    }

    pub(super) fn cursor(&self) -> u32 {
        self.cursor
    }

    pub(super) fn set_cursor(&mut self, addr: u32) {
        self.cursor = addr;
    }

    pub(super) fn freed_count(&self) -> usize {
        self.freed.len()
    }

    pub(super) fn freed_bytes(&self) -> u64 {
        self.freed.iter().map(|&(_, sz)| sz as u64).sum()
    }
}
