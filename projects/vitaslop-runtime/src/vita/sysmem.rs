//! SceSysmem: kernel memory blocks. `sceKernelAllocMemBlock` hands back real
//! guest memory (via the deterministic bump allocator) so the guest can write
//! into it, and `sceKernelGetMemBlockBase` returns that base through its
//! out-pointer.
//!
//! The handlers are written with `#[hostcall]`: a typed signature, with the AAPCS
//! argument reads and the return write generated. `Ptr` args are guest pointers;
//! `&mut GuestCtx` is taken only where a handler dereferences one (an out-param).

use crate::host::{GuestCtx, Ptr, VitaState};
use crate::hostcall;

/// SceUID sceKernelAllocMemBlock(const char *name, SceKernelMemBlockType type,
///                               SceSize size, SceKernelAllocMemBlockOpt *opt)
#[hostcall]
pub(super) fn alloc_mem_block(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    name: Ptr,
    ty: u32,
    size: u32,
    _opt: Ptr,
) -> i32 {
    // The block's NAME is what the guest calls it, and on the failure path below it is the
    // only thing that says which subsystem asked - worth the read exactly there.
    let named = || {
        if name.is_null() {
            String::from("<unnamed>")
        } else {
            ctx.read_cstr(name.addr(), 64)
        }
    };
    // CDRAM aligns to 256 KiB, other blocks to 4 KiB. The guest already rounds
    // size; align the base to match hardware granularity.
    // Every block, with the arena cursor it landed on. A title's own memory probe and its
    // pool setup are indistinguishable in a total, and the cursor beside the size is what
    // says whether a free was actually reclaimed - see `vitaslop::mem`.
    let uid_or_zero = st.alloc_memblock(size, 256 * 1024, ty);
    tracing::debug!(
        target: "vitaslop::mem",
        size, ty = format_args!("{ty:#010x}"), name = %named(), uid = uid_or_zero,
        base = format_args!("{:#010x}", st.memblock_base(uid_or_zero).unwrap_or(0)),
        cursor = format_args!("{:#010x}", st.alloc_cursor_addr()),
        free_holes = st.freed_memblock_count(),
        free_bytes = st.freed_memblock_bytes(),
        "allocMemBlock"
    );
    match uid_or_zero {
        0 => {
            // An exhausted arena must be an error the guest can act on. Reporting a live
            // SceUID whose base is 0 is a hollow success: the caller's null check passes,
            // `sceKernelGetMemBlockBase` hands it 0, and the failure surfaces much later
            // as a write through a null pointer with nothing left pointing at the cause.
            tracing::error!(
                target: "vitaslop::err",
                size, name = %named(), caller = format_args!("{:#010x}", ctx.regs[14]),
                "sceKernelAllocMemBlock: no guest memory left - reporting NO_MEMORY"
            );
            tracing::error!(
                target: "vitaslop::err",
                trail = %crate::vita::guest_return_trail(ctx, 64),
                "  ...and the guest return addresses above it"
            );
            SCE_KERNEL_ERROR_NO_MEMORY
        }
        uid => uid,
    }
}

/// `SCE_KERNEL_ERROR_NO_MEMORY`: the allocation could not be satisfied.
const SCE_KERNEL_ERROR_NO_MEMORY: i32 = 0x8002_0003u32 as i32;

/// Which physical partition a `SceKernelMemBlockType` names.
///
/// >>> A VITA HAS THREE, AND LUMPING THEM INTO ONE ARENA IS NOT A SIMPLIFICATION - IT IS A
/// >>> WRONG ANSWER TO A QUESTION TITLES ASK OUT LOUD. A title sizes its pools by ALLOCATING
/// until it fails: MEASURED here, one probes CDRAM by asking for 128 MB, freeing it, then
/// 192, 224, 240... Told (by a single flat arena) that 256 MB of CDRAM exists, it took a
/// 255 MB video block, and the 56 MB MAIN-RAM pool it asked for next - which on hardware
/// comes out of a different partition entirely - got NO_MEMORY. The console's own figures
/// are already stated in `sceAppMgrGetBudgetInfo`; these are the same numbers, so the two
/// answers cannot drift apart.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Partition {
    /// `USER_CDRAM_RW` - the GPU's 128 MB, of which the shell keeps 16.
    Cdram,
    /// `USER_MAIN_PHYCONT_*` - physically contiguous main RAM, for hardware that cannot
    /// scatter-gather (the video decoder, mostly).
    Phycont,
    /// Everything else: ordinary user main RAM, cached or not.
    Main,
}

/// The partition a memblock type word names. The discriminator is the type's high byte,
/// which is what separates the families: `0x09...` CDRAM, `0x0C80.../0x0D80...` phycont,
/// `0x0C20...` ordinary user RW.
pub(crate) fn partition_of(ty: u32) -> Partition {
    match ty {
        0x0940_8060 | 0x0940_D060 => Partition::Cdram,
        0x0C80_D060 | 0x0C80_8060 | 0x0D80_8060 | 0x0D80_D060 => Partition::Phycont,
        _ => Partition::Main,
    }
}

/// CDRAM available to a GAME: 112 of the 128 MB, the other 16 being the shell's. The
/// figure `sceAppMgrGetBudgetInfo` reports, and the ceiling `sceKernelAllocMemBlock`
/// enforces - a budget a title is told about but can exceed is not a budget.
pub(crate) const CDRAM_BUDGET_BYTES: u32 = 112 * 1024 * 1024;

/// Physically contiguous main RAM available to a game.
pub(crate) const PHYCONT_BUDGET_BYTES: u32 = 32 * 1024 * 1024;

/// int sceKernelGetFreeMemorySize(SceKernelFreeMemorySizeInfo *info)
///
/// `SceKernelFreeMemorySizeInfo` is `{ size, size_user, size_cdram, size_phycont }`
/// (`psp2/kernel/sysmem.h`, asserted there at 0x10), and like `SceKernelMemBlockInfo` its
/// leading `size` is an IN field the caller stamps: only the fields that fit inside it are
/// written, so a caller built against a shorter struct is not written past.
///
/// Every figure is the ALLOCATOR'S OWN, never a constant: what `sceKernelAllocMemBlock` would
/// still grant from each partition right now. CDRAM and phycont are their budgets
/// ([`CDRAM_BUDGET_BYTES`], [`PHYCONT_BUDGET_BYTES`]) less what is outstanding in them - the
/// same numbers `sceAppMgrGetBudgetInfo` reports - and main RAM is what the guest arena can
/// still hand out (`VitaState::main_free_bytes`). Blocks of every partition come out of that
/// one arena here, so a partition is never reported as having more free than the arena
/// could actually place.
#[hostcall]
pub(super) fn get_free_memory_size(ctx: &mut GuestCtx, st: &mut VitaState, info: Ptr) -> i32 {
    if info.is_null() {
        SCE_KERNEL_ERROR_ILLEGAL_ADDR
    } else {
        let declared = ctx.read_u32(info.addr());
        let main = st.main_free_bytes();
        let fields = [
            main,
            (CDRAM_BUDGET_BYTES - st.cdram_used()).min(main),
            (PHYCONT_BUDGET_BYTES - st.phycont_used()).min(main),
        ];
        for (i, v) in fields.iter().enumerate() {
            let off = 4 + 4 * i as u32;
            if off + 4 <= declared {
                // The fields are `int`: a figure past 2 GiB would read back negative.
                ctx.write_u32(info.addr() + off, (*v).min(i32::MAX as u32));
            }
        }
        0
    }
}

/// int sceKernelGetMemBlockBase(SceUID uid, void **base)
#[hostcall]
pub(super) fn get_mem_block_base(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    uid: i32,
    out: Ptr,
) -> i32 {
    let base = st.memblock_base(uid).unwrap_or(0);
    ctx.write_u32(out.addr(), base);
    0
}

/// `SCE_KERNEL_ERROR_UID_CANNOT_FIND_BY_ID`: freeing a block id that names no live
/// allocation. The real kernel rejects the id rather than pretending it freed.
const SCE_KERNEL_ERROR_UID_CANNOT_FIND_BY_ID: i32 = 0x8002_0064u32 as i32;

/// void sceKernelSetGPO(SceUInt32 gpo)
///
/// The debug GPIO output register, exported to user mode by SceDebugLed. On a
/// development unit its low bits light the board's diagnostic LEDs; retail hardware
/// wires none of them, so the write has no observable effect there either. The value
/// is held (it is a register, and a title using it as a boot progress marker leaves
/// its last marker behind) and the call returns nothing - void, so no return write.
///
/// The VALUE and the CALLER are traced (`vitaslop::gpo`), because a title that writes
/// this register in a tight loop is saying something: a boot progress marker changes
/// monotonically, while a small repeating cycle is a diagnostic BLINK CODE, which means
/// the title has decided something is wrong. The two look identical in a host-call
/// tally - only the values tell them apart - and the second is a report about US.
#[hostcall]
pub(super) fn set_gpo(ctx: &mut GuestCtx, st: &mut VitaState, gpo: u32) {
    tracing::debug!(
        target: "vitaslop::gpo",
        gpo = format_args!("{gpo:#010x}"),
        lr = format_args!("{:#010x}", ctx.regs[14]),
        "setGPO"
    );
    st.gpo = gpo;
}

/// int sceKernelFreeMemBlock(SceUID uid)
/// Release a memory block. The registry entry is removed so a later
/// `sceKernelGetMemBlockBase(uid)` no longer resolves it; the deterministic arena
/// does not physically reclaim the bytes (it only grows), which is invisible to the
/// guest. Rejecting an unknown id matches the kernel contract.
#[hostcall]
pub(super) fn free_mem_block(st: &mut VitaState, uid: i32) -> i32 {
    let base = st.memblock_base(uid).unwrap_or(0);
    tracing::debug!(
        target: "vitaslop::mem", uid, base = format_args!("{base:#010x}"), "freeMemBlock"
    );
    if st.free_memblock(uid) {
        0
    } else {
        SCE_KERNEL_ERROR_UID_CANNOT_FIND_BY_ID
    }
}

/// int sceKernelGetMemBlockInfoByAddr(void *base, SceKernelMemBlockInfo *info)
///
/// Describe the block an address falls inside. `SceKernelMemBlockInfo` is
/// `{ size, mappedBase, mappedSize, memoryType, access, type }`, and its leading `size`
/// is an IN field: the caller stamps it with the struct version it compiled against
/// (0x14 before the `type` field existed, 0x18 with it). It is honoured rather than
/// overwritten with our own idea of the size, so a caller built against the short form
/// does not get a sixth word written past the end of its struct.
///
/// `memoryType` and `access` are DERIVED FROM THE BLOCK'S OWN TYPE WORD, which encodes
/// both: bits 15..8 are the cacheability (0xD0 normal, 0x80 uncached) and bits 7..4 are
/// the access rights (R=4, W=2, X=1). Reading them out of the type the guest asked for is
/// the only construction that cannot contradict the allocation it is describing - e.g.
/// `..._USER_RW` 0x0C20D060 reports normal/RW, and `..._USER_RW_UNCACHE` 0x0C208060
/// reports uncached with the same rights.
#[hostcall]
pub(super) fn get_mem_block_info_by_addr(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    base: Ptr,
    info: Ptr,
) -> i32 {
    // Early exits for the two failure cases, which a `#[hostcall]` body cannot have.
    get_mem_block_info_by_addr_impl(ctx, st, base, info)
}

fn get_mem_block_info_by_addr_impl(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    base: Ptr,
    info: Ptr,
) -> i32 {
    let Some((block_base, size, ty)) = st.memblock_info_at(base.addr()) else {
        return SCE_KERNEL_ERROR_BLOCK_ERROR;
    };
    if info.is_null() {
        return SCE_KERNEL_ERROR_ILLEGAL_ADDR;
    }
    let declared = ctx.read_u32(info.addr());
    ctx.write_u32(info.addr() + 0x04, block_base);
    ctx.write_u32(info.addr() + 0x08, size);
    ctx.write_u32(info.addr() + 0x0c, (ty >> 8) & 0xff);
    ctx.write_u32(info.addr() + 0x10, (ty >> 4) & 0xf);
    if declared >= 0x18 {
        ctx.write_u32(info.addr() + 0x14, ty);
    }
    0
}

/// int sceKernelGetMemBlockInfoByRange(void *base, SceSize size, SceKernelMemBlockInfo *info)
///
/// [`get_mem_block_info_by_addr`] for a whole span: the block must contain all of
/// `[base, base+size)`, as [`find_mem_block_by_addr`] requires.
#[hostcall]
pub(super) fn get_mem_block_info_by_range(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    base: Ptr,
    size: u32,
    info: Ptr,
) -> i32 {
    if st.memblock_containing(base.addr(), size).is_none() {
        SCE_KERNEL_ERROR_BLOCK_ERROR
    } else {
        get_mem_block_info_by_addr_impl(ctx, st, base, info)
    }
}

/// `SCE_KERNEL_MODEL_VITA` (psp2common/kernel/sysmem.h): the handheld, which is the console
/// this engine presents - a front and a back touch panel, no light bar, no DualShock.
pub(super) const SCE_KERNEL_MODEL_VITA: u32 = 0x10000;

/// int sceKernelGetModel(void) / sceKernelGetModelForCDialog(void): the handheld model.
#[hostcall]
pub(super) fn get_model(_st: &mut VitaState) -> u32 {
    SCE_KERNEL_MODEL_VITA
}

/// int sceKernelIsPSVitaTV(void): 0 - this is the handheld (see [`get_model`]).
#[hostcall]
pub(super) fn is_ps_vita_tv(_st: &mut VitaState) -> u32 {
    0
}

/// `SCE_KERNEL_ERROR_BLOCK_ERROR`: the address is in no memory block.
const SCE_KERNEL_ERROR_BLOCK_ERROR: i32 = 0x8002_D082u32 as i32;
/// `SCE_KERNEL_ERROR_ILLEGAL_ADDR`: the out-pointer is null.
const SCE_KERNEL_ERROR_ILLEGAL_ADDR: i32 = 0x8002_0005u32 as i32;

/// SceUID sceKernelFindMemBlockByAddr(const void *addr, SceSize size)
///
/// Resolve an address back to the block it lives in - what a title does when it is
/// handed a pointer and needs the UID to free or query it. The block must CONTAIN the
/// whole `[addr, addr+size)` range, since a caller asking about a span that straddles
/// two blocks has no single answer. An address in no block reports
/// `SCE_KERNEL_ERROR_BLOCK_ERROR` rather than a plausible-looking UID.
#[hostcall]
pub(super) fn find_mem_block_by_addr(st: &mut VitaState, addr: u32, size: u32) -> i32 {
    match st.memblock_containing(addr, size) {
        Some(uid) => uid,
        None => SCE_KERNEL_ERROR_BLOCK_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::{SliceMemory, VFP_ARG_COUNT};
    use crate::world::DeterministicWorld;
    use vitaslop_transpiler::abi::REG_COUNT;

    /// The guest region the state accounts for: large enough that the arena has headroom
    /// above the main-stack reserve. Nothing is written into it - only the small slice below.
    const REGION: u32 = 0x0800_0000;

    fn free_sizes(st: &mut VitaState, bytes: &mut [u8], declared: u32) -> (i32, [u32; 3]) {
        bytes[0x10..0x14].copy_from_slice(&declared.to_le_bytes());
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        regs[0] = 0x10;
        let mut mem = SliceMemory(bytes);
        let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        get_free_memory_size(&mut ctx, st);
        let w = |i: usize| u32::from_le_bytes(bytes[0x14 + 4 * i..0x18 + 4 * i].try_into().unwrap());
        (regs[0] as i32, [w(0), w(1), w(2)])
    }

    /// Each partition's free figure moves with the allocator: a main block takes from main,
    /// a CDRAM block takes from its budget AND the shared arena, and a free gives it back.
    #[test]
    fn free_memory_tracks_the_allocator() {
        let mut st = VitaState::new(0, REGION, Box::new(DeterministicWorld::default()));
        let mut bytes = vec![0u8; 0x100];
        let (rc, [main0, cdram0, phy0]) = free_sizes(&mut st, &mut bytes, 0x10);
        assert_eq!(rc, 0);
        assert_eq!(main0, st.main_free_bytes());
        assert_eq!(cdram0, CDRAM_BUDGET_BYTES);
        assert_eq!(phy0, PHYCONT_BUDGET_BYTES);

        let main_block = st.alloc_memblock(0x10_0000, 4096, 0x0C20_D060);
        assert!(main_block > 0);
        let cdram_block = st.alloc_memblock(0x20_0000, 256 * 1024, 0x0940_D060);
        assert!(cdram_block > 0);
        let (_, [main1, cdram1, phy1]) = free_sizes(&mut st, &mut bytes, 0x10);
        assert_eq!(cdram1, CDRAM_BUDGET_BYTES - 0x20_0000);
        assert_eq!(phy1, PHYCONT_BUDGET_BYTES);
        assert!(main1 <= main0 - 0x30_0000, "both blocks came out of the one arena");

        assert!(st.free_memblock(main_block));
        let (_, [main2, ..]) = free_sizes(&mut st, &mut bytes, 0x10);
        assert_eq!(main2, main1 + 0x10_0000, "a released block is free again");
    }

    /// A caller stamping a shorter struct gets only the fields inside it.
    #[test]
    fn free_memory_honours_the_declared_size() {
        let mut st = VitaState::new(0, REGION, Box::new(DeterministicWorld::default()));
        let mut bytes = vec![0xAAu8; 0x100];
        let (rc, [main, cdram, _]) = free_sizes(&mut st, &mut bytes, 0x8);
        assert_eq!(rc, 0);
        assert_eq!(main, st.main_free_bytes());
        assert_eq!(cdram, 0xAAAA_AAAA, "past the declared size is left alone");
    }
}
