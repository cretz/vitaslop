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
pub(super) fn alloc_mem_block(ctx: &mut GuestCtx, st: &mut VitaState, name: Ptr, ty: u32, size: u32, _opt: Ptr) -> i32 {
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
    match st.alloc_memblock(size, 256 * 1024, ty) {
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

/// int sceKernelGetMemBlockBase(SceUID uid, void **base)
#[hostcall]
pub(super) fn get_mem_block_base(ctx: &mut GuestCtx, st: &mut VitaState, uid: i32, out: Ptr) -> i32 {
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
pub(super) fn get_mem_block_info_by_addr(ctx: &mut GuestCtx, st: &mut VitaState, base: Ptr, info: Ptr) -> i32 {
    // Early exits for the two failure cases, which a `#[hostcall]` body cannot have.
    get_mem_block_info_by_addr_impl(ctx, st, base, info)
}

fn get_mem_block_info_by_addr_impl(ctx: &mut GuestCtx, st: &mut VitaState, base: Ptr, info: Ptr) -> i32 {
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
