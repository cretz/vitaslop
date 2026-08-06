//! SceSysmem: kernel memory blocks. `sceKernelAllocMemBlock` hands back real
//! guest memory (via the deterministic bump allocator) so the guest can write
//! into it, and `sceKernelGetMemBlockBase` returns that base through its
//! out-pointer.
//!
//! The handlers are written with `#[hostcall]`: a typed signature, with the AAPCS
//! argument reads and the return write generated. `Ptr` args are guest pointers;
//! `&mut GuestCtx` is taken only where a handler dereferences one (an out-param).

use crate::hostcall;

/// SceUID sceKernelAllocMemBlock(const char *name, SceKernelMemBlockType type,
///                               SceSize size, SceKernelAllocMemBlockOpt *opt)
#[hostcall]
pub(super) fn alloc_mem_block(st: &mut VitaState, _name: Ptr, _ty: u32, size: u32, _opt: Ptr) -> i32 {
    // CDRAM aligns to 256 KiB, other blocks to 4 KiB. The guest already rounds
    // size; align the base to match hardware granularity.
    match st.alloc_memblock(size, 256 * 1024) {
        0 => {
            // An exhausted arena must be an error the guest can act on. Reporting a live
            // SceUID whose base is 0 is a hollow success: the caller's null check passes,
            // `sceKernelGetMemBlockBase` hands it 0, and the failure surfaces much later
            // as a write through a null pointer with nothing left pointing at the cause.
            tracing::error!(
                target: "vitaslop::err",
                size,
                "sceKernelAllocMemBlock: no guest memory left - reporting NO_MEMORY"
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
#[hostcall]
pub(super) fn set_gpo(st: &mut VitaState, gpo: u32) {
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
        None => 0x8002_D082u32 as i32,
    }
}
