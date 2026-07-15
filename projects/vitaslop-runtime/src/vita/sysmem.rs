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
    st.alloc_memblock(size, 256 * 1024)
}

/// int sceKernelGetMemBlockBase(SceUID uid, void **base)
#[hostcall]
pub(super) fn get_mem_block_base(ctx: &mut GuestCtx, st: &mut VitaState, uid: i32, out: Ptr) -> i32 {
    let base = st.memblock_base(uid).unwrap_or(0);
    ctx.write_u32(out.addr(), base);
    0
}
