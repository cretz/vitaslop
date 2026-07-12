//! SceSysmem: kernel memory blocks. `sceKernelAllocMemBlock` hands back real
//! guest memory (via the deterministic bump allocator) so the guest can write
//! into it, and `sceKernelGetMemBlockBase` returns that base through its
//! out-pointer.
//!
//! The handlers are written with `#[hostcall]`: a typed signature, with the AAPCS
//! argument reads and the return write generated. `Ptr` args are guest pointers;
//! `&mut GuestCtx` is taken only where a handler dereferences one (an out-param).

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::nid::sysmem as nid;
use crate::SvcOutcome;

pub fn try_dispatch(func_nid: u32, ctx: &mut GuestCtx, st: &mut VitaState) -> Option<SvcOutcome> {
    match func_nid {
        nid::ALLOC_MEM_BLOCK => alloc_mem_block(ctx, st),
        nid::GET_MEM_BLOCK_BASE => get_mem_block_base(ctx, st),
        _ => return None,
    }
    Some(SvcOutcome::Continue)
}

/// SceUID sceKernelAllocMemBlock(const char *name, SceKernelMemBlockType type,
///                               SceSize size, SceKernelAllocMemBlockOpt *opt)
#[hostcall]
fn alloc_mem_block(st: &mut VitaState, _name: Ptr, _ty: u32, size: u32, _opt: Ptr) -> i32 {
    // CDRAM aligns to 256 KiB, other blocks to 4 KiB. The guest already rounds
    // size; align the base to match hardware granularity.
    st.alloc_memblock(size, 256 * 1024)
}

/// int sceKernelGetMemBlockBase(SceUID uid, void **base)
#[hostcall]
fn get_mem_block_base(ctx: &mut GuestCtx, st: &mut VitaState, uid: i32, out: Ptr) -> i32 {
    let base = st.memblock_base(uid).unwrap_or(0);
    ctx.write_u32(out.addr(), base);
    0
}
