//! SceSysmem: kernel memory blocks. `sceKernelAllocMemBlock` hands back real
//! guest memory (via the deterministic bump allocator) so the guest can write
//! into it, and `sceKernelGetMemBlockBase` returns that base through its
//! out-pointer.

use crate::host::{GuestCtx, VitaState};
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
fn alloc_mem_block(ctx: &mut GuestCtx, st: &mut VitaState) {
    let _name = ctx.arg(0);
    let _type = ctx.arg(1);
    let size = ctx.arg(2);
    let _opt = ctx.arg(3);
    // CDRAM aligns to 256 KiB, other blocks to 4 KiB. The guest already rounds
    // size; align the base to match hardware granularity.
    let uid = st.alloc_memblock(size, 256 * 1024);
    ctx.ret(uid as u32);
}

/// int sceKernelGetMemBlockBase(SceUID uid, void **base)
fn get_mem_block_base(ctx: &mut GuestCtx, st: &mut VitaState) {
    let uid = ctx.arg(0) as i32;
    let out = ctx.arg(1);
    let base = st.memblock_base(uid).unwrap_or(0);
    ctx.write_u32(out, base);
    ctx.ret(0);
}
