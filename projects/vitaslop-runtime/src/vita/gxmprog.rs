//! The `SceGxmVertexProgram` / `SceGxmFragmentProgram` HANDLE, laid out in guest memory.
//!
//! # Why a handle is a guest structure
//! `sceGxmShaderPatcherCreateVertexProgram` hands back a pointer to a structure the shader
//! patcher built inside memory the guest gave it (the patcher is constructed with the
//! guest's own allocator callbacks). Until now this engine answered with an opaque
//! counter, which is indistinguishable to the guest - it never dereferences the pointer -
//! and costs one thing that turned out to matter: **a counter has nowhere to keep the
//! facts a hot call needs.**
//!
//! `sceGxmReserve{Vertex,Fragment}DefaultUniformBuffer` is the largest remaining block of
//! host calls in a gameplay frame - MEASURED on one title's race at **1,189 crossings a
//! frame, 53% of everything it calls** - and the only reason it had to cross was that
//! sizing the buffer meant resolving the bound handle through a HOST map and reflecting
//! the program behind it. Both answers are fixed the moment the program is created, so
//! they belong in the structure the create call returns, and then the reserve is a bump of
//! a cursor over two words the guest already holds. That is the same move
//! [`super::gxmctx`] made for the sticky context state, applied to the other operand of
//! the same call.
//!
//! # What is memoised, and why it is exact rather than close
//! The size is not a cached guess that a later reflection might disagree with: the HANDLER
//! reads these same words, so there is one definition of a program's default-uniform size
//! and both paths compute from it. A `SceGxmProgram`'s bytes are immutable while it is
//! registered with the patcher (see `VitaState::invalidate_program_reflection` for the one
//! point at which a header address can start meaning a different program), and a vertex
//! program handle names one program for its whole life, so "at create" is the earliest
//! moment the answer exists and there is no later moment at which it changes.

use crate::host::GuestCtx;

/// Byte offset of each field from the handle.
///
/// One definition, read by the handlers in [`super::gxm`] and by the
/// `InlineOp::ReserveUniformBuffer` form that replaces them, held together by
/// `the_uniform_reserve_layout_is_closed`.
pub mod off {
    /// Identity stamp, so a handle that never came from a create call - or a pointer that
    /// is not a handle at all - is caught instead of read as a size. See [`super::MAGIC`].
    pub const MAGIC: u32 = 0x00;
    /// The default uniform buffer's size in BYTES, as the program's reflected interface
    /// asks for it. This is what a draw records as the bound size.
    pub const UNIFORM_SIZE: u32 = 0x04;
    /// How many bytes a reserve takes from the ring for it, which is
    /// [`UNIFORM_SIZE`](Self) floored at [`super::super::gxmctx::UNIFORM_MIN_ALLOC`].
    /// Stored rather than derived so the emitted form needs no `max`.
    pub const UNIFORM_ALLOC: u32 = 0x08;
    /// The `SceGxmProgram *` this program was created from.
    pub const HEADER: u32 = 0x0c;
}

/// Total bytes a handle block occupies.
pub const BYTES: u32 = off::HEADER + 4;

/// The word [`init`] stamps at [`off::MAGIC`].
///
/// Not decoration: the bound-program word of a context block can hold a handle from an
/// older run of this engine, a null, or - if a title ever passes something we did not
/// create - an arbitrary pointer. The inline reserve reads a SIZE through it and hands the
/// guest a buffer of that size, so an unstamped pointer must reach the handler instead of
/// being read as a program.
pub const MAGIC: u32 = 0x5658_5047; // "VXPG"

/// Lay a handle block out at `block`.
///
/// `size` is the reflected default-uniform size in bytes; `header` the `SceGxmProgram *`.
pub fn init(ctx: &mut GuestCtx, block: u32, size: u32, header: u32) {
    ctx.write_u32(block.wrapping_add(off::UNIFORM_SIZE), size);
    ctx.write_u32(
        block.wrapping_add(off::UNIFORM_ALLOC),
        size.max(super::gxmctx::UNIFORM_MIN_ALLOC),
    );
    ctx.write_u32(block.wrapping_add(off::HEADER), header);
    // Stamped LAST, so a partially written block is never mistaken for a complete one.
    ctx.write_u32(block.wrapping_add(off::MAGIC), MAGIC);
}

/// Whether `block` points at a handle [`init`] stamped.
pub fn is_program(ctx: &GuestCtx, block: u32) -> bool {
    block != 0 && ctx.read_u32(block.wrapping_add(off::MAGIC)) == MAGIC
}

/// Everything a handle memoises: `(size, alloc, header)`, or `None` when the pointer is
/// not one of ours.
///
/// The HEADER is read from here rather than from the host's handle map for the same reason
/// the size is: the emitted reserve reads this word, so a handler that answered from
/// somewhere else would be a second definition of "which program is this" - and the two
/// would agree on every ordinary run and disagree exactly on the rare fallback, which is
/// the worst place for a difference to live.
pub fn program(ctx: &GuestCtx, block: u32) -> Option<(u32, u32, u32)> {
    is_program(ctx, block).then(|| {
        (
            ctx.read_u32(block.wrapping_add(off::UNIFORM_SIZE)),
            ctx.read_u32(block.wrapping_add(off::UNIFORM_ALLOC)),
            ctx.read_u32(block.wrapping_add(off::HEADER)),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: u32 = 0x200;

    fn with_ctx<R>(f: impl FnOnce(&mut GuestCtx) -> R) -> R {
        use crate::{SliceMemory, VFP_ARG_COUNT};
        use vitaslop_transpiler::abi::REG_COUNT;
        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 4096];
        let mut mem = SliceMemory(&mut bytes);
        let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        f(&mut ctx)
    }

    /// An unstamped block is not a program, a stamped one is, and the floor on the
    /// allocation is applied at stamp time rather than being left for a reader to remember.
    #[test]
    fn a_stamped_block_carries_its_size_and_its_floor() {
        with_ctx(|ctx| {
            assert!(!is_program(ctx, BLOCK), "untouched memory is not a program");
            assert_eq!(program(ctx, BLOCK), None);
            init(ctx, BLOCK, 0, 0x8100_0000);
            assert!(is_program(ctx, BLOCK));
            // A program with NO default uniforms still takes a distinct slice of the ring,
            // or two draws in one scene would be handed the same address.
            assert_eq!(
                program(ctx, BLOCK),
                Some((0, crate::vita::gxmctx::UNIFORM_MIN_ALLOC, 0x8100_0000))
            );
            init(ctx, BLOCK, 4096, 0x8100_0000);
            assert_eq!(program(ctx, BLOCK), Some((4096, 4096, 0x8100_0000)));
        });
    }

    /// A null handle is never a program - the case a context block holds before anything
    /// is bound, and the one an inline form must not read a size through.
    #[test]
    fn a_null_handle_is_not_a_program() {
        with_ctx(|ctx| assert!(!is_program(ctx, 0)));
    }
}
