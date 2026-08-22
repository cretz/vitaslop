//! The GUEST-RESIDENT precomputed vertex/fragment STATE.
//!
//! A `SceGxmPrecomputed{Vertex,Fragment}State` used to be recorded in a host-side table
//! keyed by the struct's address. That had two costs, one of them a fidelity gap:
//!
//! - **The binds crossed.** `sceGxmSetPrecomputed{Vertex,Fragment}State` are per-draw-loop
//!   calls - 24 each per frame on one title's race - and a host-keyed table means every one
//!   is a boundary crossing ([[vitaslop-rank-host-calls-by-phone-count]]).
//! - **A by-value copy lost the state.** The struct is a small POD the guest owns; a title
//!   is free to build one and `memcpy` it to where it draws from. An address-keyed table
//!   cannot follow the copy - the identical defect the precomputed-DRAW family fixed by
//!   moving into the guest block (see `host::pdraw`).
//!
//! So the state lives in guest memory now, in two parts:
//!
//! - **The struct itself** (7 words for the vertex state, 9 for the fragment - both use the
//!   first seven): a stage-specific magic, the guest address of the ARRAYS block, the
//!   program handle and its resolved `SceGxmProgram *`, the default uniform buffer pointer,
//!   and the stage's MEMOISED uniform size in bytes. The size is a fact of the program,
//!   fixed when the state is initialised - the same "a fact fixed at creation belongs in
//!   the handle" move the inlined reserve rests on.
//! - **The ARRAYS block**, allocated from the emulator's guest heap at `Init` (the title's
//!   own `memBlock` is NOT used: its size is whatever the real driver's
//!   `sceGxmGetPrecomputed*StateSize` returned on hardware, which this engine does not
//!   define, so writing our arrays there would be a guess about someone else's allocation).
//!   The block holds the stage's table in EXACTLY the context block's own layout - the
//!   non-default uniform-buffer table for the vertex stage, the 16-unit texture-binding
//!   array for the fragment stage - so applying the state is one bulk copy.
//!
//! A `memcpy` of the struct copies the magic and the block pointer, so the copy aliases the
//! same arrays - which is exactly what a copied state does on hardware, where the struct
//! points into driver-owned working memory.
//!
//! `sceGxmSetPrecomputed*State` is then a copy between two guest structures, which is what
//! makes it inlinable (`vitaslop_transpiler::InlineOp::BindPrecomputedState`); the layout
//! is handed to the transpiler by `gxm::bind_state_layout`, and `layout_is_closed`-style
//! tests hold the two to one set of numbers.

/// Byte offsets within the state STRUCT. One layout for both stages - the stage is carried
/// by the magic, so a vertex state handed to the fragment bind falls back to the handler
/// rather than being read with the wrong meaning.
pub mod off {
    /// Identity stamp: [`super::MAGIC_VERTEX`] / [`super::MAGIC_FRAGMENT`].
    pub const MAGIC: u32 = 0x00;
    /// Guest address of the ARRAYS block ([`super::VERTEX_BLOCK_BYTES`] /
    /// [`super::FRAGMENT_BLOCK_BYTES`] bytes).
    pub const BLOCK: u32 = 0x04;
    /// The `SceGxmVertexProgram *` / `SceGxmFragmentProgram *` HANDLE the state was
    /// initialised from.
    pub const HANDLE: u32 = 0x08;
    /// The handle's resolved `SceGxmProgram *`.
    pub const HEADER: u32 = 0x0c;
    /// The stage's default uniform buffer pointer (`Set/GetDefaultUniformBuffer`).
    pub const BUF: u32 = 0x10;
    /// The stage's MEMOISED uniform size in bytes - what the bind writes into the context
    /// record's size word. Computed at `Init` from the program's reflected interface.
    pub const SIZE: u32 = 0x14;
    /// Bytes of the struct this engine writes (the public vertex struct is 7 words; only
    /// these six are used).
    pub const BYTES: u32 = 0x18;
}

/// "PVSV" - the vertex state's identity stamp. Content-free: an arbitrary tag, like
/// `pdraw::MAGIC`.
pub const MAGIC_VERTEX: u32 = 0x5653_5650;
/// "PVSF" - the fragment state's.
pub const MAGIC_FRAGMENT: u32 = 0x4653_5650;

/// Byte offset of the vertex block's TEXTURE array (see [`VERTEX_BLOCK_BYTES`]).
pub const VERTEX_BLOCK_TEXTURES: u32 = super::gxmctx::MAX_UNIFORM_BUFFERS as u32 * 4;
/// Bytes of the vertex state's arrays block: the non-default uniform-buffer table (one
/// word per index, in the context block's own layout), then a 16-unit texture array.
///
/// The texture array preserves what `sceGxmPrecomputedVertexStateSetTexture` records - the
/// old host table recorded these too, and the vertex BIND has never applied them (only the
/// fragment bind rebinds textures), so the bind's inline copy covers the table alone and
/// the recorded textures keep exactly their old, unread fate.
pub const VERTEX_BLOCK_BYTES: u32 = VERTEX_BLOCK_TEXTURES + FRAGMENT_BLOCK_BYTES;
/// Bytes of the fragment state's arrays block: the 16-unit texture-binding array, in the
/// context block's own layout ([`super::gxmctx::TEXTURE_STRIDE`] each).
pub const FRAGMENT_BLOCK_BYTES: u32 =
    super::gxmctx::MAX_TEXTURE_UNITS as u32 * super::gxmctx::TEXTURE_STRIDE;

/// Write texture-binding slot `unit` of a state's texture ARRAY at `array` (the fragment
/// block's base, or the vertex block's [`VERTEX_BLOCK_TEXTURES`]), in exactly the context
/// block's slot layout: `[addr, w0..w3, from_precomputed]`. A zero `addr` writes a zero
/// slot, which is the unbound encoding the context block itself uses.
pub fn write_texture_slot(
    ctx: &mut crate::host::GuestCtx,
    array: u32,
    unit: u32,
    addr: u32,
    words: [u32; 4],
) {
    let at = array.wrapping_add(unit * super::gxmctx::TEXTURE_STRIDE);
    if addr == 0 {
        for w in 0..super::gxmctx::TEXTURE_STRIDE / 4 {
            ctx.write_u32(at.wrapping_add(w * 4), 0);
        }
        return;
    }
    ctx.write_u32(at, addr);
    for (k, w) in words.iter().enumerate() {
        ctx.write_u32(at.wrapping_add(4 + k as u32 * 4), *w);
    }
    ctx.write_u32(at.wrapping_add(4 + super::gxmctx::TEXTURE_CONTROL_WORDS * 4), 1);
}
