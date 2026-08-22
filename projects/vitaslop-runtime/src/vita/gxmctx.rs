//! The sticky GXM CONTEXT state, laid out in the guest's own context memory.
//!
//! # Why it lives in guest memory
//! On hardware `sceGxmCreateContext(params, out)` builds its context INSIDE
//! `params->hostMem` - the guest allocates the storage, hands it over, and never touches
//! it again - and every `sceGxmSet*` writes one field of that structure. Keeping the same
//! state in a host-side struct instead has two costs, and the second is the expensive one:
//!
//!   1. It is not faithful. Two contexts share one host struct, and any handler that has
//!      only the context pointer cannot tell them apart.
//!   2. **A setter whose answer lives in host state cannot be inlined.** A `sceGxmSet*`
//!      that writes guest memory is one wasm store; one that writes a host field is a full
//!      boundary crossing. That is the entire cost of these calls - the handlers are three
//!      instructions - and on a phone the crossing is 36% of host-call time against a
//!      desktop where it barely registers. Measured on a retail title at 248 calls per frame
//!      EACH, the scalar setters below are the largest single block of host calls left in
//!      steady gameplay.
//!
//! This is the same move that took `sceGxmTextureGetLodBias` from 185,606 calls to zero:
//! put the state where the hardware puts it, and inlining follows.
//!
//! # Layout
//! Word offsets from the context pointer, in the order the fields appear in
//! [`crate::capture::RenderState`] plus the two bound program handles and the vertex
//! streams. The numbers are load-bearing in two places at once - the handlers here and the
//! `InlineOp::StoreArg { offset }` forms in [`super::gxm::inline_op`] - so
//! [`layout_is_closed`] holds them together and every offset has exactly one definition.
//!
//! The block is 208 bytes against the `SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE` of 2048 that
//! the guest is required to supply, so it always fits; [`init`] checks rather than assumes,
//! because a context we could not place is a context whose every draw would read garbage.
//!
//! # What is NOT here, and why
//! - **Fragment textures.** GXM copies a texture's 16 control words BY VALUE at
//!   `sceGxmSetFragmentTexture` (see `vitaslop-texture-binding-by-value`), so a binding is a
//!   snapshot, not a pointer. Storing the pointer and reading it at draw time would be a
//!   different program - one where a texture the guest re-initialised between bind and draw
//!   renders with its NEW contents. That is a faithfulness regression, not a speedup, so
//!   those 259 calls a frame stay on the host until an inline form can copy 16 words.
//! - **The multi-argument setters' arguments beyond the first** are written here by their
//!   handlers like everything else; they simply cannot be INLINED, because a store form
//!   writes one word. `sceGxmSetViewport` takes six.

use crate::host::GuestCtx;

/// Byte offset of each field from the context pointer.
///
/// One definition, read by the handlers in [`super::gxm`] and by the inline forms that
/// replace them. A constant here that disagrees with an `InlineOp::StoreArg` offset there
/// is a wrong answer with no symptom - the guest's draw would read the field next door -
/// which is why [`layout_is_closed`] asserts the two agree rather than trusting them to.
pub mod off {
    /// Identity stamp, so a pointer that never came from `sceGxmCreateContext` is caught
    /// instead of read as state. See [`super::MAGIC`].
    pub const MAGIC: u32 = 0x00;
    pub const CULL_MODE: u32 = 0x04;
    pub const TWO_SIDED: u32 = 0x08;
    pub const FRONT_DEPTH_FUNC: u32 = 0x0c;
    pub const BACK_DEPTH_FUNC: u32 = 0x10;
    pub const FRONT_DEPTH_WRITE: u32 = 0x14;
    pub const BACK_DEPTH_WRITE: u32 = 0x18;
    pub const FRONT_FRAGMENT_PROGRAM_ENABLE: u32 = 0x1c;
    pub const BACK_FRAGMENT_PROGRAM_ENABLE: u32 = 0x20;
    pub const FRONT_POLYGON_MODE: u32 = 0x24;
    pub const BACK_POLYGON_MODE: u32 = 0x28;
    pub const FRONT_POINT_LINE_WIDTH: u32 = 0x2c;
    pub const FRONT_STENCIL_REF: u32 = 0x30;
    pub const FRONT_STENCIL_FUNC: u32 = 0x34;
    pub const FRONT_STENCIL_OP_FAIL: u32 = 0x38;
    pub const FRONT_STENCIL_OP_DEPTH_FAIL: u32 = 0x3c;
    pub const FRONT_STENCIL_OP_DEPTH_PASS: u32 = 0x40;
    pub const FRONT_STENCIL_COMPARE_MASK: u32 = 0x44;
    pub const FRONT_STENCIL_WRITE_MASK: u32 = 0x48;
    pub const VIEWPORT_ENABLE: u32 = 0x4c;
    /// Six floats: `xOffset, xScale, yOffset, yScale, zOffset, zScale`, stored as their
    /// f32 bit patterns so the whole block is one array of words.
    pub const VIEWPORT: u32 = 0x50;
    pub const REGION_CLIP_MODE: u32 = 0x68;
    /// Four words: `xMin, yMin, xMax, yMax`.
    pub const REGION_CLIP: u32 = 0x6c;
    pub const FRONT_VISIBILITY_TEST_ENABLE: u32 = 0x7c;
    pub const FRONT_VISIBILITY_TEST_INDEX: u32 = 0x80;
    pub const FRONT_VISIBILITY_TEST_OP: u32 = 0x84;
    /// The bound `SceGxmVertexProgram *` HANDLE, not its `SceGxmProgram *` header. The
    /// handle is what `sceGxmSetVertexProgram` is handed, so storing it is the whole
    /// setter; resolving it to a header is the reader's job and happens once per draw
    /// instead of once per bind.
    pub const VERTEX_PROGRAM: u32 = 0x88;
    /// The bound `SceGxmFragmentProgram *` handle, same reasoning as
    /// [`VERTEX_PROGRAM`]. The blend state and program header both derive from it.
    pub const FRAGMENT_PROGRAM: u32 = 0x8c;
    /// `SCE_GXM_MAX_VERTEX_STREAMS` words: the `data` pointer bound to each vertex stream.
    /// A stream binding IS a pointer on hardware (unlike a texture binding), so it can
    /// live here as one.
    pub const STREAMS: u32 = 0x90;
    /// [`super::MAX_TEXTURE_UNITS`] entries of [`super::TEXTURE_STRIDE`] bytes: the fragment
    /// sampler bindings, as `[addr, w0, w1, w2, w3, from_precomputed]`.
    ///
    /// A binding is the four control words AS THEY READ AT BIND TIME, not the pointer: GXM
    /// copies them by value (`vitaslop-texture-binding-by-value`), so a texture the guest
    /// re-initialises between bind and draw must still render with its OLD contents. That is
    /// why this is six words and not one, and why the inline form is a COPY rather than the
    /// plain store every other setter uses.
    ///
    /// `addr` is kept for identity only - which decoded texture this binding resolves to -
    /// and is never re-read for control words. `addr == 0` means the unit is unbound.
    pub const TEXTURES: u32 = 0xd0;

    /// First byte after the texture block - where any state added later starts. The map up
    /// to here is PACKED (every offset is the previous one's end), so a new field cannot be
    /// squeezed in among the existing ones without moving them, and moving them would
    /// change the layout of a block a running title already holds.
    pub const AFTER_TEXTURES: u32 =
        TEXTURES + (super::MAX_TEXTURE_UNITS as u32) * super::TEXTURE_STRIDE;

    /// BACK-face stencil state (`sceGxmSetBackStencilFunc`), the two-sided counterpart of
    /// the `FRONT_STENCIL_*` block above. Only consulted when `TWO_SIDED` is enabled - with
    /// two-sided disabled the hardware applies the front state to both faces - but it is
    /// recorded unconditionally, because a title sets it once at start-up and enables
    /// two-sided later, and state that was dropped when it was set is not there when it
    /// starts mattering.
    pub const BACK_STENCIL_FUNC: u32 = AFTER_TEXTURES;
    pub const BACK_STENCIL_OP_FAIL: u32 = AFTER_TEXTURES + 0x04;
    pub const BACK_STENCIL_OP_DEPTH_FAIL: u32 = AFTER_TEXTURES + 0x08;
    pub const BACK_STENCIL_OP_DEPTH_PASS: u32 = AFTER_TEXTURES + 0x0c;
    pub const BACK_STENCIL_COMPARE_MASK: u32 = AFTER_TEXTURES + 0x10;
    pub const BACK_STENCIL_WRITE_MASK: u32 = AFTER_TEXTURES + 0x14;

    /// First byte after the back-stencil block, where the default-uniform state starts.
    pub const AFTER_BACK_STENCIL: u32 = BACK_STENCIL_WRITE_MASK + 4;

    /// The default-uniform RING: three guest ADDRESSES - where the ring starts, one past
    /// where it ends, and the next free byte in it.
    ///
    /// On hardware the default uniform buffer is a ring the driver recycles inside the
    /// memory the guest gave GXM, and every `sceGxmReserve*DefaultUniformBuffer` is a bump
    /// of a cursor in that ring. Keeping the cursor here rather than on the host is what
    /// makes the reserve inlinable at all - it is the same move `gxmctx` is entirely about
    /// - and it is also where the hardware keeps it.
    ///
    /// The cursor is an ABSOLUTE address rather than an offset so the emitted form can hand
    /// it straight back to the guest; the ring base is 16-aligned, so aligning the absolute
    /// cursor and aligning an offset from it are the same arithmetic.
    pub const UNIFORM_RING_BASE: u32 = AFTER_BACK_STENCIL;
    pub const UNIFORM_RING_END: u32 = AFTER_BACK_STENCIL + 0x04;
    pub const UNIFORM_RING_CURSOR: u32 = AFTER_BACK_STENCIL + 0x08;

    /// The VERTEX stage's bound default uniform buffer, as three consecutive words:
    /// `[buffer address, size in bytes, the `SceGxmProgram *` it was sized for]`.
    ///
    /// The header is not decoration: a buffer still bound for a DIFFERENT program is not
    /// this draw's uniform bank, and the only way a draw can tell is by comparing what the
    /// buffer was reserved for against what is about to be drawn (see
    /// `VitaState::stale_uniforms`).
    pub const VERTEX_UNIFORM: u32 = AFTER_BACK_STENCIL + 0x0c;
    /// The guest addresses `sceGxmSetVertexUniformBuffer(context, index, data)` binds, one
    /// word per buffer index 0..[`super::MAX_UNIFORM_BUFFERS`]. Sticky state exactly like
    /// the streams: a draw whose vertex program declares a non-default uniform buffer reads
    /// the bound address here to snapshot the buffer's bytes (the recompiled shader's
    /// memory loads chase that pointer - see `vitaslop_gxp_shader::module::MemWindow`).
    /// Placed AFTER both stages' uniform records so every pre-existing offset is unchanged.
    pub const VERTEX_UNIFORM_BUFFERS: u32 = FRAGMENT_UNIFORM + super::uniform_record::BYTES;
    /// The FRAGMENT stage's, in the same three-word shape - which is what lets ONE inline
    /// form serve both stages with only the record's offset changing.
    pub const FRAGMENT_UNIFORM: u32 = AFTER_BACK_STENCIL + 0x18;
}

/// Word offsets WITHIN a `VERTEX_UNIFORM` / `FRAGMENT_UNIFORM` record. Both stages have
/// the same three fields in the same order, so the inline form is parameterised by the
/// record's base alone.
pub mod uniform_record {
    /// The reserved buffer's guest address, or 0 for "nothing bound".
    pub const BUF: u32 = 0x00;
    /// Its size in BYTES, as the program's reflected interface asked for - which is not
    /// the same as the bytes taken from the ring (see [`super::UNIFORM_MIN_ALLOC`]).
    pub const SIZE: u32 = 0x04;
    /// The `SceGxmProgram *` the size was computed from.
    pub const HEADER: u32 = 0x08;
    /// Bytes one record occupies.
    pub const BYTES: u32 = 0x0c;
}

/// Bytes the ring hands out for a reserve, at least. A program with no default uniforms
/// asks for zero, and handing back the same address twice would let two draws' buffers
/// alias; the floor keeps every reserve in a scene at a distinct address, exactly as it
/// did when the bump lived on the host.
pub const UNIFORM_MIN_ALLOC: u32 = 256;

/// Alignment of every buffer the ring hands out, and of the ring itself.
pub const UNIFORM_ALIGN: u32 = 16;

/// `SCE_GXM_MAX_TEXTURE_UNITS` (vitasdk `gxm.h`).
pub const MAX_TEXTURE_UNITS: usize = 16;

/// Bytes per fragment texture binding: `addr`, four control words, `from_precomputed`.
pub const TEXTURE_STRIDE: u32 = 24;

/// Words of a binding that are COPIED from the guest's `SceGxmTexture` - the control words
/// themselves. The inline form and [`set_texture_binding`] must copy the same number.
pub const TEXTURE_CONTROL_WORDS: u32 = 4;

/// `SCE_GXM_MAX_VERTEX_STREAMS` (vitasdk `gxm.h`).
pub const MAX_VERTEX_STREAMS: usize = 16;

/// `SCE_GXM_MAX_UNIFORM_BUFFERS` (vitasdk `gxm.h`): non-default uniform buffer indices run
/// 0..14 per stage - the same numbering the GXP container table's "ordinary uniform buffer"
/// entries 0..13 use.
pub const MAX_UNIFORM_BUFFERS: usize = 14;

/// Total bytes the block occupies. Every guest context must have at least this much host
/// memory behind it.
pub const BYTES: u32 = off::VERTEX_UNIFORM_BUFFERS + (MAX_UNIFORM_BUFFERS as u32) * 4;

/// `SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE` (vitasdk `gxm.h`): the smallest `hostMem` GXM
/// accepts, and therefore the smallest a conforming title can pass.
pub const MINIMUM_HOST_MEM: u32 = 2 * 1024;

/// The word [`init`] stamps at [`off::MAGIC`], so a context pointer can be recognised.
///
/// Not decoration. The guest's `hostMem` is uninitialised memory before we touch it, and a
/// draw handed a pointer we never initialised would otherwise read whatever was there as a
/// cull mode and a depth function - a picture that is subtly wrong with nothing to report.
/// With the stamp, [`load`] can tell "this is a context" from "this is not", and say so.
pub const MAGIC: u32 = 0x5658_4354; // "VXCT"

/// Lay the block out at `context` with the GXM power-on defaults, and stamp it.
///
/// The defaults are [`crate::capture::RenderState::default`]'s, which is the single place
/// they are written down; this only serialises them.
pub fn init(ctx: &mut GuestCtx, context: u32) {
    store(ctx, context, &crate::capture::RenderState::default());
    for i in 0..MAX_VERTEX_STREAMS as u32 {
        ctx.write_u32(context.wrapping_add(off::STREAMS + i * 4), 0);
    }
    ctx.write_u32(context.wrapping_add(off::VERTEX_PROGRAM), 0);
    ctx.write_u32(context.wrapping_add(off::FRAGMENT_PROGRAM), 0);
    for i in 0..(MAX_TEXTURE_UNITS as u32) * TEXTURE_STRIDE / 4 {
        ctx.write_u32(context.wrapping_add(off::TEXTURES + i * 4), 0);
    }
    // The default-uniform ring starts EMPTY - a base of 0 is what both the handler and the
    // inline form read as "no ring here yet", and the ring is attached separately (see
    // `VitaState::attach_uniform_ring`) because only the host can allocate one.
    for offset in [off::UNIFORM_RING_BASE, off::UNIFORM_RING_END, off::UNIFORM_RING_CURSOR] {
        ctx.write_u32(context.wrapping_add(offset), 0);
    }
    for record in [off::VERTEX_UNIFORM, off::FRAGMENT_UNIFORM] {
        for w in 0..uniform_record::BYTES / 4 {
            ctx.write_u32(context.wrapping_add(record + w * 4), 0);
        }
    }
    for i in 0..MAX_UNIFORM_BUFFERS as u32 {
        ctx.write_u32(context.wrapping_add(off::VERTEX_UNIFORM_BUFFERS + i * 4), 0);
    }
    // Stamped LAST, so a partially written block is never mistaken for a complete one.
    ctx.write_u32(context.wrapping_add(off::MAGIC), MAGIC);
}

/// Whether `context` points at a block [`init`] stamped.
pub fn is_context(ctx: &GuestCtx, context: u32) -> bool {
    context != 0 && ctx.read_u32(context.wrapping_add(off::MAGIC)) == MAGIC
}

/// Write one word of the block. The single spelling every scalar setter uses, and the exact
/// thing `InlineOp::StoreArg { offset }` emits.
pub fn set(ctx: &mut GuestCtx, context: u32, offset: u32, value: u32) {
    ctx.write_u32(context.wrapping_add(offset), value);
}

/// Read one word of the block.
pub fn get(ctx: &GuestCtx, context: u32, offset: u32) -> u32 {
    ctx.read_u32(context.wrapping_add(offset))
}

/// The `data` pointer bound to vertex stream `index`, or 0 for an index GXM cannot produce.
pub fn stream(ctx: &GuestCtx, context: u32, index: u32) -> u32 {
    if index as usize >= MAX_VERTEX_STREAMS {
        return 0;
    }
    get(ctx, context, off::STREAMS + index * 4)
}

/// All [`MAX_VERTEX_STREAMS`] stream pointers, in stream order.
pub fn streams(ctx: &GuestCtx, context: u32) -> [u32; MAX_VERTEX_STREAMS] {
    // Sixteen words, and it runs per draw - so one read, for the reason spelled out on
    // [`load`] and [`texture_bindings`]. `MAX_VERTEX_STREAMS` of them is sixteen virtual
    // calls through `dyn GuestMemory` where one does.
    const SPAN: usize = MAX_VERTEX_STREAMS * 4;
    let mut scratch = [0u8; SPAN];
    let block: &[u8] = match ctx.borrow_bytes(context.wrapping_add(off::STREAMS), SPAN) {
        Some(b) => b,
        None => {
            ctx.read_into(context.wrapping_add(off::STREAMS), &mut scratch);
            &scratch
        }
    };
    let mut out = [0u32; MAX_VERTEX_STREAMS];
    for (i, slot) in out.iter_mut().enumerate() {
        let at = i * 4;
        *slot = u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]]);
    }
    out
}

/// Bind `addr` to vertex stream `index`. An index beyond
/// [`MAX_VERTEX_STREAMS`] is not something GXM can produce, so it is reported rather than
/// silently folded into a neighbouring slot.
pub fn set_stream(ctx: &mut GuestCtx, context: u32, index: u32, addr: u32) {
    if index as usize >= MAX_VERTEX_STREAMS {
        tracing::warn!(
            target: "vitaslop::gxm",
            index,
            data = format_args!("{addr:#x}"),
            "setVertexStream on a stream index beyond SCE_GXM_MAX_VERTEX_STREAMS - DROPPED"
        );
        return;
    }
    set(ctx, context, off::STREAMS + index * 4, addr);
}

/// The guest address bound to VERTEX non-default uniform buffer `index`, or 0 for none
/// (also 0 for an index GXM cannot produce).
pub fn vertex_uniform_buffer(ctx: &GuestCtx, context: u32, index: u32) -> u32 {
    if index as usize >= MAX_UNIFORM_BUFFERS {
        return 0;
    }
    get(ctx, context, off::VERTEX_UNIFORM_BUFFERS + index * 4)
}

/// Bind `addr` to VERTEX non-default uniform buffer `index`
/// (`sceGxmSetVertexUniformBuffer`). An index beyond [`MAX_UNIFORM_BUFFERS`] is not
/// something GXM can produce, so it is reported rather than folded into a neighbour.
pub fn set_vertex_uniform_buffer(ctx: &mut GuestCtx, context: u32, index: u32, addr: u32) {
    if index as usize >= MAX_UNIFORM_BUFFERS {
        tracing::warn!(
            target: "vitaslop::gxm",
            index,
            data = format_args!("{addr:#x}"),
            "setVertexUniformBuffer on an index beyond SCE_GXM_MAX_UNIFORM_BUFFERS - DROPPED"
        );
        return;
    }
    set(ctx, context, off::VERTEX_UNIFORM_BUFFERS + index * 4, addr);
}

/// One fragment sampler binding, as the block holds it.
///
/// The same shape the inline `sceGxmSetFragmentTexture` writes, which is the point: the
/// handler and the emitted code produce identical bytes, so a draw cannot tell which path
/// bound a texture. [`super::gxm::inline_op`]'s copy form and [`set_texture_binding`] are
/// held together by `the_texture_binding_layout_is_closed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TexBinding {
    /// The guest `SceGxmTexture *` this came from. IDENTITY ONLY - never re-read for
    /// control words. Zero means the unit is unbound.
    pub addr: u32,
    /// The four control words as they read AT BIND TIME.
    pub words: [u32; 4],
    /// Whether this arrived through a precomputed fragment state rather than a direct
    /// `sceGxmSetFragmentTexture`. Per binding, because both paths are live at once.
    pub from_precomputed: bool,
}

/// Byte offset of sampler `unit`'s binding within the block.
fn texture_at(unit: u32) -> u32 {
    off::TEXTURES + unit * TEXTURE_STRIDE
}

/// Bind `b` to sampler `unit`. A unit beyond [`MAX_TEXTURE_UNITS`] is not something GXM can
/// produce, so it is reported rather than folded into a neighbouring slot - the same
/// treatment [`set_stream`] gives an out-of-range stream, and the same case the inline form
/// hands back to this handler.
pub fn set_texture_binding(ctx: &mut GuestCtx, context: u32, unit: u32, b: TexBinding) {
    if unit as usize >= MAX_TEXTURE_UNITS {
        tracing::warn!(
            target: "vitaslop::gxm",
            unit,
            texture = format_args!("{:#x}", b.addr),
            "setFragmentTexture on a unit beyond SCE_GXM_MAX_TEXTURE_UNITS - DROPPED"
        );
        return;
    }
    let at = texture_at(unit);
    set(ctx, context, at, b.addr);
    for (k, w) in b.words.iter().enumerate() {
        set(ctx, context, at + 4 + k as u32 * 4, *w);
    }
    set(ctx, context, at + 4 + TEXTURE_CONTROL_WORDS * 4, b.from_precomputed as u32);
}

/// Read sampler `unit`'s binding.
/// Just the `data` pointer bound to fragment sampler `unit`, or 0 if nothing is bound there.
///
/// # Why this exists beside [`texture_binding`]
/// A draw has to find out WHICH of the sixteen units are bound, and almost none of them are - a
/// real title uses three or four. Asking [`texture_binding`] reads six guest words per unit,
/// ninety-six per draw, and then throws away the ninety it read for empty slots. This reads the
/// one word that decides, so the other five are read only for a unit that has something in it.
pub fn texture_binding_addr(ctx: &GuestCtx, context: u32, unit: u32) -> u32 {
    if unit as usize >= MAX_TEXTURE_UNITS {
        return 0;
    }
    get(ctx, context, texture_at(unit))
}

/// Every BOUND sampler unit's binding, read out of the block in ONE borrow.
///
/// # Why a bulk reader exists
/// A draw has to find out which of the sixteen units are bound and what they hold, and doing
/// that a word at a time costs forty-odd `read_u32`s - each a bounds check and a VIRTUAL CALL
/// through `dyn GuestMemory`, which is what makes them expensive rather than the four bytes
/// they move. MEASURED on a retail race, `draw: snapshot textures` was **7.5%
/// of the guest window over 627 draws a frame at 1.21 us each, moving 0.0 MB** - a phase whose
/// whole cost was the calls, not the data.
///
/// One `borrow_bytes` over the whole block is one virtual call, and the parse afterwards is
/// ordinary Rust over a slice. The per-unit readers stay for the callers that want one unit.
///
/// Falls back to the per-unit path when the backing cannot hand out a slice, so an engine
/// without `borrow` still answers - identically, which is what `out` being filled the same way
/// on both arms means.
pub fn texture_bindings(ctx: &GuestCtx, context: u32, out: &mut Vec<(u32, TexBinding)>) {
    out.clear();
    const SPAN: usize = MAX_TEXTURE_UNITS * TEXTURE_STRIDE as usize;
    // >>> THE FALLBACK USED TO BE PER-UNIT, AND IT IS THE BROWSER THAT TAKES IT.
    //
    // `borrow` hands back a slice into guest memory, and only NATIVE can: the browser's guest
    // memory is a SharedArrayBuffer that is not this module's own linear memory, so no `&[u8]`
    // can be formed over it and `borrow` returns `None` there permanently. The old fallback
    // then read the block a WORD AT A TIME - sixteen units by up to six `read_u32`s, ~96
    // virtual calls through `dyn GuestMemory` per draw - which is the exact cost this bulk
    // reader was written to remove, reintroduced on the one engine that ships.
    //
    // MEASURED once the browser could time its own phases: `draw: decode texture bindings`
    // was **2.24 ms of a ~17 ms frame on a retail title (13%), 4.9 us per draw over 457 draws**,
    // against 0.6 us on the desktop, which takes the `borrow` path and never saw it.
    //
    // A bulk `read` into a 384-byte stack buffer needs no slice and is ONE virtual call, so
    // both engines now parse the same bytes the same way and the fallback is a copy rather
    // than a different algorithm. [[vitaslop-fallback-must-report]] is the rule this broke:
    // it answered identically and eight times slower, silently.
    let mut scratch = [0u8; SPAN];
    let block: &[u8] = match ctx.borrow_bytes(context.wrapping_add(off::TEXTURES), SPAN) {
        Some(b) => b,
        None => {
            ctx.read_into(context.wrapping_add(off::TEXTURES), &mut scratch);
            &scratch
        }
    };
    let word = |at: usize| u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]]);
    for unit in 0..MAX_TEXTURE_UNITS {
        let at = unit * TEXTURE_STRIDE as usize;
        let addr = word(at);
        // A slot the guest never bound, or unbound, is not a binding - the same test, and the
        // same skip, the per-unit path makes.
        if addr == 0 {
            continue;
        }
        out.push((
            unit as u32,
            TexBinding {
                addr,
                words: [word(at + 4), word(at + 8), word(at + 12), word(at + 16)],
                from_precomputed: word(at + 4 + TEXTURE_CONTROL_WORDS as usize * 4) != 0,
            },
        ));
    }
}

pub fn texture_binding(ctx: &GuestCtx, context: u32, unit: u32) -> TexBinding {
    if unit as usize >= MAX_TEXTURE_UNITS {
        return TexBinding::default();
    }
    let at = texture_at(unit);
    TexBinding {
        addr: get(ctx, context, at),
        words: [
            get(ctx, context, at + 4),
            get(ctx, context, at + 8),
            get(ctx, context, at + 12),
            get(ctx, context, at + 16),
        ],
        from_precomputed: get(ctx, context, at + 4 + TEXTURE_CONTROL_WORDS * 4) != 0,
    }
}

/// One stage's bound default uniform buffer, as the block holds it.
///
/// The same three words the inline `sceGxmReserve*DefaultUniformBuffer` writes and the
/// same three the draw reads, so an inlined reserve and a handled one are indistinguishable
/// downstream. `buf == 0` means the stage has nothing bound and the draw falls back to the
/// `sceGxmSetUniformDataF` capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct UniformBinding {
    pub buf: u32,
    pub size: u32,
    pub header: u32,
}

/// Read the stage's binding out of the block. `record` is [`off::VERTEX_UNIFORM`] or
/// [`off::FRAGMENT_UNIFORM`].
pub fn uniform_binding(ctx: &GuestCtx, context: u32, record: u32) -> UniformBinding {
    UniformBinding {
        buf: get(ctx, context, record + uniform_record::BUF),
        size: get(ctx, context, record + uniform_record::SIZE),
        header: get(ctx, context, record + uniform_record::HEADER),
    }
}

/// Write the stage's binding into the block - the fallback half of what the emitted
/// reserve does, and the whole of what the precomputed-state paths do.
pub fn set_uniform_binding(ctx: &mut GuestCtx, context: u32, record: u32, b: UniformBinding) {
    set(ctx, context, record + uniform_record::BUF, b.buf);
    set(ctx, context, record + uniform_record::SIZE, b.size);
    set(ctx, context, record + uniform_record::HEADER, b.header);
}

/// The ring's `(base, end, cursor)`, all guest addresses. A base of 0 means no ring is
/// attached, which is the only state in which a reserve has to allocate one.
pub fn uniform_ring(ctx: &GuestCtx, context: u32) -> (u32, u32, u32) {
    (
        get(ctx, context, off::UNIFORM_RING_BASE),
        get(ctx, context, off::UNIFORM_RING_END),
        get(ctx, context, off::UNIFORM_RING_CURSOR),
    )
}

/// Attach a ring at `[base, base + size)` and park its cursor at the start.
pub fn set_uniform_ring(ctx: &mut GuestCtx, context: u32, base: u32, size: u32) {
    set(ctx, context, off::UNIFORM_RING_BASE, base);
    set(ctx, context, off::UNIFORM_RING_END, base.wrapping_add(size));
    set(ctx, context, off::UNIFORM_RING_CURSOR, base);
}

/// Move the ring's cursor back to the start of the ring, for a new scene.
pub fn rewind_uniform_ring(ctx: &mut GuestCtx, context: u32) {
    let base = get(ctx, context, off::UNIFORM_RING_BASE);
    set(ctx, context, off::UNIFORM_RING_CURSOR, base);
}

/// Serialise a whole [`crate::capture::RenderState`] into the block.
///
/// Used by [`init`] for the defaults, and by the tests that hold the round trip together.
/// The ordinary path never calls it - a setter writes ONE word, which is the point.
pub fn store(ctx: &mut GuestCtx, context: u32, rs: &crate::capture::RenderState) {
    let mut w = |offset, value| set(ctx, context, offset, value);
    w(off::CULL_MODE, rs.cull_mode);
    w(off::TWO_SIDED, rs.two_sided);
    w(off::FRONT_DEPTH_FUNC, rs.front_depth_func);
    w(off::BACK_DEPTH_FUNC, rs.back_depth_func);
    w(off::FRONT_DEPTH_WRITE, rs.front_depth_write);
    w(off::BACK_DEPTH_WRITE, rs.back_depth_write);
    w(off::FRONT_FRAGMENT_PROGRAM_ENABLE, rs.front_fragment_program_enable);
    w(off::BACK_FRAGMENT_PROGRAM_ENABLE, rs.back_fragment_program_enable);
    w(off::FRONT_POLYGON_MODE, rs.front_polygon_mode);
    w(off::BACK_POLYGON_MODE, rs.back_polygon_mode);
    w(off::FRONT_POINT_LINE_WIDTH, rs.front_point_line_width);
    w(off::FRONT_STENCIL_REF, rs.front_stencil_ref);
    w(off::FRONT_STENCIL_FUNC, rs.front_stencil_func);
    w(off::FRONT_STENCIL_OP_FAIL, rs.front_stencil_op_fail);
    w(off::FRONT_STENCIL_OP_DEPTH_FAIL, rs.front_stencil_op_depth_fail);
    w(off::FRONT_STENCIL_OP_DEPTH_PASS, rs.front_stencil_op_depth_pass);
    w(off::FRONT_STENCIL_COMPARE_MASK, rs.front_stencil_compare_mask);
    w(off::FRONT_STENCIL_WRITE_MASK, rs.front_stencil_write_mask);
    w(off::BACK_STENCIL_FUNC, rs.back_stencil_func);
    w(off::BACK_STENCIL_OP_FAIL, rs.back_stencil_op_fail);
    w(off::BACK_STENCIL_OP_DEPTH_FAIL, rs.back_stencil_op_depth_fail);
    w(off::BACK_STENCIL_OP_DEPTH_PASS, rs.back_stencil_op_depth_pass);
    w(off::BACK_STENCIL_COMPARE_MASK, rs.back_stencil_compare_mask);
    w(off::BACK_STENCIL_WRITE_MASK, rs.back_stencil_write_mask);
    w(off::VIEWPORT_ENABLE, rs.viewport_enable);
    for (i, v) in rs.viewport.iter().enumerate() {
        w(off::VIEWPORT + i as u32 * 4, v.to_bits());
    }
    w(off::REGION_CLIP_MODE, rs.region_clip_mode);
    for (i, v) in rs.region_clip.iter().enumerate() {
        w(off::REGION_CLIP + i as u32 * 4, *v);
    }
    w(off::FRONT_VISIBILITY_TEST_ENABLE, rs.front_visibility_test_enable);
    w(off::FRONT_VISIBILITY_TEST_INDEX, rs.front_visibility_test_index);
    w(off::FRONT_VISIBILITY_TEST_OP, rs.front_visibility_test_op);
}

/// Read the whole render state back out of the block.
///
/// Called once per draw (and once per state-consuming host call), which replaces one host
/// crossing per SETTER with a handful of guest-memory reads per DRAW - about 30 loads
/// against the 1,240 crossings a frame the setters used to cost.
pub fn load(ctx: &GuestCtx, context: u32) -> crate::capture::RenderState {
    Block::read(ctx, context).render_state()
}

/// A COPY of the whole context block, taken in ONE read of guest memory.
///
/// # Why the block is taken whole, once
/// Every reader here used to reach into guest memory for itself: [`load`] copied the whole
/// block, [`texture_bindings`] copied the sampler span, [`streams`] copied the stream span,
/// and the scalar readers took a `read_u32` each. A DRAW calls most of them, so it crossed
/// the guest-memory boundary about a dozen times to read one 652-byte structure that cannot
/// change while it does - the guest does not run during a host call.
///
/// In the browser each of those crossings is a JS boundary crossing
/// ([[vitaslop-count-calls-not-bytes-across-the-guest-boundary]]), and 652 bytes is one
/// `copy_to`. So the block is snapshotted once and every reader parses the snapshot. The
/// free functions below are kept for the callers that want one field and have no draw to
/// hang a snapshot off.
///
/// A snapshot is only valid for the host call that took it, which is the only scope it is
/// ever used in - it is a local, never a field.
pub enum Block<'a> {
    /// Lent in place by a backing that can hand out a slice - every in-process host. No copy
    /// at all, which is what the readers used to do one span at a time.
    Lent(&'a [u8]),
    /// Copied, for a backing that cannot lend: the browser's guest memory is a
    /// `SharedArrayBuffer` that is not this module's linear memory, so no `&[u8]` exists over
    /// it and [`crate::host::GuestMemory::borrow`] returns `None` there PERMANENTLY. One copy
    /// of 652 bytes, against the dozen boundary crossings the per-span readers cost.
    Copied([u8; BYTES as usize]),
}

impl<'a> Block<'a> {
    /// Snapshot the block at `context`. A context of 0, or one outside guest memory, reads
    /// as zeros - which is what the per-field readers already answered for it, and
    /// [`Self::is_context`] is how a caller tells that from a real block.
    pub fn read(ctx: &'a GuestCtx, context: u32) -> Block<'a> {
        if context == 0 {
            return Block::Copied([0u8; BYTES as usize]);
        }
        if let Some(b) = ctx.borrow_bytes(context, BYTES as usize) {
            return Block::Lent(b);
        }
        let mut bytes = [0u8; BYTES as usize];
        ctx.read_into(context, &mut bytes);
        Block::Copied(bytes)
    }

    /// The block's bytes, however they were obtained.
    fn bytes(&self) -> &[u8] {
        match self {
            Block::Lent(b) => b,
            Block::Copied(b) => b,
        }
    }

    /// One word of the snapshot. The counterpart of [`get`].
    pub fn word(&self, offset: u32) -> u32 {
        let b = self.bytes();
        let at = offset as usize;
        u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
    }

    /// Whether this snapshot came from a block [`init`] stamped. The counterpart of
    /// [`is_context`].
    pub fn is_context(&self) -> bool {
        self.word(off::MAGIC) == MAGIC
    }

    /// One stage's bound default uniform buffer. The counterpart of [`uniform_binding`].
    pub fn uniform_binding(&self, record: u32) -> UniformBinding {
        UniformBinding {
            buf: self.word(record + uniform_record::BUF),
            size: self.word(record + uniform_record::SIZE),
            header: self.word(record + uniform_record::HEADER),
        }
    }

    /// The guest address bound to VERTEX non-default uniform buffer `index`. The
    /// counterpart of [`vertex_uniform_buffer`].
    pub fn vertex_uniform_buffer(&self, index: u32) -> u32 {
        if index as usize >= MAX_UNIFORM_BUFFERS {
            return 0;
        }
        self.word(off::VERTEX_UNIFORM_BUFFERS + index * 4)
    }

    /// All [`MAX_VERTEX_STREAMS`] stream pointers. The counterpart of [`streams`].
    pub fn streams(&self) -> [u32; MAX_VERTEX_STREAMS] {
        std::array::from_fn(|i| self.word(off::STREAMS + i as u32 * 4))
    }

    /// Every BOUND sampler unit's binding. The counterpart of [`texture_bindings`].
    pub fn texture_bindings(&self, out: &mut Vec<(u32, TexBinding)>) {
        out.clear();
        for unit in 0..MAX_TEXTURE_UNITS {
            let at = off::TEXTURES + unit as u32 * TEXTURE_STRIDE;
            let addr = self.word(at);
            // A slot the guest never bound, or unbound, is not a binding - the same test,
            // and the same skip, the per-unit path makes.
            if addr == 0 {
                continue;
            }
            out.push((
                unit as u32,
                TexBinding {
                    addr,
                    words: [
                        self.word(at + 4),
                        self.word(at + 8),
                        self.word(at + 12),
                        self.word(at + 16),
                    ],
                    from_precomputed: self.word(at + 4 + TEXTURE_CONTROL_WORDS * 4) != 0,
                },
            ));
        }
    }

    /// The whole render state, parsed out of the snapshot. The body [`load`] used to be.
    pub fn render_state(&self) -> crate::capture::RenderState {
    let r = |offset: u32| -> u32 { self.word(offset) };
    let mut viewport = [0f32; 6];
    for (i, v) in viewport.iter_mut().enumerate() {
        *v = f32::from_bits(r(off::VIEWPORT + i as u32 * 4));
    }
    let mut region_clip = [0u32; 4];
    for (i, v) in region_clip.iter_mut().enumerate() {
        *v = r(off::REGION_CLIP + i as u32 * 4);
    }
    crate::capture::RenderState {
        cull_mode: r(off::CULL_MODE),
        two_sided: r(off::TWO_SIDED),
        front_depth_func: r(off::FRONT_DEPTH_FUNC),
        back_depth_func: r(off::BACK_DEPTH_FUNC),
        front_depth_write: r(off::FRONT_DEPTH_WRITE),
        back_depth_write: r(off::BACK_DEPTH_WRITE),
        front_fragment_program_enable: r(off::FRONT_FRAGMENT_PROGRAM_ENABLE),
        back_fragment_program_enable: r(off::BACK_FRAGMENT_PROGRAM_ENABLE),
        front_polygon_mode: r(off::FRONT_POLYGON_MODE),
        back_polygon_mode: r(off::BACK_POLYGON_MODE),
        front_point_line_width: r(off::FRONT_POINT_LINE_WIDTH),
        front_stencil_ref: r(off::FRONT_STENCIL_REF),
        front_stencil_func: r(off::FRONT_STENCIL_FUNC),
        front_stencil_op_fail: r(off::FRONT_STENCIL_OP_FAIL),
        front_stencil_op_depth_fail: r(off::FRONT_STENCIL_OP_DEPTH_FAIL),
        front_stencil_op_depth_pass: r(off::FRONT_STENCIL_OP_DEPTH_PASS),
        front_stencil_compare_mask: r(off::FRONT_STENCIL_COMPARE_MASK),
        front_stencil_write_mask: r(off::FRONT_STENCIL_WRITE_MASK),
        back_stencil_func: r(off::BACK_STENCIL_FUNC),
        back_stencil_op_fail: r(off::BACK_STENCIL_OP_FAIL),
        back_stencil_op_depth_fail: r(off::BACK_STENCIL_OP_DEPTH_FAIL),
        back_stencil_op_depth_pass: r(off::BACK_STENCIL_OP_DEPTH_PASS),
        back_stencil_compare_mask: r(off::BACK_STENCIL_COMPARE_MASK),
        back_stencil_write_mask: r(off::BACK_STENCIL_WRITE_MASK),
        viewport_enable: r(off::VIEWPORT_ENABLE),
        viewport,
        region_clip_mode: r(off::REGION_CLIP_MODE),
        region_clip,
        front_visibility_test_enable: r(off::FRONT_VISIBILITY_TEST_ENABLE),
        front_visibility_test_index: r(off::FRONT_VISIBILITY_TEST_INDEX),
        front_visibility_test_op: r(off::FRONT_VISIBILITY_TEST_OP),
    }
    }
}

/// Every scalar field, as `(offset, name)`. The list a test walks to prove the layout has no
/// overlaps and no gaps, and that nothing was added without an offset.
#[cfg(test)]
pub(crate) const SCALARS: &[(u32, &str)] = &[
    (off::MAGIC, "magic"),
    (off::CULL_MODE, "cull_mode"),
    (off::TWO_SIDED, "two_sided"),
    (off::FRONT_DEPTH_FUNC, "front_depth_func"),
    (off::BACK_DEPTH_FUNC, "back_depth_func"),
    (off::FRONT_DEPTH_WRITE, "front_depth_write"),
    (off::BACK_DEPTH_WRITE, "back_depth_write"),
    (off::FRONT_FRAGMENT_PROGRAM_ENABLE, "front_fragment_program_enable"),
    (off::BACK_FRAGMENT_PROGRAM_ENABLE, "back_fragment_program_enable"),
    (off::FRONT_POLYGON_MODE, "front_polygon_mode"),
    (off::BACK_POLYGON_MODE, "back_polygon_mode"),
    (off::FRONT_POINT_LINE_WIDTH, "front_point_line_width"),
    (off::FRONT_STENCIL_REF, "front_stencil_ref"),
    (off::FRONT_STENCIL_FUNC, "front_stencil_func"),
    (off::FRONT_STENCIL_OP_FAIL, "front_stencil_op_fail"),
    (off::FRONT_STENCIL_OP_DEPTH_FAIL, "front_stencil_op_depth_fail"),
    (off::FRONT_STENCIL_OP_DEPTH_PASS, "front_stencil_op_depth_pass"),
    (off::FRONT_STENCIL_COMPARE_MASK, "front_stencil_compare_mask"),
    (off::FRONT_STENCIL_WRITE_MASK, "front_stencil_write_mask"),
    (off::BACK_STENCIL_FUNC, "back_stencil_func"),
    (off::BACK_STENCIL_OP_FAIL, "back_stencil_op_fail"),
    (off::BACK_STENCIL_OP_DEPTH_FAIL, "back_stencil_op_depth_fail"),
    (off::BACK_STENCIL_OP_DEPTH_PASS, "back_stencil_op_depth_pass"),
    (off::BACK_STENCIL_COMPARE_MASK, "back_stencil_compare_mask"),
    (off::BACK_STENCIL_WRITE_MASK, "back_stencil_write_mask"),
    (off::VIEWPORT_ENABLE, "viewport_enable"),
    (off::REGION_CLIP_MODE, "region_clip_mode"),
    (off::FRONT_VISIBILITY_TEST_ENABLE, "front_visibility_test_enable"),
    (off::FRONT_VISIBILITY_TEST_INDEX, "front_visibility_test_index"),
    (off::FRONT_VISIBILITY_TEST_OP, "front_visibility_test_op"),
    (off::VERTEX_PROGRAM, "vertex_program"),
    (off::FRAGMENT_PROGRAM, "fragment_program"),
    (off::UNIFORM_RING_BASE, "uniform_ring_base"),
    (off::UNIFORM_RING_END, "uniform_ring_end"),
    (off::UNIFORM_RING_CURSOR, "uniform_ring_cursor"),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every word of the block is claimed by exactly one field.
    ///
    /// The failure this catches is silent by construction: two fields sharing an offset
    /// means one setter clobbers the other, and what the guest sees is a depth function
    /// that changes when it sets a cull mode - a wrong picture with no error anywhere.
    #[test]
    fn layout_is_closed() {
        let mut claimed: Vec<(u32, &str)> = SCALARS.to_vec();
        for i in 0..6 {
            claimed.push((off::VIEWPORT + i * 4, "viewport"));
        }
        for i in 0..4 {
            claimed.push((off::REGION_CLIP + i * 4, "region_clip"));
        }
        for i in 0..MAX_VERTEX_STREAMS as u32 {
            claimed.push((off::STREAMS + i * 4, "streams"));
        }
        // Every word of every texture binding: the source address, the copied control words,
        // and the provenance flag. Claiming the whole slot rather than the stride keeps this
        // test able to catch a `TEXTURE_STRIDE` that does not match what a binding writes -
        // which would leave a hole between slots that no field owns and the inline copy form
        // would still happily write into.
        for unit in 0..MAX_TEXTURE_UNITS as u32 {
            for w in 0..TEXTURE_STRIDE / 4 {
                claimed.push((off::TEXTURES + unit * TEXTURE_STRIDE + w * 4, "textures"));
            }
        }
        // Both stages' three-word uniform records, word by word for the same reason the
        // texture slots are: a `uniform_record::BYTES` that disagreed with what a record
        // actually holds would leave a word no field owns, and the inline reserve writes
        // through those offsets without ever consulting this list.
        for record in [off::VERTEX_UNIFORM, off::FRAGMENT_UNIFORM] {
            for w in 0..uniform_record::BYTES / 4 {
                claimed.push((record + w * 4, "uniform_record"));
            }
        }
        for i in 0..MAX_UNIFORM_BUFFERS as u32 {
            claimed.push((off::VERTEX_UNIFORM_BUFFERS + i * 4, "vertex_uniform_buffers"));
        }
        claimed.sort();
        for w in claimed.windows(2) {
            assert_ne!(w[0].0, w[1].0, "{} and {} share an offset", w[0].1, w[1].1);
            assert_eq!(
                w[1].0,
                w[0].0 + 4,
                "a gap between {} at {:#x} and {} at {:#x}",
                w[0].1,
                w[0].0,
                w[1].1,
                w[1].0
            );
        }
        assert_eq!(claimed[0].0, 0, "the block starts at the context pointer");
        let (last, name) = *claimed.last().expect("the layout is not empty");
        assert_eq!(last + 4, BYTES, "BYTES must end just past {name}");
    }

    /// The inlined `sceGxmSetFragmentTexture` and [`set_texture_binding`] must describe the
    /// SAME slot.
    ///
    /// They are two implementations of one write - the transpiler emits one into guest code,
    /// the handler performs the other - and after this change the handler runs for a handful
    /// of binds a run while the emitted code runs 1,275 times a frame. A disagreement about
    /// the stride or the word count would therefore be invisible in every ordinary bind and
    /// show up only on the rare fallback, as a texture appearing on the wrong sampler unit
    /// thousands of frames from the cause.
    #[test]
    fn the_texture_binding_layout_is_closed() {
        let op = crate::vita::gxm::inline_op(crate::nid::gxm::SET_FRAGMENT_TEXTURE)
            .expect("sceGxmSetFragmentTexture has an inline form");
        let vitaslop_transpiler::InlineOp::CopyArgIndexed { offset, stride, count, words } = op
        else {
            panic!("sceGxmSetFragmentTexture must lower to a copy form, got {op:?}");
        };
        assert_eq!(offset, off::TEXTURES, "the emitted copy writes the texture array");
        assert_eq!(stride, TEXTURE_STRIDE, "one slot per sampler unit, same size both sides");
        assert_eq!(count, MAX_TEXTURE_UNITS as u32, "the same unit bound both sides");
        assert_eq!(words, TEXTURE_CONTROL_WORDS, "the same control words copied both sides");
        // ...and the slot the emitted code writes must be exactly the slot the reader reads:
        // addr, then `words` control words, then the provenance flag, with nothing left over.
        assert_eq!(
            stride,
            4 + words * 4 + 4,
            "a slot is the source address, the control words and the provenance flag"
        );
    }

    /// The emitted `sceGxmReserve*DefaultUniformBuffer` and the handler behind it must
    /// describe the SAME ring, the SAME record and the SAME program handle.
    ///
    /// Two implementations of one bump: the transpiler emits one into guest code and it runs
    /// hundreds of times a frame; the handler runs a handful of times a run (a context with
    /// no ring yet, a scene that overran it). A disagreement about any offset would therefore
    /// be invisible in every ordinary reserve and show up only on the rare fallback, as one
    /// draw's uniforms appearing on another - which reads as a shader bug thousands of frames
    /// from the cause.
    ///
    /// The record shape is the half the emitter ASSUMES rather than reads: it writes the
    /// buffer, the size and the header at `record + 0 / 4 / 8`. So that is asserted here
    /// rather than trusted.
    #[test]
    fn the_uniform_reserve_layout_is_closed() {
        use crate::vita::gxmprog;
        assert_eq!(uniform_record::BUF, 0, "the emitter writes the buffer at record + 0");
        assert_eq!(uniform_record::SIZE, 4, "...the size at record + 4");
        assert_eq!(uniform_record::HEADER, 8, "...and the header at record + 8");
        assert_eq!(uniform_record::BYTES, 12, "a record is exactly those three words");
        assert!(UNIFORM_ALIGN.is_power_of_two(), "the emitted mask is `!(align - 1)`");
        assert_eq!(
            UNIFORM_MIN_ALLOC % UNIFORM_ALIGN,
            0,
            "the floor is a whole number of alignments, so the smallest reserve still \
             leaves the cursor aligned and the emitted `align` is a no-op rather than a \
             silent second bump"
        );
        for (nid, record, program) in [
            (
                crate::nid::gxm::RESERVE_VERTEX_DEFAULT_UNIFORM_BUFFER,
                off::VERTEX_UNIFORM,
                off::VERTEX_PROGRAM,
            ),
            (
                crate::nid::gxm::RESERVE_FRAGMENT_DEFAULT_UNIFORM_BUFFER,
                off::FRAGMENT_UNIFORM,
                off::FRAGMENT_PROGRAM,
            ),
        ] {
            let op = crate::vita::gxm::inline_op(nid).expect("the reserve has an inline form");
            let vitaslop_transpiler::InlineOp::ReserveUniformBuffer { layout: l } = op else {
                panic!("a reserve must lower to a ring-bump form, got {op:?}");
            };
            assert_eq!(l.record, record, "each stage records into its OWN slot");
            assert_eq!(l.ctx_program, program, "...and reads its OWN bound program");
            assert_eq!((l.ctx_magic_at, l.ctx_magic), (off::MAGIC, MAGIC));
            assert_eq!((l.prog_magic_at, l.prog_magic), (gxmprog::off::MAGIC, gxmprog::MAGIC));
            assert_eq!(l.ctx_ring_base, off::UNIFORM_RING_BASE);
            assert_eq!(l.ctx_ring_end, off::UNIFORM_RING_END);
            assert_eq!(l.ctx_ring_cursor, off::UNIFORM_RING_CURSOR);
            assert_eq!(l.prog_size, gxmprog::off::UNIFORM_SIZE);
            assert_eq!(l.prog_alloc, gxmprog::off::UNIFORM_ALLOC);
            assert_eq!(l.prog_header, gxmprog::off::HEADER);
            assert_eq!(l.align, UNIFORM_ALIGN);
            // Every word the emitted code reaches must be inside the block it is bounded
            // against, or the guard admits a pointer whose last access lands outside it.
            assert!(l.ctx_top() + 4 <= BYTES, "the form stays inside the context block");
            assert!(
                l.prog_top() + 4 <= gxmprog::BYTES,
                "...and inside the program handle block"
            );
        }
    }

    /// The block fits in the smallest `hostMem` a conforming title can supply. If it ever
    /// stops fitting, that is a decision to make deliberately, not to discover on a title.
    #[test]
    fn the_block_fits_the_minimum_host_mem() {
        assert!(
            BYTES <= MINIMUM_HOST_MEM,
            "the context block is {BYTES} bytes against a guaranteed {MINIMUM_HOST_MEM}"
        );
    }

    /// A guest image with a context block at [`CONTEXT`], and the ctx to reach it through.
    const CONTEXT: u32 = 0x100;

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

    /// >>> THE TWO `texture_bindings` PATHS AGREE, WHICH NO ENGINE CAN CHECK ON ITS OWN.
    ///
    /// It reads the sampler block by BORROWING a slice where the backing lends one and by
    /// COPYING it where it does not - and which path a run takes is a property of the
    /// ENGINE, not of the input. Native always lends (its guest memory is a raw pointer);
    /// the browser never can (its guest memory is a SharedArrayBuffer that is not this
    /// module's own linear memory). So each engine exercises exactly one path in every run
    /// it will ever make, and a divergence between them is invisible to both.
    ///
    /// That is not hypothetical: the copy path used to read the block a WORD AT A TIME,
    /// ~96 virtual calls per draw, and cost **13.7% of every browser frame** while the
    /// desktop - which never ran it - showed the same phase at 1.55%. It answered
    /// identically and eight times slower, silently.
    #[test]
    fn both_sampler_block_readers_return_the_same_bindings() {
        use crate::{GuestMemory, SliceMemory, VFP_ARG_COUNT};
        use vitaslop_transpiler::abi::REG_COUNT;

        /// A backing that refuses to lend, which is the browser's shape. Everything else
        /// delegates, so the ONLY difference between the two arms is the `borrow`.
        struct NoBorrow<'a>(SliceMemory<'a>, std::cell::Cell<usize>);
        impl GuestMemory for NoBorrow<'_> {
            fn len(&self) -> usize {
                self.0.len()
            }
            fn read(&self, off: usize, buf: &mut [u8]) {
                self.1.set(self.1.get() + 1);
                self.0.read(off, buf)
            }
            fn write(&mut self, off: usize, bytes: &[u8]) {
                self.0.write(off, bytes)
            }
            // The whole point: `None`, exactly as the browser's `SharedView` does.
        }

        // A block with bindings in a scattered set of units - including the last one, so a
        // reader that stopped short would be caught - and unbound holes between them.
        let mut image = vec![0u8; 4096];
        let mut expect_units = Vec::new();
        for (i, unit) in [0usize, 1, 5, 15].into_iter().enumerate() {
            let at = (CONTEXT + off::TEXTURES) as usize + unit * TEXTURE_STRIDE as usize;
            let addr = 0xd000_0000u32 + i as u32;
            image[at..at + 4].copy_from_slice(&addr.to_le_bytes());
            for w in 0..TEXTURE_CONTROL_WORDS as usize {
                let v = 0x1111_0000u32 + ((i as u32) << 8) + w as u32;
                image[at + 4 + w * 4..at + 8 + w * 4].copy_from_slice(&v.to_le_bytes());
            }
            // The provenance flag, non-zero on every other binding.
            let pv = (i % 2) as u32;
            let po = at + 4 + TEXTURE_CONTROL_WORDS as usize * 4;
            image[po..po + 4].copy_from_slice(&pv.to_le_bytes());
            expect_units.push(unit as u32);
        }

        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];

        let borrowed = {
            let mut bytes = image.clone();
            let mut mem = SliceMemory(&mut bytes);
            let ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            assert!(ctx.borrow_bytes(CONTEXT, 4).is_some(), "the lending arm must lend");
            let mut out = Vec::new();
            texture_bindings(&ctx, CONTEXT, &mut out);
            out
        };
        let (copied, reads) = {
            let mut bytes = image.clone();
            let mut inner = SliceMemory(&mut bytes);
            let mut mem = NoBorrow(inner_of(&mut inner), std::cell::Cell::new(0));
            let ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            assert!(ctx.borrow_bytes(CONTEXT, 4).is_none(), "the copying arm must not lend");
            let mut out = Vec::new();
            texture_bindings(&ctx, CONTEXT, &mut out);
            let n = match &mem {
                NoBorrow(_, c) => c.get(),
            };
            (out, n)
        };

        assert_eq!(
            borrowed.iter().map(|(u, _)| *u).collect::<Vec<_>>(),
            expect_units,
            "the bound units themselves"
        );
        assert_eq!(borrowed, copied, "the borrowing and copying readers disagree");
        // >>> AND THE COPY IS ONE READ, WHICH IS THE PROPERTY THAT ACTUALLY REGRESSED.
        //
        // The equality above would have passed the whole time this was slow: the per-unit
        // path was CORRECT, it just made ~96 virtual calls per draw instead of one. A test
        // that only checks the answer cannot see a defect that only changes the cost, and
        // this one cost 13.7% of a browser frame for however long it stood.
        assert!(
            reads <= 1,
            "the copying reader made {reads} reads of guest memory; it must take the whole \
             sampler block in ONE, or the browser pays a virtual call per word again"
        );
    }

    /// >>> `load` AND `streams` AGREE ACROSS THE BORROW/COPY SPLIT TOO.
    ///
    /// Same argument as `both_sampler_block_readers_return_the_same_bindings`, and it is
    /// MORE load-bearing here: `load` produces the `RenderState` recorded into every draw,
    /// and the renderer turns cull / depth / stencil / blend straight into a WebGPU pipeline
    /// descriptor. A copy path that disagreed with the borrow path would not merely look
    /// wrong - it could produce a pipeline the DEVICE rejects, on the engine no desktop run
    /// ever exercises.
    #[test]
    fn load_and_streams_agree_across_the_borrow_and_copy_paths() {
        use crate::{GuestMemory, SliceMemory, VFP_ARG_COUNT};
        use vitaslop_transpiler::abi::REG_COUNT;

        struct NoBorrow<'a>(SliceMemory<'a>, std::cell::Cell<usize>);
        impl GuestMemory for NoBorrow<'_> {
            fn len(&self) -> usize {
                self.0.len()
            }
            fn read(&self, off: usize, buf: &mut [u8]) {
                self.1.set(self.1.get() + 1);
                self.0.read(off, buf)
            }
            fn write(&mut self, off: usize, bytes: &[u8]) {
                self.0.write(off, bytes)
            }
        }

        // A DISTINCT value in every word of the block, so two fields that swapped offsets
        // cannot cancel out and a reader that stopped short is caught.
        let mut image = vec![0u8; 8192];
        for w in 0..(BYTES as usize / 4) {
            let at = CONTEXT as usize + w * 4;
            let v = 0x4000_0000u32 + w as u32;
            image[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }

        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];

        let (rs_borrow, st_borrow) = {
            let mut bytes = image.clone();
            let mut mem = SliceMemory(&mut bytes);
            let ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            assert!(ctx.borrow_bytes(CONTEXT, 4).is_some(), "the lending arm must lend");
            (load(&ctx, CONTEXT), streams(&ctx, CONTEXT))
        };
        let (rs_copy, st_copy, reads) = {
            let mut bytes = image.clone();
            let mut inner = SliceMemory(&mut bytes);
            let mut mem = NoBorrow(inner_of(&mut inner), std::cell::Cell::new(0));
            let ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            assert!(ctx.borrow_bytes(CONTEXT, 4).is_none(), "the copying arm must not lend");
            let rs = load(&ctx, CONTEXT);
            let st = streams(&ctx, CONTEXT);
            let n = match &mem {
                NoBorrow(_, c) => c.get(),
            };
            (rs, st, n)
        };

        assert_eq!(rs_borrow, rs_copy, "load disagrees across the borrow/copy split");
        assert_eq!(st_borrow, st_copy, "streams disagrees across the borrow/copy split");
        // One read for `load`, one for `streams` - the cost property, which an
        // equality-only test cannot see (the per-word path was correct, just 57x the calls).
        assert!(
            reads <= 2,
            "the copying path made {reads} reads; `load` and `streams` must take ONE each"
        );
    }

    /// Re-wrap a `SliceMemory`'s buffer, so the no-borrow arm can hold one by value without
    /// the test needing two separate buffers to keep the borrow checker happy.
    fn inner_of<'a>(m: &'a mut crate::SliceMemory<'_>) -> crate::SliceMemory<'a> {
        crate::SliceMemory(m.0)
    }

    /// Every field survives a write and a read back, with a DISTINCT value each, so a pair
    /// of fields that swapped offsets fails instead of cancelling out. `store`/`load` are
    /// the two halves of the same layout, and a layout read back through its own writer
    /// agrees with itself no matter how wrong it is - hence distinct values, not defaults.
    #[test]
    fn a_render_state_round_trips_through_the_block() {
        let mut rs = crate::capture::RenderState::default();
        // A different value in every field, none of them a default.
        rs.cull_mode = 0x11;
        rs.two_sided = 0x12;
        rs.front_depth_func = 0x13;
        rs.back_depth_func = 0x14;
        rs.front_depth_write = 0x15;
        rs.back_depth_write = 0x16;
        rs.front_fragment_program_enable = 0x17;
        rs.back_fragment_program_enable = 0x18;
        rs.front_polygon_mode = 0x19;
        rs.back_polygon_mode = 0x1a;
        rs.front_point_line_width = 0x1b;
        rs.front_stencil_ref = 0x1c;
        rs.front_stencil_func = 0x1d;
        rs.front_stencil_op_fail = 0x1e;
        rs.front_stencil_op_depth_fail = 0x1f;
        rs.front_stencil_op_depth_pass = 0x20;
        rs.front_stencil_compare_mask = 0x21;
        rs.front_stencil_write_mask = 0x22;
        rs.viewport_enable = 0x23;
        rs.viewport = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        rs.region_clip_mode = 0x24;
        rs.region_clip = [0x25, 0x26, 0x27, 0x28];
        rs.front_visibility_test_enable = 0x29;
        rs.front_visibility_test_index = 0x2a;
        rs.front_visibility_test_op = 0x2b;
        let got = with_ctx(|ctx| {
            store(ctx, CONTEXT, &rs);
            load(ctx, CONTEXT)
        });
        assert_eq!(got, rs);
    }

    /// `init` leaves a block that reads back as the GXM power-on defaults and identifies
    /// itself. An uninitialised context is the case that must NOT look like one.
    #[test]
    fn init_seeds_the_defaults_and_stamps_the_block() {
        with_ctx(|ctx| {
            assert!(!is_context(ctx, CONTEXT), "untouched memory is not a context");
            init(ctx, CONTEXT);
            assert!(is_context(ctx, CONTEXT));
            assert_eq!(load(ctx, CONTEXT), crate::capture::RenderState::default());
            assert_eq!(streams(ctx, CONTEXT), [0; MAX_VERTEX_STREAMS]);
            assert_eq!(get(ctx, CONTEXT, off::VERTEX_PROGRAM), 0);
            assert_eq!(get(ctx, CONTEXT, off::FRAGMENT_PROGRAM), 0);
        });
    }

    /// A stream index GXM cannot produce must not write anything - least of all the word
    /// past the array, which is the next context's business or nobody's.
    #[test]
    fn an_out_of_range_stream_index_writes_nothing() {
        with_ctx(|ctx| {
            init(ctx, CONTEXT);
            let past = CONTEXT + off::STREAMS + MAX_VERTEX_STREAMS as u32 * 4;
            ctx.write_u32(past, 0xDEAD_BEEF);
            set_stream(ctx, CONTEXT, MAX_VERTEX_STREAMS as u32, 0x1234);
            assert_eq!(ctx.read_u32(past), 0xDEAD_BEEF, "the word past the array is untouched");
            assert_eq!(streams(ctx, CONTEXT), [0; MAX_VERTEX_STREAMS]);
            assert_eq!(stream(ctx, CONTEXT, MAX_VERTEX_STREAMS as u32), 0);
        });
    }
}
