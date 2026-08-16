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
//!      desktop where it barely registers. Measured on PCSA00027 at 248 calls per frame
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
//! - **The default-uniform-buffer reserves.** `sceGxmReserve*DefaultUniformBuffer` allocates
//!   and sizes a buffer from the bound program's reflected interface. That is real work, not
//!   an accessor.
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
}

/// `SCE_GXM_MAX_TEXTURE_UNITS` (vitasdk `gxm.h`).
pub const MAX_TEXTURE_UNITS: usize = 16;

/// Bytes per fragment texture binding: `addr`, four control words, `from_precomputed`.
pub const TEXTURE_STRIDE: u32 = 24;

/// Words of a binding that are COPIED from the guest's `SceGxmTexture` - the control words
/// themselves. The inline form and [`set_texture_binding`] must copy the same number.
pub const TEXTURE_CONTROL_WORDS: u32 = 4;

/// `SCE_GXM_MAX_VERTEX_STREAMS` (vitasdk `gxm.h`).
pub const MAX_VERTEX_STREAMS: usize = 16;

/// Total bytes the block occupies. Every guest context must have at least this much host
/// memory behind it.
pub const BYTES: u32 = off::TEXTURES + (MAX_TEXTURE_UNITS as u32) * TEXTURE_STRIDE;

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
    let mut out = [0u32; MAX_VERTEX_STREAMS];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = get(ctx, context, off::STREAMS + i as u32 * 4);
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
    let r = |offset| get(ctx, context, offset);
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
        viewport_enable: r(off::VIEWPORT_ENABLE),
        viewport,
        region_clip_mode: r(off::REGION_CLIP_MODE),
        region_clip,
        front_visibility_test_enable: r(off::FRONT_VISIBILITY_TEST_ENABLE),
        front_visibility_test_index: r(off::FRONT_VISIBILITY_TEST_INDEX),
        front_visibility_test_op: r(off::FRONT_VISIBILITY_TEST_OP),
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
    (off::VIEWPORT_ENABLE, "viewport_enable"),
    (off::REGION_CLIP_MODE, "region_clip_mode"),
    (off::FRONT_VISIBILITY_TEST_ENABLE, "front_visibility_test_enable"),
    (off::FRONT_VISIBILITY_TEST_INDEX, "front_visibility_test_index"),
    (off::FRONT_VISIBILITY_TEST_OP, "front_visibility_test_op"),
    (off::VERTEX_PROGRAM, "vertex_program"),
    (off::FRAGMENT_PROGRAM, "fragment_program"),
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
