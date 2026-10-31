use super::*;

/// Control words for a 128x64 U8U8U8U8 texture of `tex_type` at 0x1000, with the given
/// word-0 filter fields.
fn words(tex_type: u32, mag: u32, min_bits: u32) -> [u32; 4] {
    let w0 = (mag << 12) | (min_bits << 10);
    let w1 = (tex_type << 29) | (0x0c << 24) | ((128 - 1) << 12) | (64 - 1);
    [w0, w1, 0x1000, 0]
}

/// **A LINEAR_STRIDED TEXTURE'S ROWS ARE ITS EXPLICIT STRIDE APART.** MEASURED on a fighting
/// title: a 128x64 window onto a 512-wide surface, sampled with a 2048-byte stride. Taken at
/// the width's pitch (512) every row after the first came from the wrong surface row.
#[test]
fn a_strided_texture_takes_its_recorded_byte_stride() {
    let fmt = Some(0x0c00_0000);
    let t = build_texture_template(0, words(6, 1, 0), fmt, 2048).expect("template");
    assert_eq!(t.stride, 2048, "the recorded pitch");
    assert!(
        t.read_len >= 2048 * 63 + 128 * 4,
        "reads the last row at its pitch (read_len {})",
        t.read_len
    );
    // With no recorded stride it falls back to the width-derived pitch, as before.
    let t0 = build_texture_template(0, words(6, 1, 0), fmt, 0).expect("template");
    assert_eq!(t0.stride, 128 * 4);
    // A NON-strided texture ignores a stray recorded stride.
    let lin = build_texture_template(0, words(3, 1, 0), fmt, 2048).expect("template");
    assert_eq!(lin.stride, 128 * 4);
}

/// **A BY-VALUE COPY OF A STRIDED TEXTURE KEEPS ITS STRIDE.** A title that copies the 16 bytes
/// into its own material and binds the copy sampled a 992-pixel-pitch render target at 960 x 4
/// while the stride was held beside the struct, and every row slid 32 pixels into slanted
/// stripes. It is packed into word 0 now, so the copy carries it.
#[test]
fn a_copied_strided_texture_keeps_its_stride() {
    use crate::world::DeterministicWorld;
    use crate::{SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::REG_COUNT;
    const ORIG: u32 = 0x100;
    const COPY: u32 = 0x200;
    let mut st = VitaState::new(0, 0x1000, Box::new(DeterministicWorld::default()));
    let mut regs = [0u32; REG_COUNT];
    let mut vfp = [0u32; VFP_ARG_COUNT];
    let mut bytes = vec![0u8; 0x1000];
    let mut mem = SliceMemory(&mut bytes);
    let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
    // sceGxmTextureInitLinearStrided(tex, data, U8U8U8U8, 960, 544, stride 3968)
    (ctx.regs[0], ctx.regs[1], ctx.regs[2], ctx.regs[3]) =
        (ORIG, 0x8000_0000, 0x0c00_0000, 960);
    ctx.regs[13] = 0x800;
    ctx.write_u32(0x800, 544);
    ctx.write_u32(0x804, 3968);
    crate::vita::dispatch(
        crate::nid::lib::SCE_GXM,
        crate::nid::gxm::TEXTURE_INIT_LINEAR_STRIDED,
        &mut ctx,
        &mut st,
    );
    assert_eq!(
        st.linear_strided_stride(&ctx, ORIG),
        3968,
        "the struct it was given"
    );
    let words = ctx.read_bytes(ORIG, 16);
    ctx.write_bytes(COPY, &words);
    assert_eq!(
        st.linear_strided_stride(&ctx, COPY),
        3968,
        "a by-value copy"
    );
    // A linear texture over the same data takes no stride from the strided one.
    ctx.write_u32(
        COPY + 4,
        (ctx.read_u32(COPY + 4) & !(0x7 << 29)) | (3 << 29),
    );
    assert_eq!(st.linear_strided_stride(&ctx, COPY), 0);
    // Two views of ONE surface - same format, size and data - at different pitches keep
    // their own: the stride is in each struct, not in a table keyed by what they share.
    const OTHER: u32 = 0x300;
    ctx.regs[0] = OTHER;
    (ctx.regs[1], ctx.regs[2], ctx.regs[3]) = (0x8000_0000, 0x0c00_0000, 960);
    ctx.write_u32(0x804, 3840);
    crate::vita::dispatch(
        crate::nid::lib::SCE_GXM,
        crate::nid::gxm::TEXTURE_INIT_LINEAR_STRIDED,
        &mut ctx,
        &mut st,
    );
    let words = ctx.read_bytes(ORIG, 16);
    ctx.write_bytes(COPY, &words);
    assert_eq!(
        st.linear_strided_stride(&ctx, COPY),
        3968,
        "the first view's copy"
    );
    assert_eq!(
        st.linear_strided_stride(&ctx, OTHER),
        3840,
        "the second view"
    );
    // Every stride the words can carry round-trips through them.
    for s in [4, 3840, 3968, 4096, crate::vita::gxm::MAX_PACKED_STRIDE] {
        assert_eq!(
            crate::vita::gxm::linear_stride_from_word0(crate::vita::gxm::pack_linear_stride(s)),
            s
        );
    }
    // One that cannot be packed is refused, not truncated.
    ctx.regs[0] = OTHER;
    (ctx.regs[1], ctx.regs[2], ctx.regs[3]) = (0x8000_0000, 0x0c00_0000, 960);
    ctx.write_u32(0x804, crate::vita::gxm::MAX_PACKED_STRIDE + 4);
    crate::vita::dispatch(
        crate::nid::lib::SCE_GXM,
        crate::nid::gxm::TEXTURE_INIT_LINEAR_STRIDED,
        &mut ctx,
        &mut st,
    );
    assert!((ctx.regs[0] as i32) < 0, "an unpackable stride must fail");
}

/// **A STRIDED TEXTURE MINIFIES WITH ITS MAG FILTER**: its bits 10..11 are stride bits. The
/// type selector for LINEAR_STRIDED is 6 - it was compared against 1, no type at all.
#[test]
fn a_strided_texture_minifies_with_its_mag_filter() {
    let (_, _, _, min, mag, _, _) = word0_sampler_state(words(6, 1, 2)[0], 6);
    assert_eq!((min, mag), (1, 1));
    let (_, _, _, min, mag, _, _) = word0_sampler_state(words(3, 1, 2)[0], 3);
    assert_eq!((min, mag), (2, 1));
}
