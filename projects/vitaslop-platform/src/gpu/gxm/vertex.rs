//! Vertex fetch and per-draw program facts: reading attributes the way the hardware does,
//! the WebGPU formats that fetch them in place, the vertex-plan census, the content hash,
//! and the dual-source / colour-no-op / destination-read questions asked of a fragment blob.

use super::*;

/// Every texel of a decoded texture as four floats, read through its own seam.
///
/// The byte seam's values are `n/255`, so the two seams print on one scale and a reader does
/// not have to know which one a given texture landed on to compare two of them.
pub(super) fn seam_texels(t: &GxmTexture) -> Vec<[f32; 4]> {
    let bpt = t.texel.bytes_per_texel();
    t.rgba
        .chunks_exact(bpt)
        .map(|c| match t.texel {
            TexelSeam::Rgba8 => [
                c[0] as f32 / 255.0,
                c[1] as f32 / 255.0,
                c[2] as f32 / 255.0,
                c[3] as f32 / 255.0,
            ],
            TexelSeam::Rgba16Float => {
                let h = |i: usize| half_to_f32(u16::from_le_bytes([c[i * 2], c[i * 2 + 1]]));
                [h(0), h(1), h(2), h(3)]
            }
            TexelSeam::Rg32Float => {
                let f = |i: usize| f32::from_le_bytes([c[i], c[i + 1], c[i + 2], c[i + 3]]);
                [f(0), f(4), 0.0, 1.0]
            }
        })
        .collect()
}

/// Per-channel min/max over every texel, read through the texture's own seam.
///
/// "The first four texels" answers what a texture looks like at its corner; it does not
/// answer whether a channel carries any signal at all, and for a texture a shader reads as
/// DATA that is the only question. The map body's vertex program displaces every vertex by
/// this texture's ALPHA, so `a[0,0]` and `a[0,1]` are two completely different bugs and the
/// corner texel cannot tell them apart.
pub(super) fn channel_spread_seam(t: &GxmTexture) -> String {
    let texels = seam_texels(t);
    if texels.is_empty() {
        return "spread=(no texels)".into();
    }
    let mut lo = [f32::INFINITY; 4];
    let mut hi = [f32::NEG_INFINITY; 4];
    for c in &texels {
        for i in 0..4 {
            lo[i] = lo[i].min(c[i]);
            hi[i] = hi[i].max(c[i]);
        }
    }
    format!(
        "spread over {} texels r[{:.4},{:.4}] g[{:.4},{:.4}] b[{:.4},{:.4}] a[{:.4},{:.4}]",
        texels.len(),
        lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], lo[3], hi[3]
    )
}

/// Byte size of one component of a `SceGxmAttributeFormat`.
pub(super) fn attr_component_size(gxm_format: u8) -> usize {
    match gxm_format {
        0 | 1 | 4 | 5 => 1,     // U8 / S8 / U8N / S8N
        2 | 3 | 6 | 7 | 8 => 2, // U16 / S16 / U16N / S16N / F16
        _ => 4,                 // F32 (9) and any unknown -> treat as 4-byte float
    }
}

/// Read one attribute component from the guest stream and convert it to f32, matching the
/// GXM fixed-function vertex fetch (normalized formats scale to [0,1]/[-1,1], F16 expands).
/// Out-of-range reads yield 0.0 (a benign over-read on a short buffer, never a panic).
pub(super) fn read_attr_component(buf: &[u8], base: usize, gxm_format: u8, c: usize) -> f32 {
    let o = base + c * attr_component_size(gxm_format);
    let u16at = |o: usize| -> Option<u16> { buf.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]])) };
    match gxm_format {
        // F32, and UNTYPED (10): a 32-bit word the fetch hands the shader UNCONVERTED, so the
        // f32 carrying exactly those bits is that word. MEASURED on a fighting title:
        // its engine clears a target with a fullscreen triangle whose only
        // attribute is UNTYPED x2 - float positions. Reading it as 0.0 collapsed the triangle
        // to a point, the clear drew nothing, and a map the game clears to alpha 1 stayed at
        // alpha 0 - which its character shader multiplies every surface colour by.
        9 | 10 => buf.get(o..o + 4).map(|s| f32::from_le_bytes([s[0], s[1], s[2], s[3]])).unwrap_or(0.0),
        8 => u16at(o).map(half_to_f32).unwrap_or(0.0),
        4 => buf.get(o).map(|&b| b as f32 / 255.0).unwrap_or(0.0),
        5 => buf.get(o).map(|&b| (b as i8 as f32 / 127.0).max(-1.0)).unwrap_or(0.0),
        6 => u16at(o).map(|v| v as f32 / 65535.0).unwrap_or(0.0),
        7 => u16at(o).map(|v| (v as i16 as f32 / 32767.0).max(-1.0)).unwrap_or(0.0),
        0 => buf.get(o).map(|&b| b as f32).unwrap_or(0.0),
        1 => buf.get(o).map(|&b| b as i8 as f32).unwrap_or(0.0),
        2 => u16at(o).map(|v| v as f32).unwrap_or(0.0),
        3 => u16at(o).map(|v| v as i16 as f32).unwrap_or(0.0),
        _ => 0.0,
    }
}

/// The WebGPU vertex format that fetches a GXM attribute of this format and component count
/// **exactly as [`read_attr_component`] would** - or `None` when there is none, which is what
/// puts the attribute (and so its whole stream) back on the repack.
///
/// # Why this is a table and not a rule
/// The vertex stage converts these formats to `f32` in hardware, with the same arithmetic the
/// function above spells out: `unorm` divides by the type's maximum, `snorm` divides by its
/// maximum POSITIVE value and clamps the extra negative code to -1, `float16` expands an IEEE
/// half. So for the formats listed here the repack is a pure copy, and the WGSL - which reads
/// every attribute as `vec4<f32>` whatever it is bound to - does not change at all.
///
/// Three families are deliberately absent, each for its own reason:
///   * the **plain integer** formats (`U8`/`S8`/`U16`/`S16`, GXM 0..3). WebGPU has `uint8x2`
///     and friends, but they deliver `u32`/`i32` to the shader and this module's attributes
///     are `vec4<f32>`. Passing them through would mean changing the emitted WGSL, which is a
///     different change with a different blast radius.
///   * **three-component** narrow formats. WebGPU has no `unorm8x3`, `float16x3`, ... at all;
///     that gap is the reason the repack exists.
///   * anything **unknown** (GXM > 9), which `read_attr_component` reads as 0.0. A format
///     nobody has established must not be silently handed to the hardware to interpret.
pub(super) fn passthrough_vertex_format(gxm_format: u8, components: u8) -> Option<wgpu::VertexFormat> {
    use wgpu::VertexFormat as V;
    Some(match (gxm_format, components) {
        // F32 - the one family with every width.
        (9, 1) => V::Float32,
        (9, 2) => V::Float32x2,
        (9, 3) => V::Float32x3,
        (9, 4) => V::Float32x4,
        // F16.
        (8, 2) => V::Float16x2,
        (8, 4) => V::Float16x4,
        // U8N / S8N.
        (4, 2) => V::Unorm8x2,
        (4, 4) => V::Unorm8x4,
        (5, 2) => V::Snorm8x2,
        (5, 4) => V::Snorm8x4,
        // U16N / S16N.
        (6, 2) => V::Unorm16x2,
        (6, 4) => V::Unorm16x4,
        (7, 2) => V::Snorm16x2,
        (7, 4) => V::Snorm16x4,
        _ => return None,
    })
}

/// The two vertex plans must FETCH THE SAME NUMBERS, and that is checkable on the CPU.
///
/// The picture is the real oracle and a title sweep is what proves it, but a sweep costs an
/// hour and only says "these frames differ". These say WHICH conversion disagrees, in a
/// second, and they are the half of the claim that a picture cannot see: a wrong entry in
/// [`passthrough_vertex_format`] - `Snorm16x2` where the guest means `Unorm16x2`, say -
/// is a smooth, plausible, entirely wrong set of values, not a broken frame.
#[cfg(test)]
mod vertex_passthrough_tests {
    /// What the WebGPU vertex stage does with each mapped format, written out from the spec
    /// rather than from [`super::read_attr_component`], so the two are independent readings
    /// that have to agree.
    fn webgpu_fetch(format: wgpu::VertexFormat, bytes: &[u8], c: usize) -> f32 {
        use wgpu::VertexFormat as V;
        let u16at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
        match format {
            V::Float32 | V::Float32x2 | V::Float32x3 | V::Float32x4 => {
                f32::from_le_bytes(bytes[c * 4..c * 4 + 4].try_into().unwrap())
            }
            // An IEEE half, expanded.
            V::Float16x2 | V::Float16x4 => super::half_to_f32(u16at(c * 2)),
            // unorm: divide by the type's maximum. snorm: divide by the maximum POSITIVE
            // value, which leaves the most negative code one step out of range - and the
            // spec clamps it to -1 rather than letting it reach -1.0039.
            V::Unorm8x2 | V::Unorm8x4 => bytes[c] as f32 / 255.0,
            V::Snorm8x2 | V::Snorm8x4 => (bytes[c] as i8 as f32 / 127.0).max(-1.0),
            V::Unorm16x2 | V::Unorm16x4 => u16at(c * 2) as f32 / 65535.0,
            V::Snorm16x2 | V::Snorm16x4 => (u16at(c * 2) as i16 as f32 / 32767.0).max(-1.0),
            other => panic!("{other:?} is mapped but this test does not model it"),
        }
    }

    /// Every mapped `(GXM format, components)` fetches what the repack would have written,
    /// for EVERY byte pattern of the narrow formats and the awkward values of the wide ones.
    #[test]
    fn the_two_plans_fetch_the_same_values() {
        // Exhaustive for the 8-bit formats, and every 16-bit code would be 65,536 patterns
        // per lane - so those get the ends, the midpoint and the two that separate the
        // divisors: 32767/32768 (snorm's clamp) and 65535 (unorm's 1.0).
        let patterns: Vec<Vec<u8>> = (0..=255u8)
            .map(|b| vec![b, b ^ 0xa5, b.wrapping_add(1), b ^ 0x80])
            .chain([
                vec![0x00, 0x00, 0xff, 0xff],
                vec![0xff, 0x7f, 0x00, 0x80],
                vec![0x01, 0x80, 0xfe, 0x7f],
                // Two f32s worth of bytes, so the float arm sees real values too.
                1.5f32.to_le_bytes().to_vec(),
                (-0.0f32).to_le_bytes().to_vec(),
            ])
            .collect();
        for gxm in 0..=12u8 {
            for comps in 1..=4u8 {
                let Some(f) = super::passthrough_vertex_format(gxm, comps) else { continue };
                // A mapped format must carry exactly the components the guest binds - the
                // hardware fills the rest, and that is checked at the layout, not here.
                assert_eq!(
                    f.size() as usize,
                    comps as usize * super::attr_component_size(gxm),
                    "GXM {gxm} x{comps} is mapped to {f:?}, which is a different width"
                );
                for p in &patterns {
                    // Enough bytes for four components of the widest mapped format.
                    let mut buf = Vec::new();
                    while buf.len() < 16 {
                        buf.extend_from_slice(p);
                    }
                    for c in 0..comps as usize {
                        let repacked = super::read_attr_component(&buf, 0, gxm, c);
                        let fetched = webgpu_fetch(f, &buf, c);
                        assert_eq!(
                            repacked.to_bits(),
                            fetched.to_bits(),
                            "GXM {gxm} x{comps} ({f:?}) lane {c} of {:02x?}: the repack \
                             writes {repacked} and the hardware fetches {fetched}",
                            &buf[..8]
                        );
                    }
                }
            }
        }
    }

    /// No mapping for a format whose values the repack does not actually reproduce.
    ///
    /// The plain-integer formats are the trap: WebGPU HAS `uint8x2`/`sint16x4` and they look
    /// like the obvious match for GXM 0..3, but they deliver `u32`/`i32` to the shader while
    /// an attribute the emitted WGSL declares `vec4<f32>` needs a float fetch. A pipeline
    /// built that way is refused by the driver - and refused pairs are dropped draws.
    ///
    /// They are reachable now, but only through [`super::integer_vertex_format`] and only
    /// for an attribute the LINK declared integer-typed. This table stays float-only so the
    /// float path cannot reach one by accident.
    #[test]
    fn the_integer_and_odd_width_formats_stay_on_the_repack() {
        for gxm in [0u8, 1, 2, 3] {
            for comps in 1..=4u8 {
                assert!(
                    super::passthrough_vertex_format(gxm, comps).is_none(),
                    "GXM {gxm} is a plain INTEGER format and the shader reads f32"
                );
            }
        }
        // Narrow formats at one and three components: WebGPU has no such vertex format, and
        // that gap is the reason the repack exists at all.
        for gxm in [4u8, 5, 6, 7, 8] {
            for comps in [1u8, 3] {
                assert!(super::passthrough_vertex_format(gxm, comps).is_none());
            }
        }
        // Nothing beyond the established table, which `read_attr_component` reads as 0.0.
        for gxm in 10..=255u8 {
            for comps in 1..=4u8 {
                assert!(super::passthrough_vertex_format(gxm, comps).is_none());
            }
        }
    }

    /// A three-component narrow attribute is fetched by the FOUR-component format, and a
    /// one-component one by the two - the only widening WebGPU's format list allows, and
    /// only ever by ONE component, so the lanes a caller has to prove unread are bounded.
    #[test]
    fn a_narrow_attribute_widens_by_exactly_one_component() {
        use wgpu::VertexFormat as V;
        // F16: no x1 and no x3, so both odd widths step up.
        assert_eq!(super::widest_passthrough_format(8, 3), Some((V::Float16x4, 4)));
        assert_eq!(super::widest_passthrough_format(8, 1), Some((V::Float16x2, 2)));
        assert_eq!(super::widest_passthrough_format(8, 2), Some((V::Float16x2, 2)));
        assert_eq!(super::widest_passthrough_format(8, 4), Some((V::Float16x4, 4)));
        // U8N and S16N behave the same way; F32 never widens, because it has every width.
        assert_eq!(super::widest_passthrough_format(4, 3), Some((V::Unorm8x4, 4)));
        assert_eq!(super::widest_passthrough_format(7, 1), Some((V::Snorm16x2, 2)));
        for c in 1..=4u8 {
            let (f, fetched) = super::widest_passthrough_format(9, c).expect("F32 has every width");
            assert_eq!(fetched, c, "{f:?} fetches more than the guest binds");
        }
        // An integer format has no passthrough at ANY width, so widening cannot smuggle
        // one in.
        for c in 1..=4u8 {
            assert_eq!(super::widest_passthrough_format(0, c), None);
        }
    }

    /// >>> THE INTEGER FETCH DELIVERS EXACTLY WHAT [`super::read_attr_component`] READS.
    ///
    /// The same standard the float table is held to, and for the same reason: a wrong entry
    /// here is a smooth, plausible, entirely wrong set of numbers - a bone index off by a
    /// factor, a packed byte read as a normalised fraction - not a broken frame. Written out
    /// from the WebGPU spec rather than from the function under test, so the two readings
    /// have to agree rather than being one reading twice.
    #[test]
    fn the_integer_fetch_delivers_what_the_cpu_reader_reads() {
        use wgpu::VertexFormat as V;
        // What the WebGPU vertex stage delivers, per the spec: the integer value itself, in
        // `u32` for the unsigned formats and `i32` for the signed ones. The module then
        // converts with `f32()` - see `VertexAttribute::int_fetch`.
        fn webgpu_int_fetch(format: V, bytes: &[u8], c: usize) -> f32 {
            let u16at = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
            match format {
                V::Uint8x2 | V::Uint8x4 => bytes[c] as u32 as f32,
                V::Sint8x2 | V::Sint8x4 => bytes[c] as i8 as i32 as f32,
                V::Uint16x2 | V::Uint16x4 => u16at(c * 2) as u32 as f32,
                V::Sint16x2 | V::Sint16x4 => u16at(c * 2) as i16 as i32 as f32,
                other => panic!("{other:?} is not an integer vertex format"),
            }
        }
        for gxm in [0u8, 1, 2, 3] {
            for comps in 1..=4u8 {
                let (f, fetched) =
                    super::integer_vertex_format(gxm, comps).expect("every plain integer maps");
                assert!(fetched >= comps, "{f:?} fetches {fetched} for {comps} bound");
                assert!(fetched <= comps.max(2) + 1, "{f:?} widens by more than one step");
                // Every byte pattern of the narrow formats, and enough of the wide ones to
                // cover the sign boundary and both ends.
                let size = super::attr_component_size(gxm);
                let patterns: Vec<Vec<u8>> = if size == 1 {
                    (0..=255u8).map(|b| vec![b; 8]).collect()
                } else {
                    [0u16, 1, 0x7fff, 0x8000, 0xffff, 0x1234]
                        .iter()
                        .map(|v| v.to_le_bytes().repeat(4))
                        .collect()
                };
                for bytes in patterns {
                    for c in 0..comps as usize {
                        assert_eq!(
                            webgpu_int_fetch(f, &bytes, c),
                            super::read_attr_component(&bytes, 0, gxm, c),
                            "GXM {gxm} x{comps} lane {c} of {bytes:02x?} through {f:?}"
                        );
                    }
                }
            }
        }
        // And nothing else reaches it: the normalised and float families have exact float
        // fetches and must keep them.
        for gxm in [4u8, 5, 6, 7, 8, 9, 10, 200] {
            for comps in 1..=4u8 {
                assert_eq!(super::integer_vertex_format(gxm, comps), None, "GXM {gxm}");
            }
        }
    }

    /// An attribute reaching PAST the row copies what there is instead of panicking.
    ///
    /// # Which case this actually is, because the obvious one is not reachable
    /// Plain truncation cannot do it: `repack_vertices_into` derives its vertex count from
    /// the buffer length, so `vbase + guest_offset` stays inside the buffer for every vertex
    /// it walks. What DOES reach past the end is an attribute whose OFFSET lies beyond the
    /// stream's own stride - and that is not hypothetical, it is the exact shape of the
    /// stride-0 defect [[vitaslop-a-stride-zero-stream-needs-a-column]], where six
    /// per-instance attributes at offsets 0..64 were rebased into a 28-byte row.
    ///
    /// A frame sweep cannot see this either way: every title in the corpus submits streams
    /// that hold what they declare, so both plans agree on every pixel while one of them
    /// carries an index where the other reads through `slice::get`. The clamp is on the
    /// range START as well as the length, because `vertices[src..src]` panics on the start
    /// even when the length is zero.
    #[test]
    fn an_attribute_reaching_past_the_row_copies_what_there_is() {
        // Two attributes: an F32x2 at 0 and a U8N x4 at 8, so a 12-byte row.
        let repack = [
            super::RepackAttr {
                guest_offset: 0,
                gxm_format: 9,
                components: 2,
                slots: 2,
                fill: [0.0; 4],
                packed_offset: 0,
                copy_bytes: 8,
            },
            super::RepackAttr {
                guest_offset: 8,
                gxm_format: 4,
                components: 4,
                slots: 4,
                fill: [0.0; 4],
                packed_offset: 8,
                copy_bytes: 4,
            },
        ];
        let full: Vec<u8> = (0..24u8).collect();
        // First the ordinary case, at every truncation: the packed row IS the guest row
        // here, so each byte present must land at the same index.
        for len in 0..=full.len() {
            let mut out = Vec::new();
            super::repack_vertices_into(&full[..len], 12, &repack, 12, &mut out);
            assert!(out.len().is_multiple_of(12), "len {len} produced {} bytes", out.len());
            for v in 0..out.len() / 12 {
                for (i, b) in out[v * 12..v * 12 + 12].iter().enumerate() {
                    let src = v * 12 + i;
                    let want = if src < len { full[src] } else { 0 };
                    assert_eq!(*b, want, "len {len}, vertex {v}, byte {i}");
                }
            }
        }
        // Now the case that actually reaches past the buffer: an attribute at offset 64 on
        // a 12-byte row. Straddling and wholly-past both have to survive it - and the
        // straddling one must copy its first bytes rather than give up on the whole slot.
        for guest_offset in [8u32, 10, 12, 64, 4096] {
            let plan = [super::RepackAttr {
                guest_offset,
                gxm_format: 9,
                components: 2,
                slots: 2,
                fill: [0.0; 4],
                packed_offset: 0,
                copy_bytes: 8,
            }];
            let mut out = Vec::new();
            super::repack_vertices_into(&full, 12, &plan, 8, &mut out);
            assert_eq!(out.len(), 16, "two vertices of an 8-byte packed row");
            for v in 0..2usize {
                for i in 0..8usize {
                    let src = v * 12 + guest_offset as usize + i;
                    let want = full.get(src).copied().unwrap_or(0);
                    assert_eq!(out[v * 8 + i], want, "offset {guest_offset}, vertex {v}, byte {i}");
                }
            }
        }
    }

    /// The offset rule this path checks is WebGPU's own, and it is the FORMAT's size that is
    /// capped at four rather than the component's. Getting that wrong builds a pipeline the
    /// driver refuses at creation, which drops the pair for the rest of the run.
    #[test]
    fn a_mapped_formats_alignment_is_its_own_size_capped_at_four() {
        use wgpu::VertexFormat as V;
        for (f, want) in [
            (V::Unorm8x2, 2u64),
            (V::Unorm8x4, 4),
            (V::Snorm8x2, 2),
            (V::Float16x2, 4),
            (V::Float16x4, 4),
            (V::Float32, 4),
            (V::Float32x3, 4),
        ] {
            assert_eq!(f.size().min(4), want, "{f:?}");
        }
    }
}

/// The narrowest WebGPU vertex format that fetches AT LEAST `components` of a GXM attribute
/// of this format, with the number of components it actually fetches.
///
/// # Why a WIDER format is admissible at all
/// WebGPU has no `float16x3`, `unorm8x3` or `snorm16x3`, and that gap is not a detail: it is
/// **72 of one sports title's 109 repacking pipelines**, all of them `F16 x3`. The four-wide
/// format fetches those three components correctly and puts the two bytes that FOLLOW them
/// in the guest's row into the fourth lane - bytes the guest never bound to this attribute,
/// which usually belong to the next one.
///
/// That is admissible when, and only when, **nothing reads the fourth lane**
/// ([`vitaslop_gxp_shader::attrflow::Fill::Unobserved`]) - a stronger test than the fill
/// agreement the narrow case needs, and a different one: there the question is whether OUR
/// constant matches the hardware's, here it is whether ANY value can be told apart. The
/// caller checks it, and also that the wider read stays inside the row.
/// >>> WHICH VERTEX PLAN EVERY PIPELINE GOT, AND THE FIRST REASON IT WAS NOT THE BEST ONE.
///
/// # A REPORT THAT EXISTS IS NOT A DIAGNOSTIC A READER CAN COUNT
/// `report_vertex_plan_ok` and `report_vertex_repack` already say, per pair, which of the
/// three plans it got. They cannot be COUNTED: both go through panels that keep 96 distinct
/// lines and drop the rest (one baseball-title run dropped 379 from STATUS and 293 from the warnings
/// panel), so grepping a log for them answers a question about the panel's capacity.
/// MEASURED against that: two changes that each took pipelines off the repack moved the
/// per-frame `packed_passthrough` counter by 95 -> 140 draws of 605 while the log said every
/// pipeline was passthrough, and there was no way to tell which of those two readings was
/// wrong [[vitaslop-a-diagnostic-can-bury-the-findings]].
///
/// Four counters and two strings instead. Cumulative, because a pipeline is BUILT once and a
/// windowed count of build-time events reads zero in every steady frame.
pub(super) struct VertexPlanCensus {
    /// The guest's own row is bound as it stands - no copy at all.
    passthrough: std::sync::atomic::AtomicU64,
    /// A packed row every attribute of which keeps the guest's component format: a byte copy
    /// per attribute, no widening.
    native_copy: std::sync::atomic::AtomicU64,
    /// A packed row with at least one attribute widened to f32 - the ~7x.
    expanded: std::sync::atomic::AtomicU64,
    /// A pipeline whose program declares NO vertex attributes at all. It takes none of the
    /// three plans and is counted so the three above add up to the pipelines built.
    no_attributes: std::sync::atomic::AtomicU64,
    /// The first reason recorded for each of the two outcomes that are not passthrough.
    why_copy: std::sync::Mutex<Option<String>>,
    why_expand: std::sync::Mutex<Option<String>>,
}

pub(super) static VERTEX_PLANS: VertexPlanCensus = VertexPlanCensus {
    passthrough: std::sync::atomic::AtomicU64::new(0),
    native_copy: std::sync::atomic::AtomicU64::new(0),
    expanded: std::sync::atomic::AtomicU64::new(0),
    no_attributes: std::sync::atomic::AtomicU64::new(0),
    why_copy: std::sync::Mutex::new(None),
    why_expand: std::sync::Mutex::new(None),
};

/// One line for the panel: what plan the pipelines built so far got, and why the ones that
/// are not on the best plan are not. See [`VertexPlanCensus`].
pub fn vertex_plan_census_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let take = |m: &std::sync::Mutex<Option<String>>| {
        m.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_else(|| "-".into())
    };
    format!(
        "vertex plans: {} pipeline(s) BIND THE GUEST'S OWN ROW (no copy), {} copy it into a              packed row in the guest's own formats (first reason: {}), {} WIDEN an attribute to              f32 (first reason: {}), {} declare no vertex attributes. Cumulative, because a              pipeline is built once: a steady frame builds none. The DRAW counts for the same              three are `prepare split`'s `bound straight through` against its packed hits and              misses - a plan a hundred pipelines take and two draws use is not the frame's cost.",
        VERTEX_PLANS.passthrough.load(Relaxed),
        VERTEX_PLANS.native_copy.load(Relaxed),
        take(&VERTEX_PLANS.why_copy),
        VERTEX_PLANS.expanded.load(Relaxed),
        take(&VERTEX_PLANS.why_expand),
        VERTEX_PLANS.no_attributes.load(Relaxed),
    )
}

/// Record one pipeline's vertex plan. See [`VertexPlanCensus`].
pub(super) fn note_vertex_plan(outcome: &str, why: Option<&str>) {
    use std::sync::atomic::Ordering::Relaxed;
    let (slot, reason) = match outcome {
        "passthrough" => (&VERTEX_PLANS.passthrough, None),
        "copy" => (&VERTEX_PLANS.native_copy, Some(&VERTEX_PLANS.why_copy)),
        "expand" => (&VERTEX_PLANS.expanded, Some(&VERTEX_PLANS.why_expand)),
        _ => (&VERTEX_PLANS.no_attributes, None),
    };
    slot.fetch_add(1, Relaxed);
    if let (Some(cell), Some(why)) = (reason, why) {
        let mut held = cell.lock().unwrap_or_else(|e| e.into_inner());
        if held.is_none() {
            *held = Some(why.to_string());
        }
    }
}


/// What the guest bound to each vertex attribute, as the link wants it: `(base lane, GXM
/// format, components)`, sorted by base lane.
///
/// Sorted because the list is folded into the module cache key: the same layout declared in
/// a different order must produce the same key, or one program pair compiles twice.
pub(super) fn gxp_guest_attr_spec(gxp: &GxpRecompile) -> Vec<(u32, u8, u8)> {
    // >>> THE TWO ARMS THAT MEAN "CONVERT ON THE CPU" TURN THIS OFF TOO, or they would stop
    // >>> being the arms they are named for.
    //
    // `gxp_vertex_passthrough()` off is the whole hardware-fetch path's negative control,
    // and `VITASLOP_GXP_ATTR_FILL` overrides a surplus lane's value ON THE CPU - which is
    // only possible in a row this code writes as floats. An integer fetch under either would
    // leave the OFF arm quietly doing half of the ON arm's work.
    if !crate::knobs::gxp_guest_attrs_in_link() {
        return Vec::new();
    }
    // EVERY attribute, not only the plain-integer ones: the COMPONENT COUNT is what lets
    // the module bake a surplus lane's constant (`VertexAttribute::guest_components`), and
    // that is the blocker that holds most of a frame's draws on the repack. What keeps the
    // ahead-of-draw precompile (`precompile_pairs`, keyed on the two blobs alone) hitting is
    // not this list being empty - it is that `gxp_attr_interface_fold` reads what the LINK
    // did with it, and answers zero when it did nothing.
    // The ONE rule, shared with the patcher-time prelink - see `vitaslop_gxp_shader::guest_attr_spec`.
    vitaslop_gxp_shader::guest_attr_spec(gxp.attributes.iter().map(|a| (a.reg_index as u32, a.gxm_format, a.components)))
}

/// >>> HOW THE GUEST'S LAYOUT CHANGED THE MODULE - folded into its cache key.
///
/// Read off what the LINK DID, not off what it was told. Two things vary a module now: an
/// attribute declared integer-typed (`int_fetch`) and one whose surplus lanes are baked
/// constants (`guest_components`, and only where it actually bakes a lane). Everything else
/// about a layout - offsets, stride, a format the fetch converts - leaves the WGSL identical.
///
/// **ZERO when nothing varied**, which is the whole reason it is computed here rather than
/// from the spec: `precompile_pairs` builds its modules ahead of any draw under the two
/// blobs alone, and a key that moved for every pair would silently stop hitting them - the
/// shader compiles would come straight back into the frame they exist to leave.
pub(super) fn gxp_attr_interface_fold(attrs: &[vitaslop_gxp_shader::module::VertexAttribute]) -> u64 {
    let mut h: u64 = 0;
    for a in attrs {
        let bakes = a.guest_components.is_some_and(|b| b < a.components);
        if a.int_fetch.is_none() && !bakes {
            continue;
        }
        if h == 0 {
            h = 0xa5a5_0000_0000_0003;
        }
        let tag = match a.int_fetch {
            Some(true) => 2u64,
            Some(false) => 1,
            None => 0,
        };
        for v in [a.location as u64, tag, a.guest_components.unwrap_or(0) as u64] {
            h ^= v;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The WebGPU vertex format for an attribute the LINK declared INTEGER-fetched
/// ([`vitaslop_gxp_shader::module::VertexAttribute::int_fetch`]), with the number of
/// components it fetches.
///
/// # SEPARATE FROM [`passthrough_vertex_format`], AND THAT IS THE POINT
/// Every format in that table converts to `f32` in hardware, and the module declares every
/// attribute `vec4<f32>`; these deliver `u32`/`i32` and are legal ONLY against a module that
/// declares the matching integer type. Pooling them into one table would make it possible to
/// pick an integer format for an attribute the link left as float, and WebGPU refuses that
/// pipeline outright - so the two questions get two functions and the caller has to say
/// which one it is asking.
///
/// WebGPU has no one- or three-component 8/16-bit format, so a 1- or 3-component attribute
/// is fetched 2 or 4 wide. That is admissible under exactly the proof the link already made
/// before setting `int_fetch`: every lane above the guest's binding is unobserved.
pub(super) fn integer_vertex_format(gxm_format: u8, components: u8) -> Option<(wgpu::VertexFormat, u8)> {
    use wgpu::VertexFormat as V;
    let wide = components > 2;
    let f = match (gxm_format, wide) {
        (0, false) => V::Uint8x2,
        (0, true) => V::Uint8x4,
        (1, false) => V::Sint8x2,
        (1, true) => V::Sint8x4,
        (2, false) => V::Uint16x2,
        (2, true) => V::Uint16x4,
        (3, false) => V::Sint16x2,
        (3, true) => V::Sint16x4,
        _ => return None,
    };
    Some((f, if wide { 4 } else { 2 }))
}

pub(super) fn widest_passthrough_format(gxm_format: u8, components: u8) -> Option<(wgpu::VertexFormat, u8)> {
    // F32 has every width, so it never needs one.
    if let Some(f) = passthrough_vertex_format(gxm_format, components) {
        return Some((f, components));
    }
    let wider = components + 1;
    (wider <= 4).then(|| passthrough_vertex_format(gxm_format, wider).map(|f| (f, wider)))?
}

/// How many bytes of CAPACITY one pooled per-pass staging arena may keep between passes.
///
/// >>> THIS NUMBER IS THE WHOLE DIFFERENCE BETWEEN THIS AND THE VERSION THAT WAS REVERTED.
/// The saving is not holding a big buffer, it is not growing one from zero by doubling
/// several hundred times a pass; a couple of megabytes buys all of that and a loading
/// frame's peak is handed straight back. MEASURED on a baseball title's stadium: the
/// recompiled path's vertex arena carries ~100-300 KB per pass and the whole frame ~1.3 MB
/// across 12 passes, so 2 MB covers every pass of a gameplay frame with nothing to re-grow.
/// Six arenas, so the worst case this pins is 12 MB, against the tens of megabytes the
/// unbounded version left resident for ever.
pub(super) const ARENA_RETAIN_BYTES: usize = 2 << 20;

/// `VITASLOP_GXM_ARENA_POOL=1` pools the six per-pass staging arenas. **OFF BY DEFAULT, and
/// that is a measurement, not caution.**
///
/// MEASURED both ways on one build. Desktop: arena copy 0.33 -> 0.13 ms and `prepare` 2.59
/// -> 2.51 ms on a steady gameplay frame, four shots bit-identical. BROWSER, which is the
/// product path: **nothing** - 99% speed with it off against 97-98% with it on, i.e. inside
/// the rig's own spread. So the desktop win does not survive to the engine that has to have
/// it, and the version of this that WAS defaulted on cost the target DEVICE real frame rate
/// (see `GxmRenderer::pass_arenas` and the revert note above it). A change with no measured
/// benefit on the product path and a device regression in its history does not get the
/// default; the knob is here so a device run can settle it.
pub(super) fn arena_pool_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::knobs::var("VITASLOP_GXM_ARENA_POOL").as_deref() == Ok("1"))
}

/// The largest `array_stride` WebGPU guarantees (`maxVertexBufferArrayStride`). A guest row
/// wider than this cannot be bound as-is, so it repacks - and the repacked row, which drops
/// every byte no attribute names, is usually the narrower of the two anyway.
pub(super) const MAX_VERTEX_ARRAY_STRIDE: u32 = 2048;

/// A content hash over a byte slice, consuming EIGHT bytes per multiply instead of one.
///
/// The word-at-a-time shape matters because this runs over megabytes a frame: every draw
/// hashes its whole vertex stream to find its repacked geometry, and a byte-at-a-time fold
/// over that is milliseconds of a twenty-millisecond frame. The trailing bytes are folded in
/// individually so the result still depends on the exact length.
///
/// # Why this is not plain word-wise FNV-1a any more
/// It was, and the word-wise form of FNV has a failure mode the byte-wise form does not:
/// **a difference confined to bit 63 of a word survives the multiply unchanged, so two such
/// differences cancel.** Multiplication by an odd constant is linear mod 2^64, and 2^63 is a
/// fixed point of it (`2^63 * odd = 2^63`), so a top-bit flip in word `k` leaves the
/// accumulator differing in exactly bit 63 - which the very next word's top-bit flip then
/// XORs back to zero. **Any EVEN number of top-bit flips at 8-byte-aligned positions hashes
/// identically.** No other bit position does this: a flip at bit `b < 63` becomes `2^b *
/// PRIME`, which is spread across the word and cannot be cancelled by a single clean bit.
///
/// That is not a theoretical weakness, it is a defect that shipped and was visible. This
/// hash keys the PACKED-VERTEX CACHE, whose entries are repacked vertex buffers. A UI text
/// quad faded through a per-vertex unorm8 COLOUR, 150 vertices of a 24-byte stride with the
/// colour at byte 20, puts every vertex's ALPHA byte at absolute offset `24v + 23` - that is
/// `7 mod 8`, the top byte of a word, so the alpha's high bit IS bit 63. 150 is even.
/// So two frames of the fade whose alpha bytes differed only in bit 7 - 127 against 255, 101
/// against 229, 50 against 178, 25 against 153, 76 against 204 - collided, and the second
/// frame drew the FIRST frame's repacked vertices. MEASURED on the title screen: of the 15
/// alpha levels the guest submits over its 60-frame cycle, the 5 whose bit-7 partner also
/// appears in the cycle rendered as that partner instead, every time. The fade looked like it
/// was stepping backwards, and the guest was emitting a perfectly monotonic ramp throughout.
///
/// The rotate breaks the fixed point: a difference at bit 63 lands at bit 30 before the next
/// multiply, where it is spread rather than preserved. The final avalanche is murmur3's
/// `fmix64`, so the length also reaches every output bit instead of only the low ones.
pub(super) fn fnv64(seed: u64, bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    // FOUR independent lanes over 32-byte blocks, folded at the end. The single-lane
    // form's `mul + rotate` is a serial dependency chain - one multiply LATENCY per 8
    // bytes - and this hash is the top item of the browser's `prepare` split (MEASURED:
    // 1.34 ms over 7.26 MB a frame on a retail sports title whose rotating vertex arenas
    // defeat the allocation cache). The lanes give the CPU four chains to overlap.
    // Collisions stay harmless exactly as before: every consumer verifies a hit with a
    // full byte compare, so hash quality costs lookups, never correctness - but the
    // per-lane mix and the cross-lane fold below keep the bit-63 story from the comment
    // above true for every lane position.
    let mut h = seed;
    let (blocks, rest) = bytes.split_at(bytes.len() & !31);
    if !blocks.is_empty() {
        let mut lanes = [
            seed,
            seed ^ 0x9e37_79b9_7f4a_7c15,
            seed.rotate_left(17) ^ PRIME,
            !seed,
        ];
        for b in blocks.chunks_exact(32) {
            for (i, lane) in lanes.iter_mut().enumerate() {
                let w = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
                *lane = (*lane ^ w).wrapping_mul(PRIME).rotate_left(31);
            }
        }
        for lane in lanes {
            h = (h ^ lane).wrapping_mul(PRIME).rotate_left(29);
        }
    }
    let (words, tail) = rest.split_at(rest.len() & !7);
    for w in words.chunks_exact(8) {
        h ^= u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
        h = h.wrapping_mul(PRIME).rotate_left(31);
    }
    for &b in tail {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h ^= bytes.len() as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

/// Whether this FRAGMENT blob blends for itself by reading the destination colour out of its
/// output bank - memoised on the blob's content hash, because the answer needs decoding the
/// whole program and the question is asked once per draw per frame.
///
/// Asked BEFORE a pass is encoded (see `encode_pass`), because the answer decides how the
/// pass is CUT: a draw that reads the destination needs the attachment copied immediately
/// before it, and a render pass cannot be split once it has begun.
/// The dual-source plan of a fragment blob (`vitaslop_gxp_shader::fragment_dual_source_plan`),
/// memoised the same way as [`fragment_reads_dest`] and for the same reason.
pub(super) fn fragment_dual_plan(fprog: &std::sync::Arc<[u8]>) -> Option<std::sync::Arc<Vec<(u32, u32, u8)>>> {
    use std::sync::{Arc, Mutex, OnceLock};
    static MEMO: OnceLock<Mutex<HashMap<(usize, usize), Option<Arc<Vec<(u32, u32, u8)>>>>>> = OnceLock::new();
    let memo = MEMO.get_or_init(|| Mutex::new(HashMap::default()));
    let key = (Arc::as_ptr(fprog) as *const u8 as usize, fprog.len());
    let mut g = memo.lock().unwrap_or_else(|e| e.into_inner());
    g.entry(key)
        .or_insert_with(|| match vitaslop_gxp_shader::fragment_dual_source_plan(fprog) {
            Ok(p) => Some(Arc::new(p)),
            Err(why) => {
                // Only for a blob that READS the destination: every other fragment blob
                // is refused for the same uninteresting reason.
                if vitaslop_gxp_shader::fragment_reads_dest_color(fprog) {
                    report_status!(
                        "gxp dual-source: fragment blob {:#x} keeps the pass split - {why}",
                        key.0
                    );
                }
                None
            }
        })
        .clone()
}

/// Whether THIS draw takes the DUAL-SOURCE pipeline for its destination read: the blob has
/// a plan and every gate the plan names is zero in the draw's own memory window. Asked
/// per draw, because the gates are per-draw data - see `module::dual_source_plan`.
pub(super) fn draw_dual_source(gxp: &GxpRecompile) -> bool {
    let Some(plan) = fragment_dual_plan(&gxp.fprog) else { return false };
    // Once per blob: the plan and this draw's gate bytes, so a lowering that never
    // fires can be told from one whose gates are never zero.
    {
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
        let id = std::sync::Arc::as_ptr(&gxp.fprog) as *const u8 as usize;
        if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(id) {
            let bytes: Vec<String> = plan
                .iter()
                .map(|&(w, byte, len)| format!("{w}+{byte}:{:02x?}", dual_gate_bytes(gxp, w, byte, len)))
                .collect();
            report_status!(
                "gxp dual-source plan for fragment blob {id:#x}: {} gate(s) {plan:?}; this draw's gate bytes {bytes:?}; windows {}",
                plan.len(),
                gxp.frag_mem_windows.iter().map(|(a, b)| format!("{a:#x}/{}B", b.len())).collect::<Vec<_>>().join(",")
            );
        }
    }
    plan.iter().all(|&(w, byte, len)| dual_gate_bytes(gxp, w, byte, len).is_some_and(|b| b.iter().all(|x| *x == 0)))
}

/// Whether THIS DRAW's fragment program writes the DESTINATION COLOUR BACK UNCHANGED - a
/// draw whose colour work cannot change a single pixel, proven from the draw's own uniform
/// bytes by [`vitaslop_gxp_shader::fold::fragment_colour_is_destination`].
///
/// # Why a per-draw question and not a per-shader one
/// A title whose blending is ALU carries ONE generic-blend pair and configures it out of a
/// uniform buffer, so the SAME program is a source-over in one draw and a no-op in the next.
/// MEASURED on a baseball title's world pass (`6357447cb96d21b1`, `swing-and-play` f11636):
/// eighteen submissions in one frame, **3,064,899 of the pass's 3,963,212 samples - 77% of
/// all the fill in the frame's biggest pass** - with every source coefficient in its window
/// zero and its destination coefficient one. The phone is GPU-bound at 84% with a 12.45 ms
/// world pass, so this is the largest single lever the renderer has, and nothing in the
/// SHADER says it is there.
///
/// # Memoised in two levels, because the fold is not free
/// Level one is per BLOB and asks the question that does not depend on a draw at all: can
/// this program EVER be an identity (does it read the destination, and is it linear in it)?
/// That is false for all but a handful of programs in any title, and it short-circuits
/// before any per-draw work. Level two is keyed on the draw's uniform BYTES - the only
/// thing that can change the answer for a program that passes level one - so a pair
/// submitted eighteen times with the same coefficients folds once.
/// LEVEL ONE of that question, on its own: can this program EVER be an identity, whatever a
/// draw's bytes say? Memoised on the blob's shared allocation exactly as
/// [`fragment_reads_dest`] is and for the same measured reason - the answer needs a full
/// RECOMPILE of the fragment program, and this is asked once per draw per frame.
///
/// >>> AND IT NOW HOLDS THE PREPARED PROGRAM, NOT A BOOLEAN, which is what took the
/// >>> RECOMPILE off the per-draw path.
///
/// The level-two memo below misses on a genuine fraction of draws - MEASURED at 19 of 141
/// asks a frame on a baseball title's world pass - and every one of those misses used to
/// re-run `recompile_fragment`: a full USSE decode AND a full WGSL emission, for a question
/// about a few dozen bytes of blend coefficients. Every value that work produces is a pure
/// function of the BLOB, so it belongs here, keyed on the blob's shared allocation exactly
/// as [`fragment_reads_dest`] is. A miss now costs the fold's own evaluation and nothing
/// else.
pub(super) fn fragment_fold_program(
    fprog: &std::sync::Arc<[u8]>,
) -> Option<std::sync::Arc<vitaslop_gxp_shader::fold::FoldProgram>> {
    use std::sync::{Arc, Mutex, OnceLock};
    type Prepared = Option<Arc<vitaslop_gxp_shader::fold::FoldProgram>>;
    static MEMO: OnceLock<Mutex<HashMap<(usize, usize), Prepared>>> = OnceLock::new();
    let cell = MEMO.get_or_init(|| Mutex::new(HashMap::default()));
    let key = (Arc::as_ptr(fprog) as *const u8 as usize, fprog.len());
    let mut g = cell.lock().unwrap_or_else(|e| e.into_inner());
    g.entry(key)
        .or_insert_with(|| vitaslop_gxp_shader::fold::FoldProgram::prepare(fprog).map(Arc::new))
        .clone()
}

/// >>> THIS IS THE LARGEST SINGLE CPU ITEM IN A FRAME ON THIS TITLE, AND TWO ATTEMPTS TO
/// >>> MAKE IT CHEAPER BOTH FAILED. READ THIS BEFORE TRYING A THIRD.
///
/// MEASURED on the desktop browser, with the sub-split added for the purpose:
/// `key 5.40 ms, of which pair-key 0.06 + colour-fold 5.16`, and in other windows up to
/// `colour-fold 11.03`. `key` is 62% of `prepare` and `prepare` is the biggest CPU phase in
/// a device frame, so this function alone is roughly a fifth of it. Over those same runs it
/// elided **0.0 draws**. It is also implicated in the WASH-OUT, which is a latch on
/// writeback latency: of four runs of one configuration, the two that washed were the two
/// whose `colour-fold` read 9.46 and 10.39 ms against 5.41 and 5.16 for the two that did not.
///
/// **What was tried and REFUTED, both reverted:**
///   * A per-frame memo keyed on the uniform buffers' POINTERS in front of the byte-hash
///     memo below, on the reasoning that `frag_sa` and the windows are `Arc<[u8]>` so
///     pointer identity implies content identity. Sound, and it never hit: this title hands
///     every draw a FRESH snapshot allocation, so the pointers differ every time and the
///     walk was pure added cost. `colour-fold` did not move (9.46 / 5.41 / 5.16 / 10.39
///     before, 9.76 / 5.64 / 5.39 / 10.67 after).
///   * A patience gate: after 64 asks with no win, only re-ask every 64th draw. It does not
///     fire on the program that costs the time, because that program DOES fold sometimes -
///     one win puts it back on every draw, which is the whole point of the gate. Added a
///     mutex-guarded lookup per draw for nothing.
///
/// >>> **THAT MEASUREMENT WAS TAKEN, AND IT REFUTED THE SUSPECT.** The instruction here was
/// to measure how many bytes the hash covers per draw before trying a third fix, with the
/// memory windows named as the likely culprit. The census
/// ([`EncodeWork::fold_asks`]) says the hash reads **128-144 BYTES AN ASK - 0.02 MB a
/// frame**, so it was never the cost and all three guesses above were aimed at the same
/// wrong half.
///
/// The cost is the MISS: about 8 of 19 asks a frame at gameplay, and a miss used to run
/// `recompile_fragment` - a full USSE decode AND a full WGSL emission - for a question
/// about a few dozen bytes of blend coefficients. Everything that work produces is a pure
/// function of the BLOB, so it now lives in [`fragment_fold_program`] and a miss costs the
/// fold's own evaluation alone: **108.0 us -> 1.7 us, 64.9x**, measured over the corpus's
/// eligible programs by
/// `what_a_colour_fold_memo_miss_costs_with_and_without_the_prepared_program`, which also
/// asserts that both paths return the identical answer.
///
/// The three failures above were only visible because the sub-split existed - before it,
/// `key` was a phase name that implied a hash - and this one only because a census that
/// needs no clock could be left ON while the thing it measures runs.
pub(super) fn draw_colour_is_noop(gxp: &GxpRecompile) -> bool {
    use std::sync::Arc;
    if !crate::gpu::elide_noop_draws() {
        return false;
    }
    let Some(fp) = fragment_fold_program(&gxp.fprog) else {
        enc(&ENC.fold_l1_rejects, 1);
        return false;
    };
    let blob = (Arc::as_ptr(&gxp.fprog) as *const u8 as usize, gxp.fprog.len());
    // LEVEL TWO: the draw's own bytes. Hashed rather than compared because the key has to
    // be `Copy` and small; a collision can only misreport ONE draw's blend as another's,
    // and both sides of that are draws of the SAME program whose windows are a few dozen
    // bytes - so the hash covers the base addresses too, and the fold is re-run for any
    // byte pattern not seen before.
    draw_colour_is_noop_by_bytes(gxp, &fp, blob)
}

pub(super) fn draw_colour_is_noop_by_bytes(
    gxp: &GxpRecompile,
    fp: &vitaslop_gxp_shader::fold::FoldProgram,
    blob: (usize, usize),
) -> bool {
    use std::sync::{Mutex, OnceLock};
    // The census (see `EncodeWork::fold_asks`): what this hash READS, counted before it
    // reads it, so a memo that hits perfectly and still costs milliseconds is visible as
    // bytes rather than having to be inferred from a time.
    enc(&ENC.fold_asks, 1);
    enc(&ENC.fold_hash_bytes, gxp.frag_sa.len() as u64);
    enc(&ENC.fold_hash_windows, gxp.frag_mem_windows.len() as u64);
    let mut h = fnv64(0xcbf2_9ce4_8422_2325, &gxp.frag_sa);
    for (base, bytes) in &gxp.frag_mem_windows {
        enc(&ENC.fold_hash_bytes, bytes.len() as u64);
        h = fnv64(h ^ u64::from(*base), bytes);
    }
    static MEMO: OnceLock<Mutex<HashMap<(usize, u64), bool>>> = OnceLock::new();
    let cell = MEMO.get_or_init(|| Mutex::new(HashMap::default()));
    let mut g = cell.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(v) = g.get(&(blob.0, h)) {
        enc(&ENC.fold_memo_hits, 1);
        return *v;
    }
    enc(&ENC.fold_memo_misses, 1);
    let windows: Vec<(u32, &[u8])> = gxp.frag_mem_windows.iter().map(|(a, b)| (*a, &b[..])).collect();
    let terms = vitaslop_gxp_shader::fold::fragment_colour_terms_prepared(
        fp,
        &vitaslop_gxp_shader::fold::DrawUniforms { frag_sa: &gxp.frag_sa, windows: &windows },
        &mut None,
    );
    // G == 0 and F == 1 in every channel, or this draw paints something - the same verdict
    // `fragment_colour_is_destination` reads off the terms, taken here because the prepared
    // path returns the terms themselves.
    let v = terms.is_some_and(|(g, f)| (0..4).all(|c| g[c] == Some(0.0) && f[c] == Some(1.0)));
    // Bounded like every other cache here: the key is derived from the entry's own
    // contents, so clearing wholesale costs a re-fold and never correctness.
    if g.len() >= PAIR_KEY_CACHE_CAP {
        g.clear();
    }
    g.insert((blob.0, h), v);
    v
}

/// The bytes of one dual-source gate in THIS draw's data - a memory window, or the
/// fragment default uniform buffer. `None` when the draw carries no such bytes, which
/// the gate treats as NOT zero.
pub(super) fn dual_gate_bytes(gxp: &GxpRecompile, source: u32, byte: u32, len: u8) -> Option<&[u8]> {
    let range = byte as usize..(byte + len as u32) as usize;
    if source == vitaslop_gxp_shader::DUAL_GATE_UNIFORM {
        gxp.frag_sa.get(range)
    } else {
        gxp.frag_mem_windows.get(source as usize).and_then(|(_, b)| b.get(range))
    }
}

pub(super) fn fragment_reads_dest(fprog: &std::sync::Arc<[u8]>) -> bool {
    use std::sync::{Mutex, OnceLock};
    static MEMO: OnceLock<Mutex<HashMap<(usize, usize), bool>>> = OnceLock::new();
    let memo = MEMO.get_or_init(|| Mutex::new(HashMap::default()));
    // Keyed on the SHARED ALLOCATION, not on a content hash. The capture reads each
    // container out of guest memory once and every draw bound to it carries the same `Arc`
    // ([`GxpRecompile::fprog`]), so the pointer is a stable identity - and hashing a few
    // hundred bytes per draw per frame is not. That is not a micro-optimisation: this runs
    // in the pre-scan, before any pass is encoded, and the work it does lands in the guest's
    // own clock. Hashing here moved two unrelated titles' TIMELINES far enough that an A/B
    // of the arm this scan gates came back "9 of 10 frames differ" on a change that touches
    // neither title. The length joins the key because a freed allocation can be reused.
    let key = (std::sync::Arc::as_ptr(fprog) as *const u8 as usize, fprog.len());
    let mut g = memo.lock().unwrap_or_else(|e| e.into_inner());
    *g.entry(key).or_insert_with(|| vitaslop_gxp_shader::fragment_reads_dest_color(fprog))
}

/// Whether a fragment blob blends against the destination at REGISTER (half/float)
/// precision - see `vitaslop_gxp_shader::fragment_blends_at_register_precision` and
/// `GxmRenderer::tile_shadow`. Memoised on the shared allocation for the reason
/// [`fragment_reads_dest`] gives: it runs in the per-pass pre-scan.
pub(super) fn fragment_blends_wide(fprog: &std::sync::Arc<[u8]>) -> bool {
    use std::sync::{Mutex, OnceLock};
    static MEMO: OnceLock<Mutex<HashMap<(usize, usize), bool>>> = OnceLock::new();
    let memo = MEMO.get_or_init(|| Mutex::new(HashMap::default()));
    let key = (std::sync::Arc::as_ptr(fprog) as *const u8 as usize, fprog.len());
    let mut g = memo.lock().unwrap_or_else(|e| e.into_inner());
    *g.entry(key).or_insert_with(|| vitaslop_gxp_shader::fragment_blends_at_register_precision(fprog))
}
