//! The repack: a guest vertex stream into the packed row a pipeline binds, and what a lane
//! above the guest's binding is fed.

use super::*;

/// Repack a guest vertex stream into the tightly-packed `Float32xN` layout the recompiled
/// pipeline expects. One packed vertex per guest vertex, in order, so the index buffer is
/// unchanged.
///
/// Appends into a caller-owned arena rather than returning a `Vec`: the arena is uploaded
/// once per pass, so a per-draw allocation here would put back exactly the cost the arena
/// exists to remove.
/// What a vertex attribute's components ABOVE the guest's binding are fed. The value comes
/// from the LINKER, per attribute and per lane (`vitaslop_gxp_shader::attrflow`); this
/// function is only the knob that can override it with a whole-renderer constant for an A/B
/// (`api` = the graphics API's `(0, 0, 0, 1)`, `zero`, `one`).
///
/// # ONE CONSTANT CANNOT SERVE TWO TITLES - MEASURED, BOTH DIRECTIONS
/// A racer's cows are correct at 1.0 and turn BRIGHT GREEN/CYAN under a zero
/// (`f008800` is the discriminating frame; 10 of its 12 oracle frames are identical either
/// way). A fighting title's effect meshes do the exact reverse: `Input.Color` with BLUE pinned
/// to 1.0 is a saturated cyan mass, and `Input.CornerIndex` with two lanes pinned to 1 is a
/// stretched quad. The obvious discriminator - the guest's attribute FORMAT - is refuted:
/// both titles carry over-wide attributes on plain F32. What separates them is what each
/// lane FEEDS, which is why the answer is computed from the programs.
///
/// # THE (0, 0, 0, 1) FILL WAS NEVER A READING, AND IT COST A WHOLE TITLE ITS COLOUR
/// It is what WebGPU, GL and D3D supply, and this renderer supplied it only by binding the
/// NARROW vertex format and letting the API decide - which is not the same thing as deciding.
/// The guest's vertex fetch is what has to be reproduced, and nothing had ever asked what it
/// leaves there.
///
/// **MEASURED.** one title's sky/background family forwards `In.UV1` - declared four
/// components, BOUND `F16x2` - straight into a varying its fragment reads as a colour
/// MODULATE. Components `z`/`w` of an attribute nothing writes therefore multiply a channel,
/// and a zero there is a DEAD CHANNEL: that is the in-race world reading green/yellow with
/// the road at `(194, 220, 119)` and the sky's blue at 2. At 1.0 - a missing modulate
/// component being the identity rather than zero - the road's blue comes back to 199 and the
/// sky goes neutral. **`f008900.png` and the whole outdoor stretch stop being green.**
///
/// **Blast radius, measured rather than argued:** one title's title screen
/// (`f005600`) and another title's tutorial drive (`f002400`) are **BIT-IDENTICAL**
/// either way - their programs
/// do not read past what their guest binds. The only component this can change that the old
/// default also set is `w`, which both readings make 1.0.
pub(super) fn attr_fill(component: usize, linked: f32) -> f32 {
    use std::sync::OnceLock;
    static MODE: OnceLock<Option<f32>> = OnceLock::new();
    let mode = *MODE.get_or_init(|| match crate::knobs::var("VITASLOP_GXP_ATTR_FILL").ok().as_deref() {
        // The graphics API's fill, kept so the change above stays an A/B.
        Some("api") => Some(f32::NAN),
        Some("zero") => Some(0.0),
        // The whole-renderer 1.0 this replaced, kept as the other end of the same A/B.
        Some("one") => Some(1.0),
        _ => None,
    });
    match mode {
        Some(v) if v.is_nan() => {
            if component == 3 {
                1.0
            } else {
                0.0
            }
        }
        Some(v) => v,
        None => linked,
    }
}

pub(super) fn repack_vertices_into(
    vertices: &[u8],
    guest_stride: u32,
    repack: &[RepackAttr],
    packed_stride: u32,
    out: &mut Vec<u8>,
) {
    let gstride = guest_stride.max(1) as usize;
    let nverts = vertices.len() / gstride;
    out.reserve(nverts * packed_stride as usize);
    for i in 0..nverts {
        let vbase = i * gstride;
        // Zero-fill this packed vertex, then write each attribute at its packed offset.
        let start = out.len();
        out.resize(start + packed_stride as usize, 0);
        for a in repack {
            // The slot carries the guest's own component format, so its bytes are the
            // guest's bytes. A short guest buffer copies what there is and leaves the rest
            // of the slot at the row's zero fill, which is what the per-component reader
            // does for the same case.
            if a.copy_bytes > 0 {
                // CLAMPED, not just bounded. A stream too short to reach this attribute at
                // all puts `src` PAST the end, and `vertices[src..src]` panics on the range
                // start even when the length is zero - so the clamp is what makes the empty
                // copy legal rather than merely correct. The per-component path below reads
                // through `slice::get` and answers 0.0 for the same case; this is the same
                // answer, since the packed row is already zeroed.
                let src = (vbase + a.guest_offset as usize).min(vertices.len());
                let n = (a.copy_bytes as usize).min(vertices.len() - src);
                let po = start + a.packed_offset as usize;
                out[po..po + n].copy_from_slice(&vertices[src..src + n]);
                continue;
            }
            for c in 0..a.slots as usize {
                let f = if c < a.components as usize {
                    read_attr_component(vertices, vbase + a.guest_offset as usize, a.gxm_format, c)
                } else {
                    attr_fill(c, a.fill[c])
                };
                let po = start + a.packed_offset as usize + c * 4;
                out[po..po + 4].copy_from_slice(&f.to_le_bytes());
            }
        }
    }
}
