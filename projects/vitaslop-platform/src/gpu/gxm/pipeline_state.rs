//! The guest's fixed-function state as wgpu pipeline state - topology, cull, depth, stencil,
//! blend, colour mask - plus the small utility pipelines and the reports about all of them.

use super::*;

/// Link a guest shader pair and build its two pipeline variants + bind-group layouts.
/// `None` (fall back) on any link error or an unmappable vertex format.
///
/// Every `None` here is REPORTED, unconditionally and by name - see [`report_fallback`].
/// The translator itself never guesses (an unknown opcode hard-fails naming its raw word),
/// but the renderer's answer to that hard failure is to draw the fixed-function
/// approximation, and an unreported approximation reads exactly like a faithful render.
/// Report the fixed-function state a recompiled pair's pipeline was BAKED with, once per
/// pair. Every field here is the guest's own captured state, and a wrong one is invisible
/// in the frame in the worst way: a draw with the wrong depth function does not look
/// broken, it looks like a different draw won. This is the line that says which.
pub(super) fn report_gxp_pipeline_state(
    key: u64,
    gxp: &GxpRecompile,
    depth_write: bool,
    depth_compare: wgpu::CompareFunction,
    blend: Option<wgpu::BlendState>,
    write_mask: wgpu::ColorWrites,
) {
    report!(
        "gxp pipeline {key:x} (fragment program {:#x}{}): depth {depth_compare:?}{} (guest func {:#x}, write {}), \
         guest viewport depth zScale={} zOffset={}, depth bias (factor {}, units {}), \
         blend {}, colour mask {write_mask:?}, guest cull {:#x}{}",
        gxp.fprog_header,
        if gxp.fragment_program_enabled { "" } else { ", DEPTH/STENCIL ONLY - the guest disabled the fragment program" },
        if depth_write { " + WRITE" } else { "" },
        gxp.depth_func,
        gxp.depth_write,
        gxp.viewport[5],
        gxp.viewport[4],
        gxp.depth_bias.0,
        gxp.depth_bias.1,
        match blend {
            Some(b) => format!("{:?}/{:?}", b.color, b.alpha),
            None => "REPLACE (the guest supplied no SceGxmBlendInfo)".into(),
        },
        gxp.cull_mode,
        match gxm_cull_face(gxp.cull_mode) {
            Some(wgpu::Face::Front) => " - culling FRONT faces",
            Some(wgpu::Face::Back) => " - culling BACK faces",
            None => "",
        }
    );
}

/// Map a `SceGxmCullMode` to the wgpu face to discard, against `FrontFace::Ccw`.
///
/// # How the sense is pinned, since getting it backwards turns a model inside out
/// The software rasteriser pinned its own winding EMPIRICALLY against a title's ground
/// plane (`render::cull_backface`): in the Y-down framebuffer space `render::project`
/// emits, with `edge(a,b,c) = (bx-ax)(cy-ay) - (by-ay)(cx-ax)`, `SCE_GXM_CULL_CCW`
/// discards `edge < 0` and `SCE_GXM_CULL_CW` discards `edge > 0`.
///
/// The GPU side is pinned the same way, MEASURED, and it came out the OPPOSITE way round to
/// the obvious derivation - which is why it is measured rather than reasoned. Vulkan
/// defines its signed area as `1/2 * sum(x_i*y_{i+1} - x_{i+1}*y_i)` over framebuffer
/// coordinates, algebraically the same expression as `edge`, and calls it counter-clockwise
/// when POSITIVE; reading that straight across gives `CULL_CW -> Face::Front`, and a
/// retail race frame under it LOSES ITS WHOLE ROAD SURFACE - the camera looks
/// through the track at the terrain below. Under the
/// mapping below the same frame is the correct picture. Something between the guest's
/// viewport, our clip fixup and wgpu's framebuffer convention flips the sign once more than
/// that reading accounts for.
///
/// So: `CULL_CW` discards the BACK face and `CULL_CCW` the FRONT, and any future change here
/// has to be re-rendered against that frame rather than argued from a spec.
///
/// Both engines now keep the same triangles, which is the property that matters: the
/// software path has culled since it was written and the GPU path never did, so the two
/// halves of this renderer disagreed about which faces exist and neither could check the
/// other.
/// The wgpu topology for a GXM `SceGxmPrimitiveType`, and whether it is a triangle
/// family at all.
///
/// # Why only three answers, when GXM has six
/// The capture EXPANDS strips and fans into a flat, winding-normalised triangle LIST
/// before the geometry ever reaches this crate (`render::tri_indices`), so a strip
/// arrives here already spelled as a list and asking wgpu for `TriangleStrip` would
/// re-read the same indices under the wrong rule. An EDGE list (0x1400_0000) is
/// expanded the same way, into the LINE segments its per-triangle
/// `SceGxmEdgeEnableFlags` word enables (`render.rs`, the edge-list arm of the index
/// expansion), so it arrives here as line pairs and draws as `LineList`. Lines and
/// points cannot be expanded into anything, and used to be DROPPED - a self-reported
/// missing draw on a title that asks for one every frame ("not a triangle topology (a
/// line/point list emits no triangles)"). They now get the topology they asked for.
pub(super) fn gxm_topology(primitive: u32) -> wgpu::PrimitiveTopology {
    match primitive {
        0x0400_0000 | 0x1400_0000 => wgpu::PrimitiveTopology::LineList,
        0x0800_0000 => wgpu::PrimitiveTopology::PointList,
        _ => wgpu::PrimitiveTopology::TriangleList,
    }
}

/// Whether `primitive` rasterises faces, and so whether a cull mode means anything.
/// wgpu REJECTS a pipeline that pairs a cull mode with a line or point topology, and a
/// rejected pipeline blanks every draw of its pair
/// ([[vitaslop-one-refused-pipeline-blanks-the-frame]]). The edge list rasterises the
/// EDGES of its triangles, so what reaches the GPU is lines: no faces.
pub(super) fn gxm_topology_has_faces(primitive: u32) -> bool {
    !matches!(primitive, 0x0400_0000 | 0x0800_0000 | 0x1400_0000)
}

pub(super) fn gxm_cull_face(mode: u32) -> Option<wgpu::Face> {
    // `VITASLOP_GXP_CULL=0` is the A/B arm that restores "draw both windings", the behaviour
    // every pipeline had before 2026-08-19b. VALUE-sensitive, because it is an arm.
    if !crate::gpu::gxp_cull() {
        return None;
    }
    // `SceGxmCullMode`: NONE = 0, CW = 1, CCW = 2. The same three values `render.rs` names.
    match mode {
        1 => Some(wgpu::Face::Back),
        2 => Some(wgpu::Face::Front),
        _ => None,
    }
}

/// Map a `SceGxmDepthFunc` to its wgpu equivalent. The enum is a SHIFTED field (vitasdk
/// `gxm.h` spaces the values 0x00400000 apart), which is how it is stored in the sticky
/// render state, so it is normalised back to 0..7 here.
pub(super) fn gxm_depth_func(f: u32) -> wgpu::CompareFunction {
    use wgpu::CompareFunction as C;
    match (f >> 22) & 0x7 {
        0 => C::Never,
        1 => C::Less,
        2 => C::Equal,
        3 => C::LessEqual,
        4 => C::Greater,
        5 => C::NotEqual,
        6 => C::GreaterEqual,
        _ => C::Always,
    }
}

/// Map a `SceGxmStencilFunc` (bits 27:25, the depth func's order: NEVER, LESS, EQUAL,
/// LESS_EQUAL, GREATER, NOT_EQUAL, GREATER_EQUAL, ALWAYS) to its wgpu equivalent.
pub(super) fn gxm_stencil_func(f: u32) -> wgpu::CompareFunction {
    use wgpu::CompareFunction as C;
    match (f >> 25) & 0x7 {
        0 => C::Never,
        1 => C::Less,
        2 => C::Equal,
        3 => C::LessEqual,
        4 => C::Greater,
        5 => C::NotEqual,
        6 => C::GreaterEqual,
        _ => C::Always,
    }
}

/// Map a `SceGxmStencilOp` (vitasdk `gxm.h`: KEEP, ZERO, REPLACE, INCR, DECR, INVERT,
/// INCR_WRAP, DECR_WRAP) to its wgpu equivalent. INCR/DECR saturate; the WRAP forms wrap.
pub(super) fn gxm_stencil_op(op: u32) -> wgpu::StencilOperation {
    use wgpu::StencilOperation as S;
    match op & 0x7 {
        0 => S::Keep,
        1 => S::Zero,
        2 => S::Replace,
        3 => S::IncrementClamp,
        4 => S::DecrementClamp,
        5 => S::Invert,
        6 => S::IncrementWrap,
        _ => S::DecrementWrap,
    }
}

/// The draw's stencil state for its pipeline. With two-sided mode off the hardware applies
/// the FRONT state to both faces. WebGPU has ONE read mask and ONE write mask for both faces
/// (and one dynamic reference, set per draw from the front face) - a two-sided draw whose
/// faces disagree on those is reported, and takes the front face's.
/// One FIXED-FUNCTION pipeline: blend, depth AND fragment entry point differ between the
/// two modes the software rasterizer switches between per draw - the opaque (3D) path
/// depth-tests and writes, replaces, and runs `fs_opaque` (albedo texel + exposure/Reinhard);
/// the 2D path disables the depth test, straight-alpha src-over blends in submission order,
/// and runs `fs_blend` (vertex_color * texel modulate). `stencil` is the draw's own stencil
/// state - default (ALWAYS / KEEP) for the eager pair, the guest's for a masked draw.
pub(in crate::gpu) fn build_fixed_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    pipeline_layout: &wgpu::PipelineLayout,
    opaque: bool,
    target_format: wgpu::TextureFormat,
    samples: u32,
    stencil: wgpu::StencilState,
) -> wgpu::RenderPipeline {
    // Canonical vertex: pos float32x3 @0, uv float32x2 @12, color unorm8x4 @20,
    // world-space normal float32x3 @24 (for the opaque lighting term).
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: GXM_VERTEX_STRIDE,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 12, shader_location: 1 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Unorm8x4, offset: 20, shader_location: 2 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 3 },
        ],
    };
    let (blend, depth_write, depth_compare, fs) = if opaque {
        (
            Some(wgpu::BlendState::REPLACE),
            true,
            // LessEqual is GXM's default depth func and what this title's opaque 3D draws
            // set (SCE_GXM_DEPTH_FUNC_LESS_EQUAL); the software oracle compares the same way
            // (`depth_passes`), so a coincident later face ties and repaints identically on
            // both paths. The CPU builder already culled back faces.
            wgpu::CompareFunction::LessEqual,
            "fs_opaque",
        )
    } else {
        (
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            false,
            wgpu::CompareFunction::Always,
            "fs_blend",
        )
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(if opaque { "gxm-opaque" } else { "gxm-blend" }),
        layout: Some(pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs"),
            buffers: &[Some(vertex_layout)],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            // No culling: match the software rasterizer, which draws both windings.
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format(),
            depth_write_enabled: Some(depth_write),
            depth_compare: Some(depth_compare),
            stencil,
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
        multiview_mask: None,
        cache: None,
    })
}

/// The stencil state of a draw's raw GXM words (`[func, fail, depth_fail, depth_pass,
/// compare_mask, write_mask, ref]` front then back) - the recompiled path's
/// [`gxm_stencil_state`] on the raw array, for the fixed-function path.
pub(in crate::gpu) fn gxm_stencil_from_words(s: &[u32; 14], two_sided: bool) -> wgpu::StencilState {
    let face = |o: usize| wgpu::StencilFaceState {
        compare: gxm_stencil_func(s[o]),
        fail_op: gxm_stencil_op(s[o + 1]),
        depth_fail_op: gxm_stencil_op(s[o + 2]),
        pass_op: gxm_stencil_op(s[o + 3]),
    };
    let front = face(0);
    let back = if two_sided { face(7) } else { front };
    wgpu::StencilState { front, back, read_mask: s[4] & 0xff, write_mask: s[5] & 0xff }
}

/// Whether raw stencil words are GXM's default - ALWAYS, all ops KEEP - so the draw needs no
/// stencil variant (the masks and reference cannot change anything under ALWAYS/KEEP).
pub(in crate::gpu) fn stencil_words_are_default(s: &[u32; 14], two_sided: bool) -> bool {
    let face = |o: usize| (s[o] >> 25) & 7 == 7 && s[o + 1..o + 4].iter().all(|op| op & 7 == 0);
    face(0) && (!two_sided || face(7))
}

pub(super) fn gxm_stencil_state(gxp: &GxpRecompile) -> wgpu::StencilState {
    if crate::gpu::stencil_disabled() {
        return wgpu::StencilState::default();
    }
    let s = &gxp.stencil;
    let face = |o: usize| wgpu::StencilFaceState {
        compare: gxm_stencil_func(s[o]),
        fail_op: gxm_stencil_op(s[o + 1]),
        depth_fail_op: gxm_stencil_op(s[o + 2]),
        pass_op: gxm_stencil_op(s[o + 3]),
    };
    let front = face(0);
    let back = if gxp.two_sided { face(7) } else { front };
    if gxp.two_sided && (s[4] != s[11] || s[5] != s[12] || s[6] != s[13]) {
        report_two_sided_stencil_masks_differ(s);
    }
    wgpu::StencilState { front, back, read_mask: s[4] & 0xff, write_mask: s[5] & 0xff }
}

pub(super) fn report_two_sided_stencil_masks_differ(s: &[u32; 14]) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    report_warn!(
        "gxm stencil: a TWO-SIDED draw's faces disagree on compare mask / write mask / reference \
         (front {:#x}/{:#x}/{}, back {:#x}/{:#x}/{}) - WebGPU carries one of each for both \
         faces, so the FRONT face's are used. Reported once.",
        s[4], s[5], s[6], s[11], s[12], s[13]
    );
}

/// Map a `SceGxmBlendFactor` to its wgpu equivalent. The enum order is the vitasdk
/// `gxm.h` one; `SRC_ALPHA_SATURATE` has a wgpu counterpart and `DST_ALPHA_SATURATE`
/// (which wgpu has no factor for) is reported and treated as `One` rather than silently
/// becoming something else.
pub(super) fn gxm_blend_factor(f: u8) -> wgpu::BlendFactor {
    use wgpu::BlendFactor as F;
    match f {
        0 => F::Zero,
        1 => F::One,
        2 => F::Src,
        3 => F::OneMinusSrc,
        4 => F::SrcAlpha,
        5 => F::OneMinusSrcAlpha,
        6 => F::Dst,
        7 => F::OneMinusDst,
        8 => F::DstAlpha,
        9 => F::OneMinusDstAlpha,
        10 => F::SrcAlphaSaturated,
        _ => {
            report_unmapped_blend("DST_ALPHA_SATURATE blend factor");
            F::One
        }
    }
}

/// Map a `SceGxmBlendFunc` pair (func, src, dst) to a wgpu blend component. `MIN`/`MAX`
/// ignore the factors, exactly as the hardware does.
pub(super) fn gxm_blend_component(func: u8, src: u8, dst: u8) -> wgpu::BlendComponent {
    use wgpu::BlendOperation as O;
    let operation = match func {
        2 => O::Subtract,
        3 => O::ReverseSubtract,
        4 => O::Min,
        5 => O::Max,
        _ => O::Add,
    };
    // MIN and MAX take the operands unscaled.
    let (src, dst) = if matches!(func, 4 | 5) { (1, 1) } else { (src, dst) };
    wgpu::BlendComponent {
        src_factor: gxm_blend_factor(src),
        dst_factor: gxm_blend_factor(dst),
        operation,
    }
}

/// The wgpu blend state for a draw's captured `SceGxmBlendInfo`, or `None` when the
/// program does not blend at all (both funcs `NONE`), which is a REPLACE.
pub(super) fn gxm_blend_state(b: [u8; 7]) -> Option<wgpu::BlendState> {
    let [_mask, color_func, alpha_func, color_src, color_dst, alpha_src, alpha_dst] = b;
    if color_func == 0 && alpha_func == 0 {
        return Some(wgpu::BlendState::REPLACE);
    }
    Some(wgpu::BlendState {
        color: gxm_blend_component(color_func, color_src, color_dst),
        alpha: gxm_blend_component(alpha_func, alpha_src, alpha_dst),
    })
}

/// The wgpu blend state for an equation `lower_dest_blend` recovered from the shader.
/// Every factor maps one for one; nothing is approximated here.
pub(super) fn dest_blend_state(
    b: Option<vitaslop_gxp_shader::module::DestBlend>,
) -> Option<wgpu::BlendState> {
    use vitaslop_gxp_shader::module::BlendFactor as F;
    fn f(x: F) -> wgpu::BlendFactor {
        match x {
            F::Zero => wgpu::BlendFactor::Zero,
            F::One => wgpu::BlendFactor::One,
            F::Src => wgpu::BlendFactor::Src,
            F::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
            F::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        }
    }
    let b = b?;
    Some(wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: f(b.color.src),
            dst_factor: f(b.color.dst),
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: f(b.alpha.src),
            dst_factor: f(b.alpha.dst),
            operation: wgpu::BlendOperation::Add,
        },
    })
}

/// Report - once per pair - that this pipeline's blend came from the SHADER's own ALU rather
/// than from the guest's blend info or from the shader's SOP2 epilogue.
///
/// It reports for the same reason `report_rop_blend` does: a frame whose blending was
/// RECOVERED must not be indistinguishable from one where the guest asked for it. That is
/// the difference between reading a program correctly and having guessed well.
pub(super) fn report_lowered_dest_blend(
    key: u64,
    b: Option<vitaslop_gxp_shader::module::DestBlend>,
) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
        report!(
            "gxp pair {key:016x}: its fragment program BLENDS ITSELF over the destination colour in ALU, and that equation was lowered to pipeline state ({b:?}) - the shader now emits only the source term. `VITASLOP_GXP_DEST_BLEND=0` keeps the ALU form instead, which costs this draw a render-pass split."
        );
    }
}

/// Report - once per pair - that this pipeline's destination read was lowered to a
/// DUAL-SOURCE blend, so the draw costs no pass split. Same reason as
/// `report_lowered_dest_blend`: a recovered blend must be visible as one.
pub(super) fn report_dual_source_blend(key: u64, guest_blend: [u8; 7]) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
        // At STATUS, and with the guest's own blend info beside it: a program that blends
        // in ALU was created with blend info NONE, and one that was not is a program the
        // split path composited TWICE - the two arms then differ and the reading of that
        // difference starts here.
        report_status!(
            "gxp pair {key:016x}: its fragment program reads the destination colour and is \
             LINEAR in it, so it was lowered to a DUAL-SOURCE blend (two shader outputs, \
             `src0 + dst*src1`) - no attachment copy and no render-pass split. \
             `VITASLOP_GXP_DUAL_SOURCE=0` keeps the split instead. Guest blend info                  {guest_blend:?}"
        );
    }
}

/// The wgpu write mask for a `SceGxmColorMask`: bit 0 **A**, 1 R, 2 G, 3 B (vitasdk `gxm.h`:
/// `SCE_GXM_COLOR_MASK_A = 1 << 0`, `_R = 1 << 1`, `_G = 1 << 2`, `_B = 1 << 3`).
///
/// >>> ALPHA IS THE LOW BIT. This read bit 0 as red until a fighting title's hair pass asked
/// for mask 14 (RGB, keep the destination ALPHA its post chain reads) and got "G, B and A,
/// no red": the hair kept the red of the face drawn under it and came out cyan with a pink
/// band - measured by key-colouring the face pair, which moved the hair pixel's RED alone.
/// `VITASLOP_GXM_COLOR_MASK_ORDER=0` restores the old order (the negative control).
pub(super) fn gxm_color_mask(mask: u8) -> wgpu::ColorWrites {
    use std::sync::OnceLock;
    static LEGACY: OnceLock<bool> = OnceLock::new();
    let legacy = *LEGACY.get_or_init(|| {
        crate::knobs::var("VITASLOP_GXM_COLOR_MASK_ORDER").map(|v| v.trim() == "0").unwrap_or(false)
    });
    let order = if legacy {
        [
            (0, wgpu::ColorWrites::RED),
            (1, wgpu::ColorWrites::GREEN),
            (2, wgpu::ColorWrites::BLUE),
            (3, wgpu::ColorWrites::ALPHA),
        ]
    } else {
        [
            (0, wgpu::ColorWrites::ALPHA),
            (1, wgpu::ColorWrites::RED),
            (2, wgpu::ColorWrites::GREEN),
            (3, wgpu::ColorWrites::BLUE),
        ]
    };
    let mut w = wgpu::ColorWrites::empty();
    for (bit, flag) in order {
        if mask & (1 << bit) != 0 {
            w |= flag;
        }
    }
    w
}

/// Report - once per render target - that `VITASLOP_GXP_NEGW=force` corrected this pass's
/// clip `w` without measuring anything.
pub(super) fn report_forced_negw(target: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(target) {
        report!(
            "gxp clip: pass into {target:#010x}: VITASLOP_GXP_NEGW=force - correcting the \
             clip w sign with NO measurement behind it. This is a diagnostic; the frame it \
             produces is not evidence of anything except what the correction does."
        );
    }
}

/// Report how many render-pass SPLITS a pass took to serve its destination-colour reads,
/// because that count is the whole cost of the feature and nothing else states it outside
/// the perf panel.
///
/// A tiling GPU stores and reloads the framebuffer's tiles at every split, so this number
/// times the attachment size is the memory traffic the phone pays that a desktop barely
/// notices.
///
/// # >>> ONCE PER PASS SHAPE, NOT ONCE PER RUN
/// This fired exactly once for a whole run, which on a title whose frame carries FIFTEEN
/// passes described the first one and never mentioned the other fourteen - and the first
/// pass a frame encodes is routinely a small offscreen one, not the pass the splits are in
/// [[vitaslop-a-zero-draw-pass-can-report-the-queues-wait]]. A run reading "82 splits" was
/// therefore an unknown fraction of the frame's real total, which is the number any fix has
/// to be judged against.
///
/// Keyed on the SHAPE instead - the split count, the draw count and the attachment size -
/// so every distinct pass in the frame gets one line and a steady state repeating that same
/// shape every frame gets no more. The set is bounded by the number of distinct pass shapes
/// a title has, which is small; it is not a per-frame log.
pub(super) fn report_dest_split_count(splits: u64, draws: usize, w: u32, h: u32, pairs: &[u64]) {
    use std::sync::Mutex;
    static SEEN: std::sync::OnceLock<Mutex<HashSet<(u64, usize, u32, u32)>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert((splits, draws, w, h)) {
        let mut by_pair: HashMap<u64, usize> = HashMap::default();
        for k in pairs {
            *by_pair.entry(*k).or_default() += 1;
        }
        let mut ranked: Vec<(u64, usize)> = by_pair.into_iter().collect();
        ranked.sort_by_key(|&(k, n)| (std::cmp::Reverse(n), k));
        let who: Vec<String> =
            ranked.iter().map(|(k, n)| format!("{k:016x} x{n}")).collect();
        report_status!(
            "gxp dest colour: this pass took {splits} render-pass SPLIT(S) over {draws} draws to give its self-blending draws the destination colour - {splits} copies of a {w}x{h} attachment, and on a tiling GPU {splits} extra tile store/reload cycles. The perf panel counts every one (`DESTINATION-COLOUR pass splits`). BY PAIR, most first: {}",
            who.join(", ")
        );
    }
}

/// Report - once - that a pass carries a draw whose fragment program reads the destination
/// colour but is MULTISAMPLED, so the copy that feeds it cannot be taken.
///
/// `copy_texture_to_texture` refuses a multisampled source, and resolving mid-pass produces
/// a different image from the one the ROP would feed back. Neither is the destination, so
/// the draws are dropped and this says why rather than letting them blend against a stale
/// or empty texture.
/// The draw that copies a multisampled attachment into a multisampled [`DestColor`] sample
/// for sample - see [`GxmRenderer::copy_attachment_to_dest`].
pub(super) fn dest_ms_copy_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    samples: u32,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    const SRC: &str = "@group(0) @binding(0) var src: texture_multisampled_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4<f32>, @builtin(sample_index) s: u32) -> @location(0) vec4<f32> {
    return textureLoad(src, vec2<i32>(pos.xy), i32(s));
}
";
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gxp-dest-ms-copy"),
        source: wgpu::ShaderSource::Wgsl(SRC.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gxp-dest-ms-copy"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: true,
            },
            count: None,
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gxp-dest-ms-copy"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gxp-dest-ms-copy"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, layout)
}

/// The draw that writes a single-sample image into EVERY sample of a multisampled
/// attachment - see [`GxmRenderer::seed_msaa_from_color`].
pub(super) fn msaa_seed_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    samples: u32,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    const SRC: &str = "@group(0) @binding(0) var src: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(src, vec2<i32>(pos.xy), 0);
}
";
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gxm-msaa-seed"),
        source: wgpu::ShaderSource::Wgsl(SRC.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gxm-msaa-seed"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gxm-msaa-seed"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gxm-msaa-seed"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
        }),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, layout)
}

/// A full-screen draw writing alpha 1.0 and nothing else - see
/// [`GxmRenderer::fill_alpha_one`].
pub(super) fn alpha_fill_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    const SRC: &str = "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
";
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gxm-alpha-fill"),
        source: wgpu::ShaderSource::Wgsl(SRC.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gxm-alpha-fill"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gxm-alpha-fill"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALPHA })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// The stencil-plane conversion of [`GxmRenderer::convert_gxm_stencil`]: load the stencil
/// aspect as an integer and write it as the RGBA8 a U8 texture with `SWIZZLE1` selector
/// `selector` decodes to - the same table as the CPU decode (`render::swizzle1`).
pub(super) fn stencil_convert_pipeline(device: &wgpu::Device, selector: u32) -> (wgpu::BindGroupLayout, wgpu::RenderPipeline) {
    let swizzled = match selector {
        0 => "vec4<f32>(r, 0.0, 0.0, 1.0)",
        1 => "vec4<f32>(r, 0.0, 0.0, 0.0)",
        2 => "vec4<f32>(r, 1.0, 1.0, 1.0)",
        3 => "vec4<f32>(r, r, r, r)",
        4 => "vec4<f32>(r, r, r, 0.0)",
        5 => "vec4<f32>(r, r, r, 1.0)",
        6 => "vec4<f32>(0.0, 0.0, 0.0, r)",
        _ => "vec4<f32>(1.0, 1.0, 1.0, r)",
    };
    let src = format!(
        "@group(0) @binding(0) var s: texture_2d<u32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {{
    let r = f32(textureLoad(s, vec2<i32>(pos.xy), 0).r) / 255.0;
    return {swizzled};
}}
"
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gxm-stencil-convert"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("gxm-stencil-convert"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("gxm-stencil-convert"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("gxm-stencil-convert"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    (layout, pipe)
}

/// Say, once per address, that a separate stencil plane is sampled and served from the
/// stencil attachment - see [`GxmRenderer::convert_gxm_stencil`].
pub(super) fn report_stencil_plane_sampled(addr: u32, w: u32, h: u32, selector: u32) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
    if SEEN.get_or_init(|| Mutex::new(HashSet::default())).lock().unwrap_or_else(|e| e.into_inner()).insert(addr) {
        report_status!(
            "gxm stencil plane: a later pass SAMPLES the {w}x{h} stencil plane at {addr:#x} as a U8 texture (SWIZZLE1 selector {selector}) - it is converted from the stencil attachment after each pass that writes it, so the sampler reads what the scene marked rather than guest bytes no pass wrote"
        );
    }
}

/// Say, once per address, that a sampled stencil plane is NOT served, and why: the sampler
/// reads the guest bytes at that address instead, which no pass wrote.
pub(super) fn report_stencil_sample_unserved(addr: u32, why: &str) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u32>>> = OnceLock::new();
    if SEEN.get_or_init(|| Mutex::new(HashSet::default())).lock().unwrap_or_else(|e| e.into_inner()).insert(addr) {
        report_warn!(
            "gxm stencil plane: a later pass samples the stencil plane at {addr:#x}, but {why} - the sampler reads the guest bytes at that address, which no pass wrote"
        );
    }
}

pub(super) fn report_dest_needs_single_sample(samples: u32, has_texture: bool) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        report_warn!(
            "gxp dest colour: a draw in this pass has a fragment program that reads the DESTINATION colour, and the pass cannot supply one ({}). Those draws are DROPPED rather than blended against the wrong image.",
            if !has_texture {
                "the colour attachment is a view the host owns, with no texture behind it this renderer can copy from"
            } else {
                "the attachment is multisampled: it cannot be copied texel-for-texel, and its resolve is not what the ROP feeds back"
            }
        );
        let _ = samples;
    }
}

/// Report - once per pair - that a pair whose fragment program READS THE DESTINATION COLOUR
/// was drawn into a pass that has no destination texture, so the draw was dropped.
///
/// This is a renderer bug if it ever fires: the pass pre-scan and this check ask the same
/// question of the same blob, so they cannot disagree unless one of them is wrong. It is a
/// report rather than a panic because a dropped draw is recoverable and a dead run is not.
pub(super) fn report_missing_dest_texture(key: u64) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
        report_warn!(
            "gxp pair {key:016x}: its fragment program reads the DESTINATION colour, but this pass has no destination texture to read - the draw is DROPPED. The pass pre-scan and the pipeline disagree about this pair, which is a renderer bug."
        );
    }
}

/// Report - once per shader pair - that the guest's own `SceGxmBlendInfo` gives this pair a
/// colour write mask of ZERO, so every draw using it writes no colour channel.
///
/// This is not a fallback and not an approximation: it is the guest's state, faithfully
/// applied, and a depth-only prepass is exactly what it looks like. It is reported because
/// the RESULT on screen - a pass that renders nothing - is identical to a draw whose
/// geometry, textures or transform we got wrong, and this is the only one of those that is
/// correct. Knowing which costs one line here and a whole investigation otherwise.
pub(super) fn report_zero_color_mask(key: u64) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    if seen.lock().unwrap_or_else(|e| e.into_inner()).insert(key) {
        report!(
            "gxp pair {key:016x}: the guest's blend info gives this pair a colour write mask \
             of 0 - it writes NO colour channel, so its draws affect depth only. That is the \
             guest's own state, not a fallback."
        );
    }
}

/// Report - once per case - a GXM blend value with no exact wgpu equivalent, so the
/// substitution is visible rather than silently changing what a draw composites like.
pub(super) fn report_unmapped_blend(what: &str) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::default()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert(what.to_string()) {
        report_warn!("gxm blend: {what} has no wgpu equivalent - substituting ONE, which is an approximation");
    }
}

/// The blend for a single-channel ALPHA target (`alpha_single_color_format`): the guest's
/// ALPHA component drives every host channel. The stored alpha lives in the host's RED
/// channel, so a factor naming the DESTINATION ALPHA has to read the destination colour
/// instead; source-alpha factors are unchanged because the source is an alpha splat.
pub(super) fn alpha_single_blend(b: wgpu::BlendState) -> wgpu::BlendState {
    use wgpu::BlendFactor as F;
    let dst_as_red = |f: F| match f {
        F::DstAlpha => F::Dst,
        F::OneMinusDstAlpha => F::OneMinusDst,
        other => other,
    };
    let c = wgpu::BlendComponent {
        src_factor: dst_as_red(b.alpha.src_factor),
        dst_factor: dst_as_red(b.alpha.dst_factor),
        operation: b.alpha.operation,
    };
    wgpu::BlendState { color: c, alpha: c }
}
