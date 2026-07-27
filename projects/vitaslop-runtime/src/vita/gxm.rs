//! SceGxm: the graphics API. These handlers hand back opaque handles for GXM
//! objects, remember the surfaces and vertex-program layouts the guest sets up,
//! and record the per-frame draw stream (BeginScene to EndScene) with the vertex,
//! index, and uniform data snapshotted from guest memory. No GPU is emulated and
//! no pixel is drawn here; that is the renderer's job over this capture.

use crate::capture::{ColorSurface, VertexAttribute};
use crate::host::{GuestCtx, VitaState, MAX_VERTEX_STREAMS};
use crate::hostcall;

/// SceGxmInitializeParams: displayQueueCallback at offset 8, its data size at 12.
const INIT_CB_OFFSET: u32 = 8;
const INIT_CB_DATA_SIZE_OFFSET: u32 = 12;

/// A call that succeeds with return code 0 and no out-params.
pub(super) fn ok(ctx: &mut GuestCtx) {
    ctx.ret(0);
}

/// Map*UsseMemory(base, size, unsigned int *usseOffset): return offset 0.
pub(super) fn map_usse(ctx: &mut GuestCtx) {
    let out = ctx.arg(2);
    ctx.write_u32(out, 0);
    ctx.ret(0);
}

/// A create-call that writes a fresh opaque handle to its out-pointer at
/// positional argument `out_arg`, returning 0.
pub(super) fn out_handle(ctx: &mut GuestCtx, st: &mut VitaState, out_arg: usize) {
    let out = ctx.arg(out_arg);
    let handle = st.new_handle();
    ctx.write_u32(out, handle);
    ctx.ret(0);
}

/// int sceGxmShaderPatcherRegisterProgram(patcher, const SceGxmProgram
///     *programHeader, SceGxmShaderPatcherId *programId)
/// Records the program header so a later GetProgramFromId can return it, and hands
/// back the opaque id.
pub(super) fn register_program(ctx: &mut GuestCtx, st: &mut VitaState) {
    let program_header = ctx.arg(1);
    let out = ctx.arg(2);
    let id = st.register_shader_program(program_header);
    ctx.write_u32(out, id);
    ctx.ret(0);
}

/// const SceGxmProgram *sceGxmShaderPatcherGetProgramFromId(SceGxmShaderPatcherId
///     programId)
/// Returns the program header the guest registered under `programId`.
pub(super) fn get_program_from_id(ctx: &mut GuestCtx, st: &mut VitaState) {
    let id = ctx.arg(0);
    let program = st.shader_program(id);
    ctx.ret(program);
}

/// int sceGxmInitialize(const SceGxmInitializeParams *params)
pub(super) fn initialize(ctx: &mut GuestCtx, st: &mut VitaState) {
    let params = ctx.arg(0);
    st.display_queue_cb = ctx.read_u32(params + INIT_CB_OFFSET);
    st.display_queue_cb_data_size = ctx.read_u32(params + INIT_CB_DATA_SIZE_OFFSET);
    ctx.ret(0);
}

// --- SceGxmProgram reflection over the real gxp parameter table -------------
//
// The compiled shader (`SceGxmProgram`) carries a table of `SceGxmProgramParameter`
// records that a title reflects over to build its own uniform/attribute bookkeeping.
// Layout (stable across the public permissive gxp tooling, verified against the real
// shaders in the game image): the program header holds `parameter_count` at +0x24 and
// `parameters_offset` at +0x28 (relative to the +0x28 field itself); each parameter is
// a 16-byte record - `name_offset` (i32, relative to the record) at +0x00, a packed
// u16 at +0x04 (`category`:4, `type`:4, `component_count`:4, `container_index`:4),
// `array_size` (u32) at +0x08, and `resource_index` (u32) at +0x0C.
const GXP_PARAM_COUNT_OFF: u32 = 0x24;
const GXP_PARAMS_OFF_OFF: u32 = 0x28;
const GXM_PARAM_SIZE: u32 = 16;

/// Base guest address of the parameter array for `program`.
fn params_base(ctx: &GuestCtx, program: u32) -> u32 {
    program
        .wrapping_add(GXP_PARAMS_OFF_OFF)
        .wrapping_add(ctx.read_u32(program.wrapping_add(GXP_PARAMS_OFF_OFF)))
}

/// The packed u16 attribute word (`category`/`type`/`component_count`/`container_index`).
fn param_word(ctx: &GuestCtx, param: u32) -> u32 {
    ctx.read_u32(param.wrapping_add(4)) & 0xffff
}

/// `VITASLOP_NO_INLINE_IMPORTS`: route every host call through the host, even the
/// ones the transpiler could emit inline.
///
/// This is the A/B switch for the inline mechanism, and it earns its keep because
/// inlining changes how much wasm the guest executes, which changes fuel consumption,
/// which changes WHERE the preemptive scheduler switches threads - so an inlined build
/// legitimately reports a different determinism signature without computing anything
/// differently. Turning inlining off is how a signature is compared against a
/// pre-inlining run, which is the only way to tell a real behaviour change from that
/// re-interleaving. Read at LINK time, so it must be set for the whole run.
fn no_inline_imports() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VITASLOP_NO_INLINE_IMPORTS").is_some())
}

/// Byte offset of the packed attribute word within a `SceGxmProgramParameter`.
const GXM_PARAM_WORD_OFF: u32 = 4;
/// Byte offset of a parameter's `array_size`.
const GXM_PARAM_ARRAY_SIZE_OFF: u32 = 8;
/// Byte offset of a parameter's `resource_index`.
const GXM_PARAM_RESOURCE_INDEX_OFF: u32 = 0xC;

/// The inline form of a host import, for the pure GXM reflection getters - or `None`
/// for every NID that has real behaviour and must stay a host call.
///
/// These four getters are, together, the majority of every host call a gameplay frame
/// makes: a title re-reflects its shader parameter tables per material, per frame. Each
/// is one word load plus a shift and a mask over a guest structure, with no host state
/// involved at all, so the transpiler can emit it straight into the guest code and skip
/// the boundary entirely. See [`vitaslop_transpiler::InlineImport`].
///
/// **Every entry here duplicates a handler above, and the two must agree.** The
/// `inline_ops_match_their_handlers` test holds them to that; add nothing here without
/// extending it.
pub(crate) fn inline_op(func_nid: u32) -> Option<vitaslop_transpiler::InlineOp> {
    use crate::nid::gxm as g;
    use vitaslop_transpiler::InlineOp::LoadShiftMask;
    // The packed word's fields; `param_word` masks the word to 16 bits first, which
    // the 4-bit field masks below make redundant.
    let word = |shift| LoadShiftMask { offset: GXM_PARAM_WORD_OFF, shift, mask: 0xf };
    if no_inline_imports() {
        return None;
    }
    Some(match func_nid {
        g::PROGRAM_PARAMETER_GET_CATEGORY => word(0),
        g::PROGRAM_PARAMETER_GET_TYPE => word(4),
        g::PROGRAM_PARAMETER_GET_COMPONENT_COUNT => word(8),
        g::PROGRAM_PARAMETER_GET_CONTAINER_INDEX => word(12),
        g::PROGRAM_PARAMETER_GET_ARRAY_SIZE => {
            LoadShiftMask { offset: GXM_PARAM_ARRAY_SIZE_OFF, shift: 0, mask: u32::MAX }
        }
        g::PROGRAM_PARAMETER_GET_RESOURCE_INDEX => {
            LoadShiftMask { offset: GXM_PARAM_RESOURCE_INDEX_OFF, shift: 0, mask: u32::MAX }
        }
        _ => return None,
    })
}

/// unsigned int sceGxmProgramGetParameterCount(const SceGxmProgram *program)
pub(super) fn program_get_parameter_count(ctx: &mut GuestCtx) {
    let program = ctx.arg(0);
    let count = ctx.read_u32(program.wrapping_add(GXP_PARAM_COUNT_OFF));
    ctx.ret(count);
}

/// const SceGxmProgramParameter *sceGxmProgramGetParameter(program, unsigned int index)
pub(super) fn program_get_parameter(ctx: &mut GuestCtx) {
    let program = ctx.arg(0);
    let index = ctx.arg(1);
    ctx.ret(params_base(ctx, program).wrapping_add(index.wrapping_mul(GXM_PARAM_SIZE)));
}

/// SceGxmParameterCategory sceGxmProgramParameterGetCategory(param)
pub(super) fn param_get_category(ctx: &mut GuestCtx) {
    let param = ctx.arg(0);
    ctx.ret(param_word(ctx, param) & 0xf);
}

/// SceGxmParameterType sceGxmProgramParameterGetType(param)
pub(super) fn param_get_type(ctx: &mut GuestCtx) {
    let param = ctx.arg(0);
    ctx.ret((param_word(ctx, param) >> 4) & 0xf);
}

/// unsigned int sceGxmProgramParameterGetComponentCount(param)
pub(super) fn param_get_component_count(ctx: &mut GuestCtx) {
    let param = ctx.arg(0);
    ctx.ret((param_word(ctx, param) >> 8) & 0xf);
}

/// unsigned int sceGxmProgramParameterGetContainerIndex(param)
pub(super) fn param_get_container_index(ctx: &mut GuestCtx) {
    let param = ctx.arg(0);
    ctx.ret((param_word(ctx, param) >> 12) & 0xf);
}

/// unsigned int sceGxmProgramParameterGetArraySize(param)
pub(super) fn param_get_array_size(ctx: &mut GuestCtx) {
    let param = ctx.arg(0);
    ctx.ret(ctx.read_u32(param.wrapping_add(8)));
}

/// unsigned int sceGxmProgramParameterGetResourceIndex(param)
pub(super) fn param_get_resource_index(ctx: &mut GuestCtx) {
    let param = ctx.arg(0);
    ctx.ret(ctx.read_u32(param.wrapping_add(0xC)));
}

/// const char *sceGxmProgramParameterGetName(param): the name string lives at
/// `param + name_offset` (a signed byte offset relative to the record).
pub(super) fn param_get_name(ctx: &mut GuestCtx) {
    let param = ctx.arg(0);
    let name_off = ctx.read_u32(param) as i32;
    ctx.ret(param.wrapping_add(name_off as u32));
}

/// const SceGxmProgramParameter *sceGxmProgramFindParameterByName(program, name):
/// walk the real parameter table and return the first record whose name matches, or
/// null. Returning a real record (not an opaque token) lets a title reflect on the
/// match with the accessors above - the resource index, container, type, etc.
pub(super) fn find_parameter(ctx: &mut GuestCtx, _st: &mut VitaState) {
    let program = ctx.arg(0);
    let want = ctx.read_cstr(ctx.arg(1), 256);
    let count = ctx.read_u32(program.wrapping_add(GXP_PARAM_COUNT_OFF));
    let base = params_base(ctx, program);
    for i in 0..count {
        let p = base.wrapping_add(i.wrapping_mul(GXM_PARAM_SIZE));
        let name_off = ctx.read_u32(p) as i32;
        if ctx.read_cstr(p.wrapping_add(name_off as u32), 256) == want {
            ctx.ret(p);
            return;
        }
    }
    ctx.ret(0);
}

/// Marks a `SceGxmColorSurface` this implementation initialised, in the first of the
/// eight opaque words hardware GXM fills with PBE state (`pbeSidebandWord` +
/// `pbeEmitWords[6]` + `outputRegisterSize` = 32 bytes, ahead of `backgroundTex`).
///
/// The surface's identity has to live IN the guest struct, not only in a host table
/// keyed by the struct's address, because a title is entitled to COPY the struct - and
/// PCSE00001 does: it initialises its nine render targets into a static array and then
/// assigns each into a per-pass command object, so every `sceGxmBeginScene` in a race
/// frame names an address `sceGxmColorSurfaceInit` was never called with. Address-keyed
/// lookup then reports "no surface" for exactly the render-to-texture passes that draw
/// the world, and only the final composite (the HUD) survives to be rendered.
const COLOR_SURFACE_MAGIC: u32 = 0x5343_5356;

/// Byte offsets of our fields within the surface's opaque word block.
const CS_FORMAT: u32 = 4;
const CS_TYPE: u32 = 8;
const CS_WIDTH: u32 = 12;
const CS_HEIGHT: u32 = 16;
const CS_STRIDE: u32 = 20;
const CS_DATA: u32 = 24;

/// Write `surface`'s fields into the guest struct so a copy of it stays resolvable.
fn write_color_surface(ctx: &mut GuestCtx, addr: u32, s: &ColorSurface) {
    ctx.write_u32(addr, COLOR_SURFACE_MAGIC);
    ctx.write_u32(addr + CS_FORMAT, s.format);
    ctx.write_u32(addr + CS_TYPE, s.surface_type);
    ctx.write_u32(addr + CS_WIDTH, s.width);
    ctx.write_u32(addr + CS_HEIGHT, s.height);
    ctx.write_u32(addr + CS_STRIDE, s.stride_pixels);
    ctx.write_u32(addr + CS_DATA, s.data_addr);
}

/// Read back a surface written by [`write_color_surface`], or `None` if this address
/// does not hold one.
pub(super) fn read_color_surface(ctx: &mut GuestCtx, addr: u32) -> Option<ColorSurface> {
    if addr == 0 || ctx.read_u32(addr) != COLOR_SURFACE_MAGIC {
        return None;
    }
    Some(ColorSurface {
        format: ctx.read_u32(addr + CS_FORMAT),
        surface_type: ctx.read_u32(addr + CS_TYPE),
        width: ctx.read_u32(addr + CS_WIDTH),
        height: ctx.read_u32(addr + CS_HEIGHT),
        stride_pixels: ctx.read_u32(addr + CS_STRIDE),
        data_addr: ctx.read_u32(addr + CS_DATA),
    })
}

/// int sceGxmColorSurfaceInit(surface, format, type, scale, outputRegisterSize,
///     width, height, strideInPixels, data)  -- 9 args (5 on the stack).
pub(super) fn color_surface_init(ctx: &mut GuestCtx, st: &mut VitaState) {
    let surface = ctx.arg(0);
    let format = ctx.arg(1);
    let surface_type = ctx.arg(2);
    let width = ctx.arg(5);
    let height = ctx.arg(6);
    let stride_pixels = ctx.arg(7);
    let data_addr = ctx.arg(8);
    tracing::debug!(
        target: "vitaslop::gxm",
        surface = format_args!("{surface:#x}"),
        data = format_args!("{data_addr:#x}"),
        width, height, stride_pixels,
        format = format_args!("{format:#x}"),
        "colorSurfaceInit"
    );
    let s = ColorSurface { format, surface_type, width, height, stride_pixels, data_addr };
    write_color_surface(ctx, surface, &s);
    st.set_color_surface(surface, s);
    ctx.ret(0);
}

/// int sceGxmShaderPatcherCreateVertexProgram(patcher, programId, attributes,
///     attributeCount, streams, streamCount, vertexProgram)  -- 7 args.
pub(super) fn create_vertex_program(ctx: &mut GuestCtx, st: &mut VitaState) {
    let program_id = ctx.arg(1);
    let attributes_addr = ctx.arg(2);
    let attribute_count = ctx.arg(3);
    let streams_addr = ctx.arg(4);
    let stream_count = ctx.arg(5);
    let out = ctx.arg(6);

    let mut attributes = Vec::new();
    for i in 0..attribute_count {
        let a = attributes_addr + i * 8;
        let raw = ctx.read_bytes(a, 8);
        if raw.len() < 8 {
            break;
        }
        attributes.push(VertexAttribute {
            stream_index: u16::from_le_bytes([raw[0], raw[1]]),
            offset: u16::from_le_bytes([raw[2], raw[3]]),
            format: raw[4],
            component_count: raw[5],
            reg_index: u16::from_le_bytes([raw[6], raw[7]]),
        });
    }
    // SceGxmVertexStream is `{ uint16_t stride; uint16_t indexSource; }` - one per stream.
    // `indexSource` 2 and 3 are the INSTANCE variants (16- and 32-bit), which step the
    // stream by instance rather than by vertex.
    let streams: Vec<(u32, bool)> = (0..stream_count.min(MAX_VERTEX_STREAMS as u32))
        .map(|i| {
            let w = ctx.read_u32(streams_addr + i * 4);
            (w & 0xFFFF, ((w >> 16) & 0xFFFF) >= 2)
        })
        .collect();
    let stride = streams.first().map(|s| s.0).unwrap_or(0);

    tracing::debug!(
        target: "vitaslop::gxm",
        attribute_count, stream_count, stride, attrs = attributes.len(),
        attrs_addr = format_args!("{attributes_addr:#x}"),
        streams_addr = format_args!("{streams_addr:#x}"),
        "createVertexProgram"
    );
    // A vertex program with NO attributes over a stream that has a real stride cannot
    // fetch anything: every draw through it arrives with geometry the renderer has no
    // layout for, and the frame comes out empty. That is almost always a HOST bug (the
    // title built its attribute array by reflecting the shader, and some reflection call
    // told it there was nothing to bind), so it is reported unconditionally rather than
    // accepted quietly - the empty frame it produces is otherwise indistinguishable from
    // a title that simply drew nothing.
    if attributes.is_empty() && stride > 0 {
        tracing::warn!(
            target: "vitaslop::gxm",
            stride,
            attribute_count,
            program = format_args!("{program_id:#x}"),
            "vertex program created with ZERO attributes over a stride-{stride} stream - \
             every draw through it will have no fetchable geometry"
        );
    }

    // Resolve the shader-patcher id back to its `SceGxmProgram*` so a precomputed
    // vertex state built from this vertex program can size its default uniform buffer.
    let program_header = st.shader_program(program_id);
    let handle = st.new_handle();
    st.set_vertex_program(handle, attributes, streams, program_header);
    ctx.write_u32(out, handle);
    ctx.ret(0);
}

/// int sceGxmShaderPatcherCreateFragmentProgram(patcher, programId, outputFormat,
///     multisampleMode, blendInfo, vertexProgram, fragmentProgram)  -- 7 args.
/// Hand back a fresh handle (as the generic `out_handle` did) and additionally record
/// the handle -> `SceGxmProgram*` mapping, so a precomputed fragment state built from
/// this fragment program can size its default uniform buffer.
pub(super) fn create_fragment_program(ctx: &mut GuestCtx, st: &mut VitaState) {
    let program_id = ctx.arg(1);
    let out = ctx.arg(6);
    let program_header = st.shader_program(program_id);
    let handle = st.new_handle();
    st.set_fragment_program(handle, program_header);
    ctx.write_u32(out, handle);
    ctx.ret(0);
}

/// int sceGxmBeginScene(context, flags, renderTarget, validRegion,
///     vertexSyncObject, fragmentSyncObject, colorSurface, depthStencil) -- 8 args.
pub(super) fn begin_scene(ctx: &mut GuestCtx, st: &mut VitaState) {
    let color_surface = ctx.arg(6);
    // Resolve from the struct's own contents first (survives a copy), then from the
    // address table (covers a surface whose bytes a title has since overwritten).
    let color = read_color_surface(ctx, color_surface).or_else(|| st.color_surface(color_surface));
    if color.is_none() {
        tracing::debug!(
            target: "vitaslop::gxm",
            surface = format_args!("{color_surface:#x}"),
            "beginScene with an unrecognised colour surface - this scene's render target is \
             unknown, so a later pass sampling it cannot be chained to it"
        );
    }
    st.begin_scene(color);
    ctx.ret(0);
}

/// int sceGxmEndScene(context, notification, notification2)
pub(super) fn end_scene(ctx: &mut GuestCtx, st: &mut VitaState) {
    st.end_scene();
    ctx.ret(0);
}

/// void sceGxmSetVertexProgram(context, vertexProgram)
pub(super) fn set_vertex_program(ctx: &mut GuestCtx, st: &mut VitaState) {
    let vp = ctx.arg(1);
    st.bind_vertex_program(vp);
    ctx.ret(0);
}

/// void sceGxmSetFragmentProgram(context, fragmentProgram)
/// Record the bound fragment program's `SceGxmProgram*` so `record_draw` can reflect
/// its samplers to pick the albedo texture. Rendering does not otherwise consume the
/// fragment program (the capture renderer is fixed-function).
pub(super) fn set_fragment_program(ctx: &mut GuestCtx, st: &mut VitaState) {
    let fp = ctx.arg(1);
    st.bind_fragment_program(fp);
    ctx.ret(0);
}

/// int sceGxmReserveVertexDefaultUniformBuffer(context, void **uniformBuffer)
/// Hand back a real guest buffer AND bind it as the vertex uniform source, so the
/// uniforms the guest writes into it (its MVP and friends, on the direct draw path)
/// are captured at draw time - see [`VitaState::reserve_vertex_uniform_buffer`].
pub(super) fn reserve_vertex_uniforms(ctx: &mut GuestCtx, st: &mut VitaState) {
    let out = ctx.arg(1);
    let buf = st.reserve_vertex_uniform_buffer(ctx);
    ctx.write_u32(out, buf);
    ctx.ret(0);
}

/// int sceGxmReserveFragmentDefaultUniformBuffer(context, void **uniformBuffer)
/// Hand back a real guest buffer for the fragment stage's default uniforms so a guest
/// read-back is faithful. The software/GPU renderers reproduce a draw from the vertex
/// transform + textures and do not consume fragment default uniforms, so this buffer
/// is allocated but not bound as a capture source.
pub(super) fn reserve_fragment_uniforms(ctx: &mut GuestCtx, st: &mut VitaState) {
    let out = ctx.arg(1);
    let buf = st.reserve_fragment_uniform_buffer(ctx);
    ctx.write_u32(out, buf);
    ctx.ret(0);
}

/// int sceGxmSetUniformDataF(void *uniformBuffer, const SceGxmProgramParameter
///     *parameter, unsigned int componentOffset, unsigned int componentCount,
///     const float *sourceData)  -- 5 args (1 on the stack).
pub(super) fn set_uniform_data_f(ctx: &mut GuestCtx, st: &mut VitaState) {
    let uniform_buffer = ctx.arg(0);
    let component_offset = ctx.arg(2);
    let component_count = ctx.arg(3);
    let source = ctx.arg(4);

    let mut values = Vec::with_capacity(component_count as usize);
    for i in 0..component_count {
        values.push(ctx.read_f32(source + i * 4));
    }
    // Faithful copy into the reserved buffer (in case the guest reads it back).
    for (i, v) in values.iter().enumerate() {
        ctx.write_u32(uniform_buffer + (component_offset + i as u32) * 4, v.to_bits());
    }
    tracing::trace!(
        target: "vitaslop::gxm",
        buffer = format_args!("{uniform_buffer:#x}"),
        component_offset,
        component_count,
        "setUniformDataF"
    );
    st.set_uniforms(values);
    ctx.ret(0);
}

/// void sceGxmSetVertexStream(context, unsigned int streamIndex, const void *data)
pub(super) fn set_vertex_stream(ctx: &mut GuestCtx, st: &mut VitaState) {
    let stream_index = ctx.arg(1);
    let data = ctx.arg(2);
    st.bind_stream(stream_index, data);
    ctx.ret(0);
}

/// int sceGxmDraw(context, SceGxmPrimitiveType primitive, SceGxmIndexFormat
///     indexType, const void *indexData, unsigned int indexCount)  -- 5 args.
pub(super) fn draw(ctx: &mut GuestCtx, st: &mut VitaState) {
    let primitive = ctx.arg(1);
    let index_format = ctx.arg(2);
    let index_data = ctx.arg(3);
    let index_count = ctx.arg(4);
    st.record_draw(ctx, primitive, index_format, index_data, index_count);
    ctx.ret(0);
}

/// int sceGxmDrawInstanced(context, SceGxmPrimitiveType primitive, SceGxmIndexFormat
///     indexType, const void *indexData, unsigned int indexCount, unsigned int
///     indexWrap)  -- 6 args. Same draw as `sceGxmDraw` with hardware instancing: the
/// index buffer is replayed once per instance, incrementing the instance index every
/// `indexWrap` indices. We capture the base geometry (the `indexCount` index run) - the
/// per-instance transform is a vertex-program input the capture already carries, so the
/// first instance renders correctly; broader instancing can layer on later.
pub(super) fn draw_instanced(ctx: &mut GuestCtx, st: &mut VitaState) {
    let primitive = ctx.arg(1);
    let index_format = ctx.arg(2);
    let index_data = ctx.arg(3);
    let index_count = ctx.arg(4);
    let index_wrap = ctx.arg(5);
    // Only the first instance is captured, so record how many the guest asked for: a title
    // that instances its scenery would otherwise silently render one copy of it.
    tracing::debug!(
        target: "vitaslop::gxm",
        index_count, index_wrap,
        instances = if index_wrap > 0 { index_count / index_wrap } else { 1 },
        "drawInstanced"
    );
    st.record_draw(ctx, primitive, index_format, index_data, index_count);
    ctx.ret(0);
}

/// `SceGxmTextureType` 3-bit selector (the top 3 bits of the full type enum), as
/// stored in control word 1 bits 29..31.
pub(super) const TYPE_SWIZZLED: u32 = 0;
pub(super) const TYPE_CUBE: u32 = 2; // 0x4000_0000 >> 29
pub(super) const TYPE_LINEAR: u32 = 3; // 0x6000_0000 >> 29
pub(super) const TYPE_TILED: u32 = 4; // 0x8000_0000 >> 29
pub(super) const TYPE_SWIZZLED_ARBITRARY: u32 = 5; // 0xA000_0000 >> 29
pub(super) const TYPE_LINEAR_STRIDED: u32 = 6; // 0xC000_0000 >> 29

/// void sceGxmSetFragmentTexture(context, unsigned int textureIndex, const
///     SceGxmTexture *texture)
pub(super) fn set_fragment_texture(ctx: &mut GuestCtx, st: &mut VitaState) {
    let unit = ctx.arg(1);
    let texture = ctx.arg(2);
    st.bind_fragment_texture(unit, texture);
    ctx.ret(0);
}

/// The `sceGxmTextureInit*` family: (SceGxmTexture *texture, const void *data,
/// SceGxmTextureFormat texFormat, unsigned int width, unsigned int height,
/// unsigned int mipCount|byteStride). Writes the 16-byte control-word struct the
/// guest will later hand to sceGxmSetFragmentTexture, and records the exact format
/// so the channel swizzle survives. Layout per vitasdk `gxm.h struct SceGxmTexture`.
pub(super) fn texture_init(ctx: &mut GuestCtx, st: &mut VitaState, type_field: u32) {
    let texture = ctx.arg(0);
    let data = ctx.arg(1);
    let tex_format = ctx.arg(2);
    let width = ctx.arg(3);
    let height = ctx.arg(4);

    // The 6th argument is `mipCount` for every layout except LINEAR_STRIDED, where it
    // is the explicit byte stride. Record it so GetMipmapCountUnsafe / GetStride read
    // back the exact value the guest passed.
    let mip_or_stride = ctx.arg(5);
    if type_field == TYPE_LINEAR_STRIDED {
        st.set_texture_init_extra(texture, 1, mip_or_stride);
    } else {
        st.set_texture_init_extra(texture, mip_or_stride, 0);
    }

    let base_format = (tex_format >> 24) & 0x1f;
    let w1 = (height.saturating_sub(1) & 0xfff)
        | ((width.saturating_sub(1) & 0xfff) << 12)
        | (base_format << 24)
        | (type_field << 29);
    ctx.write_u32(texture, 0);
    ctx.write_u32(texture + 4, w1);
    ctx.write_u32(texture + 8, data & 0xffff_fffc);
    ctx.write_u32(texture + 12, 0);
    st.set_texture_format(texture, tex_format);
    ctx.ret(0);
}

/// int sceGxmTextureSetData(SceGxmTexture *texture, const void *data): rewrite the
/// data-address control word (word 2), preserving its low 2 lod bits.
pub(super) fn texture_set_data(ctx: &mut GuestCtx, _st: &mut VitaState) {
    let texture = ctx.arg(0);
    let data = ctx.arg(1);
    let lod = ctx.read_u32(texture + 8) & 0x3;
    ctx.write_u32(texture + 8, (data & 0xffff_fffc) | lod);
    ctx.ret(0);
}

/// int sceGxmTextureSetFormat(SceGxmTexture *texture, SceGxmTextureFormat fmt)
pub(super) fn texture_set_format(ctx: &mut GuestCtx, st: &mut VitaState) {
    let texture = ctx.arg(0);
    let fmt = ctx.arg(1);
    let base_format = (fmt >> 24) & 0x1f;
    let w1 = (ctx.read_u32(texture + 4) & !(0x1f << 24)) | (base_format << 24);
    ctx.write_u32(texture + 4, w1);
    st.set_texture_format(texture, fmt);
    ctx.ret(0);
}

/// void *sceGxmTextureGetData(const SceGxmTexture *texture)
pub(super) fn texture_get_data(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let data = ctx.read_u32(texture + 8) & 0xffff_fffc;
    ctx.ret(data);
}

/// sceGxmTextureGetWidth/Height: the 12-bit size-1 field at `shift` in word 1.
pub(super) fn texture_get_dim(ctx: &mut GuestCtx, shift: u32) {
    let texture = ctx.arg(0);
    let field = (ctx.read_u32(texture + 4) >> shift) & 0xfff;
    ctx.ret(field + 1);
}

/// SceGxmTextureFormat sceGxmTextureGetFormat(const SceGxmTexture *texture)
pub(super) fn texture_get_format(ctx: &mut GuestCtx, st: &mut VitaState) {
    let texture = ctx.arg(0);
    // Prefer the exact format we recorded; otherwise reconstruct the base format.
    let fmt = st
        .texture_format(texture)
        .unwrap_or_else(|| ((ctx.read_u32(texture + 4) >> 24) & 0x1f) << 24);
    ctx.ret(fmt);
}

/// int sceGxmDisplayQueueAddEntry(oldBuffer, newBuffer, const void *callbackData)
/// The callback data's first field is the display buffer address to present.
pub(super) fn display_queue_add_entry(ctx: &mut GuestCtx, st: &mut VitaState) {
    let callback_data = ctx.arg(2);
    let buffer = ctx.read_u32(callback_data);
    if buffer != 0 {
        st.present(buffer);
    }
    // Run the registered display callback as guest code (preemptive mode): the
    // game's own buffer bookkeeping lives there, and a double-buffered title
    // spins forever after two frames if it never runs.
    st.enqueue_display_callback(ctx, callback_data);
    ctx.ret(0);
}

// --- Fixed-function pipeline state setters ----------------------------------
//
// Each `sceGxmSet*` below mutates one field of the sticky GXM context state
// ([`crate::capture::RenderState`]); the current state is snapshotted into every
// draw at record time (see `VitaState::record_draw`). All return `void` on the
// Vita - the `-> i32` (0) here just parks a defined value in r0 the caller ignores.
// The first argument is the `SceGxmContext *` (a single implicit context here, so
// it is unused); the enum arguments are stored verbatim as their raw GXM words.

/// void sceGxmSetCullMode(SceGxmContext *context, SceGxmCullMode mode)
#[hostcall]
pub(super) fn set_cull_mode(st: &mut VitaState, _context: u32, mode: u32) -> i32 {
    st.render_state_mut().cull_mode = mode;
    0
}

/// void sceGxmSetTwoSidedEnable(SceGxmContext *context, SceGxmTwoSidedMode enable)
#[hostcall]
pub(super) fn set_two_sided_enable(st: &mut VitaState, _context: u32, enable: u32) -> i32 {
    st.render_state_mut().two_sided = enable;
    0
}

/// void sceGxmSetFrontDepthFunc(SceGxmContext *context, SceGxmDepthFunc depthFunc)
#[hostcall]
pub(super) fn set_front_depth_func(st: &mut VitaState, _context: u32, func: u32) -> i32 {
    st.render_state_mut().front_depth_func = func;
    0
}

/// void sceGxmSetBackDepthFunc(SceGxmContext *context, SceGxmDepthFunc depthFunc)
#[hostcall]
pub(super) fn set_back_depth_func(st: &mut VitaState, _context: u32, func: u32) -> i32 {
    st.render_state_mut().back_depth_func = func;
    0
}

/// void sceGxmSetFrontDepthWriteEnable(SceGxmContext *context, SceGxmDepthWriteMode enable)
#[hostcall]
pub(super) fn set_front_depth_write_enable(st: &mut VitaState, _context: u32, enable: u32) -> i32 {
    st.render_state_mut().front_depth_write = enable;
    0
}

/// void sceGxmSetFrontFragmentProgramEnable(SceGxmContext *context, SceGxmFragmentProgramMode enable)
#[hostcall]
pub(super) fn set_front_fragment_program_enable(st: &mut VitaState, _context: u32, enable: u32) -> i32 {
    st.render_state_mut().front_fragment_program_enable = enable;
    0
}

/// void sceGxmSetBackFragmentProgramEnable(SceGxmContext *context, SceGxmFragmentProgramMode enable)
#[hostcall]
pub(super) fn set_back_fragment_program_enable(st: &mut VitaState, _context: u32, enable: u32) -> i32 {
    st.render_state_mut().back_fragment_program_enable = enable;
    0
}

/// void sceGxmSetFrontPointLineWidth(SceGxmContext *context, unsigned int width)
#[hostcall]
pub(super) fn set_front_point_line_width(st: &mut VitaState, _context: u32, width: u32) -> i32 {
    st.render_state_mut().front_point_line_width = width;
    0
}

/// void sceGxmSetFrontPolygonMode(SceGxmContext *context, SceGxmPolygonMode mode)
#[hostcall]
pub(super) fn set_front_polygon_mode(st: &mut VitaState, _context: u32, mode: u32) -> i32 {
    st.render_state_mut().front_polygon_mode = mode;
    0
}

/// void sceGxmSetFrontStencilRef(SceGxmContext *context, unsigned int sref)
#[hostcall]
pub(super) fn set_front_stencil_ref(st: &mut VitaState, _context: u32, sref: u32) -> i32 {
    st.render_state_mut().front_stencil_ref = sref;
    0
}

/// void sceGxmSetFrontStencilFunc(SceGxmContext *context, SceGxmStencilFunc func,
///     SceGxmStencilOp stencilFail, SceGxmStencilOp depthFail, SceGxmStencilOp
///     depthPass, unsigned char compareMask, unsigned char writeMask)
#[hostcall]
pub(super) fn set_front_stencil_func(
    st: &mut VitaState,
    _context: u32,
    func: u32,
    stencil_fail: u32,
    depth_fail: u32,
    depth_pass: u32,
    compare_mask: u32,
    write_mask: u32,
) -> i32 {
    let rs = st.render_state_mut();
    rs.front_stencil_func = func;
    rs.front_stencil_op_fail = stencil_fail;
    rs.front_stencil_op_depth_fail = depth_fail;
    rs.front_stencil_op_depth_pass = depth_pass;
    rs.front_stencil_compare_mask = compare_mask & 0xff;
    rs.front_stencil_write_mask = write_mask & 0xff;
    0
}

/// void sceGxmSetViewport(SceGxmContext *context, float xOffset, float xScale,
///     float yOffset, float yScale, float zOffset, float zScale)
#[hostcall]
pub(super) fn set_viewport(
    st: &mut VitaState,
    _context: u32,
    x_offset: f32,
    x_scale: f32,
    y_offset: f32,
    y_scale: f32,
    z_offset: f32,
    z_scale: f32,
) -> i32 {
    st.render_state_mut().viewport = [x_offset, x_scale, y_offset, y_scale, z_offset, z_scale];
    0
}

/// void sceGxmSetViewportEnable(SceGxmContext *context, SceGxmViewportMode enable)
#[hostcall]
pub(super) fn set_viewport_enable(st: &mut VitaState, _context: u32, enable: u32) -> i32 {
    st.render_state_mut().viewport_enable = enable;
    0
}

/// void sceGxmSetRegionClip(SceGxmContext *context, SceGxmRegionClipMode mode,
///     unsigned int xMin, unsigned int yMin, unsigned int xMax, unsigned int yMax)
#[hostcall]
pub(super) fn set_region_clip(
    st: &mut VitaState,
    _context: u32,
    mode: u32,
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
) -> i32 {
    let rs = st.render_state_mut();
    rs.region_clip_mode = mode;
    rs.region_clip = [x_min, y_min, x_max, y_max];
    0
}

// --- Getters ----------------------------------------------------------------

/// SceGxmColorFormat sceGxmColorSurfaceGetFormat(const SceGxmColorSurface *surface)
/// Returns the exact format the guest set on this surface (recorded at
/// `sceGxmColorSurfaceInit`), or 0 if the surface was never initialized here.
#[hostcall]
pub(super) fn color_surface_get_format(ctx: &mut GuestCtx, st: &mut VitaState, surface: u32) -> u32 {
    resolve_color_surface(ctx, st, surface).map(|s| s.format).unwrap_or(0)
}

/// The live surface at `addr`: from the guest struct's own contents if it holds one
/// (so a COPY of an initialised surface still answers), else from the address table.
fn resolve_color_surface(ctx: &mut GuestCtx, st: &VitaState, addr: u32) -> Option<ColorSurface> {
    read_color_surface(ctx, addr).or_else(|| st.color_surface(addr))
}

/// SceGxmColorSurfaceType sceGxmColorSurfaceGetType(const SceGxmColorSurface *surface)
/// Returns the surface layout type (LINEAR/TILED/SWIZZLED) the guest set at
/// `sceGxmColorSurfaceInit`, or 0 (LINEAR, the enum default) if never initialized here.
#[hostcall]
pub(super) fn color_surface_get_type(ctx: &mut GuestCtx, st: &mut VitaState, surface: u32) -> u32 {
    resolve_color_surface(ctx, st, surface).map(|s| s.surface_type).unwrap_or(0)
}

/// void sceGxmColorSurfaceSetClip(SceGxmColorSurface *surface, unsigned int xMin,
///     unsigned int yMin, unsigned int xMax, unsigned int yMax)
/// The color-surface clip rectangle constrains where a scene writes. Our capture
/// records the surface geometry (not a sub-clip) and the renderer draws the whole
/// surface, so this is accepted with no state change; a title sets it and proceeds.
#[hostcall]
pub(super) fn color_surface_set_clip(_context: u32) -> i32 {
    0
}

/// SceGxmTextureType sceGxmTextureGetType(const SceGxmTexture *texture)
/// The 3-bit type selector lives in control word 1 bits 29..31; return it back in
/// the enum's high-bit position (e.g. LINEAR = 0x6000_0000).
#[hostcall]
pub(super) fn texture_get_type(ctx: &mut GuestCtx, _st: &mut VitaState, texture: Ptr) -> u32 {
    let w1 = ctx.read_u32(texture.addr().wrapping_add(4));
    ((w1 >> 29) & 0x7) << 29
}

/// The `semantic` u16 packed into a `SceGxmProgramParameter` at record offset +6
/// (following the +4 category/type/component/container word verified against the
/// real shaders in the image). It carries the `SceGxmParameterSemantic` in the LOW
/// byte and the semantic INDEX in the high byte - see [`param_get_semantic`] for the
/// measurement that pins that, and why the split matters.
fn param_semantic_word(ctx: &GuestCtx, param: u32) -> u32 {
    (ctx.read_u32(param.wrapping_add(4)) >> 16) & 0xffff
}

/// Byte mask of the `SceGxmParameterSemantic` within [`param_semantic_word`].
const GXM_SEMANTIC_MASK: u32 = 0xff;
/// Bit position of the semantic INDEX within [`param_semantic_word`].
const GXM_SEMANTIC_INDEX_SHIFT: u32 = 8;

/// SceGxmParameterSemantic sceGxmProgramParameterGetSemantic(const SceGxmProgramParameter *param)
///
/// # The split, and how it was pinned
/// A title that builds its `SceGxmVertexAttribute` array by matching shader semantics -
/// rather than by hard-coding offsets - gets ZERO attributes when this returns the wrong
/// field, and an empty attribute array is silent: `sceGxmShaderPatcherCreateVertexProgram`
/// accepts it, every draw then arrives with real geometry and no way to fetch it, and the
/// frame renders as a bare clear colour that looks exactly like a title drawing nothing.
///
/// So the split is measured, not assumed. Over the captured shader corpus of two retail
/// titles, the attribute parameters' raw words are: `position` 0x000b, `normal` 0x0009,
/// `VertexColour1` 0x0006, `Uv1` 0x000e, `Uv2` 0x010e, `rightVector` 0x020e, `upVector`
/// 0x030e, `tangent` 0x060e, `lightColour` 0x0106. The low byte matches
/// `SceGxmParameterSemantic` exactly (POSITION 11, NORMAL 9, COLOR 6, TEXCOORD 14 -
/// vitasdk `gxm.h`), and the high byte counts 0,1,2,3,6 across names that differ only by
/// their index. A 4/12 split would instead read those indices as 16/32/48/96, which no
/// enumerated semantic index takes.
#[hostcall]
pub(super) fn param_get_semantic(ctx: &mut GuestCtx, _st: &mut VitaState, param: Ptr) -> u32 {
    param_semantic_word(ctx, param.addr()) & GXM_SEMANTIC_MASK
}

/// unsigned int sceGxmProgramParameterGetSemanticIndex(const SceGxmProgramParameter *param)
#[hostcall]
pub(super) fn param_get_semantic_index(ctx: &mut GuestCtx, _st: &mut VitaState, param: Ptr) -> u32 {
    param_semantic_word(ctx, param.addr()) >> GXM_SEMANTIC_INDEX_SHIFT
}

// --- Sampler state ----------------------------------------------------------

/// int sceGxmTextureSetUAddrMode[Safe](SceGxmTexture *texture, SceGxmTextureAddrMode mode)
#[hostcall]
pub(super) fn texture_set_u_addr_mode(st: &mut VitaState, texture: u32, mode: u32) -> i32 {
    st.set_texture_sampler(texture, 0, mode);
    0
}

/// int sceGxmTextureSetVAddrMode[Safe](SceGxmTexture *texture, SceGxmTextureAddrMode mode)
#[hostcall]
pub(super) fn texture_set_v_addr_mode(st: &mut VitaState, texture: u32, mode: u32) -> i32 {
    st.set_texture_sampler(texture, 1, mode);
    0
}

/// int sceGxmTextureSetLodBias(SceGxmTexture *texture, unsigned int bias)
#[hostcall]
pub(super) fn texture_set_lod_bias(st: &mut VitaState, texture: u32, bias: u32) -> i32 {
    st.set_texture_sampler(texture, 2, bias);
    0
}

/// int sceGxmTextureSetMinFilter(SceGxmTexture *texture, SceGxmTextureFilter minFilter)
#[hostcall]
pub(super) fn texture_set_min_filter(st: &mut VitaState, texture: u32, filter: u32) -> i32 {
    st.set_texture_filter(texture, 0, filter);
    0
}

/// int sceGxmTextureSetMagFilter(SceGxmTexture *texture, SceGxmTextureFilter magFilter)
#[hostcall]
pub(super) fn texture_set_mag_filter(st: &mut VitaState, texture: u32, filter: u32) -> i32 {
    st.set_texture_filter(texture, 1, filter);
    0
}

/// int sceGxmTextureSetMipFilter(SceGxmTexture *texture, SceGxmTextureMipFilter mipFilter)
#[hostcall]
pub(super) fn texture_set_mip_filter(st: &mut VitaState, texture: u32, filter: u32) -> i32 {
    st.set_texture_filter(texture, 2, filter);
    0
}

/// int sceGxmTextureSetGammaMode(SceGxmTexture *texture, SceGxmTextureGammaMode gammaMode)
#[hostcall]
pub(super) fn texture_set_gamma_mode(st: &mut VitaState, texture: u32, gamma: u32) -> i32 {
    st.set_texture_gamma(texture, gamma);
    0
}

/// void sceGxmSetVertexUniformBuffer / sceGxmSetFragmentUniformBuffer
///     (SceGxmContext *context, unsigned int bufferIndex, const void *bufferData)
///
/// Binds a NON-default uniform buffer for the given stage.
///
/// The capture carries only the DEFAULT uniform buffer (the SA bank) with each draw, so
/// this buffer's CONTENTS do not reach the scene: a recompiled shader that reads uniform
/// buffer `bufferIndex` reads nothing. That is a real gap rather than a no-op, so it says
/// so once - an unreported approximation is indistinguishable on screen from a faithful
/// render, which is exactly how a wrong "it renders correctly" claim gets made. Once per
/// run rather than per call, because it is a property of the title's shaders, not an event.
pub(super) fn set_uniform_buffer(ctx: &mut GuestCtx, stage: &'static str) {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let index = ctx.arg(1);
    let data = ctx.arg(2);
    if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            target: "vitaslop::gxm",
            stage,
            index,
            data = format_args!("{data:#x}"),
            "a non-default uniform buffer was bound; the capture records only the DEFAULT \
             uniform buffer, so a recompiled shader reading this buffer index gets nothing"
        );
    }
    ctx.ret(0);
}

// --- Texture getters (read back the sticky sampler/format state) -------------

/// unsigned int sceGxmTextureGetMipmapCountUnsafe(const SceGxmTexture *texture)
#[hostcall]
pub(super) fn texture_get_mipmap_count(st: &mut VitaState, texture: u32) -> u32 {
    st.texture_mip_count(texture)
}

/// unsigned int sceGxmTextureGetLodBias(const SceGxmTexture *texture)
#[hostcall]
pub(super) fn texture_get_lod_bias(st: &mut VitaState, texture: u32) -> u32 {
    st.texture_lod_bias(texture)
}

/// SceGxmTextureAddrMode sceGxmTextureGetUAddrModeSafe(const SceGxmTexture *texture)
#[hostcall]
pub(super) fn texture_get_u_addr_mode(st: &mut VitaState, texture: u32) -> u32 {
    st.texture_addr_mode(texture, 0)
}

/// SceGxmTextureAddrMode sceGxmTextureGetVAddrModeSafe(const SceGxmTexture *texture)
#[hostcall]
pub(super) fn texture_get_v_addr_mode(st: &mut VitaState, texture: u32) -> u32 {
    st.texture_addr_mode(texture, 1)
}

/// SceGxmTextureFilter sceGxmTextureGetMinFilter(const SceGxmTexture *texture)
#[hostcall]
pub(super) fn texture_get_min_filter(st: &mut VitaState, texture: u32) -> u32 {
    st.texture_filter(texture, 0)
}

/// SceGxmTextureFilter sceGxmTextureGetMagFilter(const SceGxmTexture *texture)
#[hostcall]
pub(super) fn texture_get_mag_filter(st: &mut VitaState, texture: u32) -> u32 {
    st.texture_filter(texture, 1)
}

/// SceGxmTextureGammaMode sceGxmTextureGetGammaMode(const SceGxmTexture *texture)
#[hostcall]
pub(super) fn texture_get_gamma_mode(st: &mut VitaState, texture: u32) -> u32 {
    st.texture_filter(texture, 2)
}

/// unsigned int sceGxmTextureGetStride(const SceGxmTexture *texture)
pub(super) fn texture_get_stride(ctx: &mut GuestCtx, st: &mut VitaState) {
    let texture = ctx.arg(0);
    let stride = st.texture_stride(ctx, texture);
    ctx.ret(stride);
}

// --- Color surface getters/setters beyond format ----------------------------

/// void *sceGxmColorSurfaceGetData(const SceGxmColorSurface *surface)
#[hostcall]
pub(super) fn color_surface_get_data(ctx: &mut GuestCtx, st: &mut VitaState, surface: u32) -> u32 {
    resolve_color_surface(ctx, st, surface).map(|s| s.data_addr).unwrap_or(0)
}

/// unsigned int sceGxmColorSurfaceGetStrideInPixels(const SceGxmColorSurface *surface)
#[hostcall]
pub(super) fn color_surface_get_stride_in_pixels(ctx: &mut GuestCtx, st: &mut VitaState, surface: u32) -> u32 {
    resolve_color_surface(ctx, st, surface).map(|s| s.stride_pixels).unwrap_or(0)
}

/// int sceGxmColorSurfaceSetGammaMode(SceGxmColorSurface *surface, SceGxmColorSurfaceGammaMode gammaMode)
#[hostcall]
pub(super) fn color_surface_set_gamma_mode(st: &mut VitaState, surface: u32, gamma: u32) -> i32 {
    st.set_color_surface_gamma(surface, gamma);
    0
}

// --- Render-target sizing + GPU notification region -------------------------

/// int sceGxmGetRenderTargetMemSize(const SceGxmRenderTargetParams *params,
///     unsigned int *driverMemSize)
/// We emulate no GPU render-target control structures, but a title reads this size to
/// allocate the `driverMemBlock` it hands to `sceGxmCreateRenderTarget`. Return a
/// page-aligned size proportionate to the render-target dimensions (`width` u16 at
/// +4, `height` u16 at +6), so the guest's allocation is plausible and never zero.
/// The block is opaque to us. This is a deliberate proxy, not the driver's exact
/// formula (which is proprietary); the returned block is never interpreted here.
pub(super) fn get_render_target_mem_size(ctx: &mut GuestCtx, _st: &mut VitaState) {
    let params = ctx.arg(0);
    let out = ctx.arg(1);
    let wh = ctx.read_u32(params + 4);
    let width = wh & 0xffff;
    let height = (wh >> 16) & 0xffff;
    // ~one control word per 8x8 tile plus a fixed header, page-aligned.
    let tiles = (width.div_ceil(8)) * (height.div_ceil(8));
    let size = (0x1000 + tiles * 16 + 0xfff) & !0xfff;
    ctx.write_u32(out, size);
    ctx.ret(0);
}

/// volatile unsigned int *sceGxmGetNotificationRegion(void)
pub(super) fn get_notification_region(ctx: &mut GuestCtx, st: &mut VitaState) {
    let region = st.notification_region();
    ctx.ret(region);
}

// --- Program reflection: default uniform buffer size + pass type ------------

/// unsigned int sceGxmProgramGetDefaultUniformBufferSize(const SceGxmProgram *program)
///
/// The size is the container's `default_uniform_buffer_count` (header +0x64) - a count of
/// 32-bit SA registers - times four. This is NOT bookkeeping we can ignore: a title uses the
/// answer as the LENGTH of the block it memcpys into the buffer
/// `sceGxmReserveVertexDefaultUniformBuffer` handed it, so under-reporting truncates the
/// title's own uniform upload and the shader then reads zeros for everything past the cut.
///
/// This previously read +0x2C, which is the varyings-block offset, not a size: it is a fixed
/// 108 on every program of a title regardless of that program's real uniform block (verified
/// across the whole captured corpus, vertex and fragment). A title given 108 writes exactly 27
/// floats into a buffer whose shader declares up to 74 registers, so its shared camera, light
/// and exposure uniforms never arrive - which is precisely what was observed.
pub(super) fn program_get_default_uniform_buffer_size(ctx: &mut GuestCtx, _st: &mut VitaState) {
    let program = ctx.arg(0);
    let size = crate::host::default_uniform_buffer_bytes(ctx, program);
    ctx.ret(size);
}

/// SceGxmPassType sceGxmFragmentProgramGetPassType(const SceGxmFragmentProgram *fp)
/// Fragment programs are opaque handles here (no struct is emulated), so we cannot
/// reflect a stored pass type; return SCE_GXM_PASS_TYPE_OPAQUE (0), the pass type of
/// the standard opaque/blended fragment programs a title builds. Revisit if a title
/// is found to branch on a non-opaque pass type.
#[hostcall]
pub(super) fn fragment_program_get_pass_type(_fragment_program: u32) -> u32 {
    0
}

// --- Precomputed draw family ------------------------------------------------

/// unsigned int sceGxmGetPrecomputedDrawSize(const SceGxmVertexProgram *vertexProgram)
/// The guest allocates a memblock of this size for a `SceGxmPrecomputedDraw`. The
/// public struct is `SCE_GXM_PRECOMPUTED_DRAW_WORD_COUNT` (11) u32 words = 44 bytes.
#[hostcall]
pub(super) fn get_precomputed_draw_size(_vertex_program: u32) -> u32 {
    11 * 4
}

/// int sceGxmPrecomputedDrawInit(SceGxmPrecomputedDraw *precomputedDraw,
///     const SceGxmVertexProgram *vertexProgram, void *memBlock)
#[hostcall]
pub(super) fn precomputed_draw_init(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    precomputed: u32,
    vertex_program: u32,
    _mem_block: u32,
) -> i32 {
    st.precomputed_draw_init(ctx, precomputed, vertex_program);
    0
}

/// int sceGxmPrecomputedDrawSetVertexStream(SceGxmPrecomputedDraw *precomputedDraw,
///     unsigned int streamIndex, const void *streamData)
#[hostcall]
pub(super) fn precomputed_draw_set_vertex_stream(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    precomputed: u32,
    stream_index: u32,
    stream_data: u32,
) -> i32 {
    st.precomputed_draw_set_stream(ctx, precomputed, stream_index, stream_data);
    0
}

/// void sceGxmPrecomputedDrawSetParams(SceGxmPrecomputedDraw *precomputedDraw,
///     SceGxmPrimitiveType primType, SceGxmIndexFormat indexType, const void
///     *indexData, unsigned int indexCount)
#[hostcall]
pub(super) fn precomputed_draw_set_params(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    precomputed: u32,
    prim_type: u32,
    index_type: u32,
    index_data: u32,
    index_count: u32,
) -> i32 {
    st.precomputed_draw_set_params(ctx, precomputed, prim_type, index_type, index_data, index_count);
    0
}

/// int sceGxmDrawPrecomputed(SceGxmContext *context, const SceGxmPrecomputedDraw
///     *precomputedDraw): replay the bundled draw into the current scene.
pub(super) fn draw_precomputed(ctx: &mut GuestCtx, st: &mut VitaState) {
    let precomputed = ctx.arg(1);
    st.draw_precomputed(ctx, precomputed);
    ctx.ret(0);
}

// --- Precomputed vertex/fragment state family -------------------------------
//
// A precomputed state bundles one shader stage's default uniform buffer + textures
// into a guest struct the game builds once and binds per draw (this title draws
// almost entirely through this path - `sceGxmSetUniformDataF` is never called). The
// struct is opaque, so we key the recorded state by its guest address, mirroring the
// precomputed-draw family. `Init`/`SetDefaultUniformBuffer`/`SetTexture` record the
// bundle; `sceGxmSetPrecomputed{Vertex,Fragment}State` applies it to the live bind
// state so `record_draw` snapshots the same uniforms and textures the direct path would.

/// unsigned int sceGxmGetPrecomputedVertexStateSize(const SceGxmVertexProgram *program)
/// The size the guest allocates for the state's memBlock. The public struct is
/// SCE_GXM_PRECOMPUTED_VERTEX_STATE_WORD_COUNT (7) u32 words = 0x1C bytes; the state
/// data lives in our side table, so the guest's block is bookkeeping we do not consume.
#[hostcall]
pub(super) fn get_precomputed_vertex_state_size(_program: u32) -> u32 {
    7 * 4
}

/// unsigned int sceGxmGetPrecomputedFragmentStateSize(const SceGxmFragmentProgram *program)
/// As above; the fragment state is SCE_GXM_PRECOMPUTED_FRAGMENT_STATE_WORD_COUNT (9)
/// u32 words = 0x24 bytes.
#[hostcall]
pub(super) fn get_precomputed_fragment_state_size(_program: u32) -> u32 {
    9 * 4
}

/// int sceGxmPrecomputedVertexStateInit(SceGxmPrecomputedVertexState *state,
///     const SceGxmVertexProgram *vertexProgram, void *memBlock)
#[hostcall]
pub(super) fn precomputed_vertex_state_init(st: &mut VitaState, state: u32, vertex_program: u32, _mem_block: u32) -> i32 {
    st.precomputed_vertex_state_init(state, vertex_program);
    0
}

/// int sceGxmPrecomputedFragmentStateInit(SceGxmPrecomputedFragmentState *state,
///     const SceGxmFragmentProgram *fragmentProgram, void *memBlock)
#[hostcall]
pub(super) fn precomputed_fragment_state_init(st: &mut VitaState, state: u32, fragment_program: u32, _mem_block: u32) -> i32 {
    st.precomputed_fragment_state_init(state, fragment_program);
    0
}

/// void sceGxmPrecomputedVertexStateSetDefaultUniformBuffer(state, void *defaultBuffer)
#[hostcall]
pub(super) fn precomputed_vertex_state_set_default_uniform_buffer(st: &mut VitaState, state: u32, buffer: u32) -> i32 {
    st.precomputed_vertex_state_set_uniform_buffer(state, buffer);
    0
}

/// void sceGxmPrecomputedFragmentStateSetDefaultUniformBuffer(state, void *defaultBuffer)
#[hostcall]
pub(super) fn precomputed_fragment_state_set_default_uniform_buffer(st: &mut VitaState, state: u32, buffer: u32) -> i32 {
    st.precomputed_fragment_state_set_uniform_buffer(state, buffer);
    0
}

/// void *sceGxmPrecomputedVertexStateGetDefaultUniformBuffer(const ...State *state)
#[hostcall]
pub(super) fn precomputed_vertex_state_get_default_uniform_buffer(st: &mut VitaState, state: u32) -> u32 {
    st.precomputed_vertex_state_uniform_buffer(state)
}

/// void *sceGxmPrecomputedFragmentStateGetDefaultUniformBuffer(const ...State *state)
#[hostcall]
pub(super) fn precomputed_fragment_state_get_default_uniform_buffer(st: &mut VitaState, state: u32) -> u32 {
    st.precomputed_fragment_state_uniform_buffer(state)
}

/// int sceGxmPrecomputedVertexStateSetTexture(state, unsigned int textureIndex,
///     const SceGxmTexture *texture)
#[hostcall]
pub(super) fn precomputed_vertex_state_set_texture(st: &mut VitaState, state: u32, index: u32, texture: u32) -> i32 {
    st.precomputed_vertex_state_set_texture(state, index, texture);
    0
}

/// int sceGxmPrecomputedFragmentStateSetTexture(state, unsigned int textureIndex,
///     const SceGxmTexture *texture)
#[hostcall]
pub(super) fn precomputed_fragment_state_set_texture(st: &mut VitaState, state: u32, index: u32, texture: u32) -> i32 {
    st.precomputed_fragment_state_set_texture(state, index, texture);
    0
}

/// void sceGxmSetPrecomputedVertexState(SceGxmContext *context,
///     const SceGxmPrecomputedVertexState *precomputedState)
#[hostcall]
pub(super) fn set_precomputed_vertex_state(ctx: &mut GuestCtx, st: &mut VitaState, _context: u32, state: u32) -> i32 {
    st.bind_precomputed_vertex_state(ctx, state);
    0
}

/// void sceGxmSetPrecomputedFragmentState(SceGxmContext *context,
///     const SceGxmPrecomputedFragmentState *precomputedState)
#[hostcall]
pub(super) fn set_precomputed_fragment_state(ctx: &mut GuestCtx, st: &mut VitaState, _context: u32, state: u32) -> i32 {
    st.bind_precomputed_fragment_state(ctx, state);
    0
}

#[cfg(test)]
pub(crate) mod inline_op_tests {
    use super::*;
    use crate::host::VitaState;
    use crate::nid::gxm as g;
    use crate::world::DeterministicWorld;
    use crate::{SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::REG_COUNT;

    /// Guest address the synthetic `SceGxmProgramParameter` sits at. Word-aligned and
    /// away from 0 so a bug that reads the wrong base shows up as a mismatch.
    const PARAM: u32 = 0x40;

    /// The parameter record: name offset, the packed word, array size, resource index.
    /// The packed word's nibbles are all DIFFERENT so a getter that reads the wrong
    /// field, or shifts by the wrong amount, cannot accidentally agree.
    fn param_record() -> [u32; 4] {
        [0x1234_5678, 0x0000_9CB6, 0x0000_002A, 0x0000_0007]
    }

    /// Run a NID through the real dispatch over a synthetic parameter record and
    /// return the r0 the guest would see.
    fn handler_result(func_nid: u32) -> u32 {
        handler_result_over(func_nid, param_record())
    }

    /// As [`handler_result`], over a caller-supplied record.
    pub(crate) fn handler_result_over(func_nid: u32, record: [u32; 4]) -> u32 {
        let mut regs = [0u32; REG_COUNT];
        regs[0] = PARAM;
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 4096];
        for (i, w) in record.iter().enumerate() {
            let off = PARAM as usize + i * 4;
            bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }
        let mut st = VitaState::new(0, 4096, Box::new(DeterministicWorld::default()));
        let mut mem = SliceMemory(&mut bytes);
        let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        super::super::dispatch(crate::nid::lib::SCE_GXM, func_nid, &mut ctx, &mut st);
        regs[0]
    }

    /// The word an inline op reads, out of the same synthetic record.
    fn word_at(offset: u32) -> u32 {
        param_record()[(offset / 4) as usize]
    }

    /// EVERY NID with an inline form must compute exactly what its host handler
    /// computes. The inline form is a SECOND implementation of the same call - the
    /// transpiler emits it into guest code and the host handler never runs - so
    /// nothing else in the system would notice them drifting apart: the render would
    /// simply be wrong, in a way that looks like a shader bug.
    ///
    /// Every NID this project inlines is listed here explicitly rather than iterated
    /// from the table, so adding an entry to `inline_op` without a case here leaves
    /// `every_inlined_nid_is_covered` failing.
    #[test]
    fn inline_ops_match_their_handlers() {
        for func_nid in COVERED {
            let op = inline_op(func_nid).expect("listed NID has an inline form");
            assert_eq!(
                op.eval(word_at(op.offset())),
                handler_result(func_nid),
                "inline form of {} disagrees with its handler",
                crate::nid::name(func_nid)
            );
        }
    }

    /// The NIDs the test above checks.
    const COVERED: [u32; 6] = [
        g::PROGRAM_PARAMETER_GET_CATEGORY,
        g::PROGRAM_PARAMETER_GET_TYPE,
        g::PROGRAM_PARAMETER_GET_COMPONENT_COUNT,
        g::PROGRAM_PARAMETER_GET_CONTAINER_INDEX,
        g::PROGRAM_PARAMETER_GET_ARRAY_SIZE,
        g::PROGRAM_PARAMETER_GET_RESOURCE_INDEX,
    ];

    /// Only a call with NO host behaviour may be inlined. Inlining is invisible to
    /// the host - the handler simply never runs - so inlining a NID that touches
    /// `VitaState` would silently drop that state change, and the symptom would appear
    /// somewhere else entirely.
    ///
    /// The two directions are checked separately: every NID this test claims to cover
    /// really does have an inline form (so `COVERED` cannot silently go stale against
    /// `inline_op`), and a sample of NIDs with real behaviour do not.
    #[test]
    fn only_pure_getters_are_inlined() {
        for &nid in &COVERED {
            assert!(
                inline_op(nid).is_some(),
                "{} is listed as covered but has no inline form",
                crate::nid::name(nid)
            );
        }
        for nid in [
            g::DRAW,                          // records a whole draw into the scene
            g::END_SCENE,                     // completes and folds a frame
            g::SET_VERTEX_PROGRAM,            // updates the bound program
            g::PROGRAM_PARAMETER_GET_NAME,    // returns a pointer, not a bitfield
            g::PROGRAM_GET_PARAMETER,         // computes an address from two reads
            g::PROGRAM_GET_PARAMETER_COUNT,   // reads the PROGRAM, not a parameter
        ] {
            assert!(
                inline_op(nid).is_none(),
                "{} does more than one bitfield read and must not be inlined",
                crate::nid::name(nid)
            );
        }
    }
}

#[cfg(test)]
mod semantic_tests {
    //! How `sceGxmProgramParameterGetSemantic` / `GetSemanticIndex` split the `semantic`
    //! u16 in a `SceGxmProgramParameter`.
    //!
    //! Content-free: the words below are the encoding, not game data. They are the values
    //! measured over a captured shader corpus (see the doc comment on
    //! [`super::param_get_semantic`]), reproduced here as the specification this code is
    //! held to. A title that builds its vertex-attribute array by matching semantics
    //! renders NOTHING when this split is wrong, and does so silently, so the encoding is
    //! pinned by a test rather than left to a comment.
    use super::inline_op_tests::handler_result_over;
    use crate::nid::gxm as g;

    /// A parameter record carrying `semantic_word` in the upper half of the +4 word.
    /// The lower half is a plausible attribute descriptor (category 0 = ATTRIBUTE,
    /// float, 4 components) so nothing about the record is degenerate.
    fn record(semantic_word: u16) -> [u32; 4] {
        [0x1234_5678, ((semantic_word as u32) << 16) | 0x0400, 1, 0]
    }

    /// vitasdk `gxm.h` `SceGxmParameterSemantic` ordinals.
    const COLOR: u32 = 6;
    const NORMAL: u32 = 9;
    const POSITION: u32 = 11;
    const TEXCOORD: u32 = 14;

    #[test]
    fn semantic_is_the_low_byte_and_the_index_is_the_high_byte() {
        // (raw word, semantic, index) - each row an attribute name observed in the corpus:
        // position, normal, VertexColour1, Uv1, Uv2, rightVector, upVector, tangent.
        for (word, semantic, index) in [
            (0x000bu16, POSITION, 0),
            (0x0009, NORMAL, 0),
            (0x0006, COLOR, 0),
            (0x000e, TEXCOORD, 0),
            (0x010e, TEXCOORD, 1),
            (0x020e, TEXCOORD, 2),
            (0x030e, TEXCOORD, 3),
            (0x060e, TEXCOORD, 6),
            (0x0106, COLOR, 1),
        ] {
            assert_eq!(
                handler_result_over(g::PROGRAM_PARAMETER_GET_SEMANTIC, record(word)),
                semantic,
                "semantic of {word:#06x}"
            );
            assert_eq!(
                handler_result_over(g::PROGRAM_PARAMETER_GET_SEMANTIC_INDEX, record(word)),
                index,
                "semantic index of {word:#06x}"
            );
        }
    }

    /// The two fields must not overlap: reading the whole u16 as either one is the exact
    /// mistake this encoding invites, and it is invisible for index-0 attributes (the
    /// common case) while breaking every indexed one.
    #[test]
    fn the_two_fields_are_disjoint() {
        let word = 0x060e;
        let semantic = handler_result_over(g::PROGRAM_PARAMETER_GET_SEMANTIC, record(word));
        let index = handler_result_over(g::PROGRAM_PARAMETER_GET_SEMANTIC_INDEX, record(word));
        assert_eq!(semantic | (index << 8), word as u32, "the two fields reassemble the word");
    }
}
