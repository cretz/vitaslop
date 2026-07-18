//! SceGxm: the graphics API. These handlers hand back opaque handles for GXM
//! objects, remember the surfaces and vertex-program layouts the guest sets up,
//! and record the per-frame draw stream (BeginScene to EndScene) with the vertex,
//! index, and uniform data snapshotted from guest memory. No GPU is emulated and
//! no pixel is drawn here; that is the renderer's job over this capture.

use crate::capture::{ColorSurface, VertexAttribute};
use crate::host::{GuestCtx, VitaState};
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

/// int sceGxmColorSurfaceInit(surface, format, type, scale, outputRegisterSize,
///     width, height, strideInPixels, data)  -- 9 args (5 on the stack).
pub(super) fn color_surface_init(ctx: &mut GuestCtx, st: &mut VitaState) {
    let surface = ctx.arg(0);
    let format = ctx.arg(1);
    let width = ctx.arg(5);
    let height = ctx.arg(6);
    let stride_pixels = ctx.arg(7);
    let data_addr = ctx.arg(8);
    st.set_color_surface(
        surface,
        ColorSurface { format, width, height, stride_pixels, data_addr },
    );
    ctx.ret(0);
}

/// int sceGxmShaderPatcherCreateVertexProgram(patcher, programId, attributes,
///     attributeCount, streams, streamCount, vertexProgram)  -- 7 args.
pub(super) fn create_vertex_program(ctx: &mut GuestCtx, st: &mut VitaState) {
    let attributes_addr = ctx.arg(2);
    let attribute_count = ctx.arg(3);
    let streams_addr = ctx.arg(4);
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
    // SceGxmVertexStream: u16 stride at offset 0 of streams[0].
    let stride = (ctx.read_u32(streams_addr) & 0xFFFF) as u32;

    tracing::debug!(
        target: "vitaslop::gxm",
        attribute_count, stride, attrs = attributes.len(),
        attrs_addr = format_args!("{attributes_addr:#x}"),
        streams_addr = format_args!("{streams_addr:#x}"),
        "createVertexProgram"
    );

    let handle = st.new_handle();
    st.set_vertex_program(handle, attributes, stride);
    ctx.write_u32(out, handle);
    ctx.ret(0);
}

/// int sceGxmBeginScene(context, flags, renderTarget, validRegion,
///     vertexSyncObject, fragmentSyncObject, colorSurface, depthStencil) -- 8 args.
pub(super) fn begin_scene(ctx: &mut GuestCtx, st: &mut VitaState) {
    let color_surface = ctx.arg(6);
    st.begin_scene(color_surface);
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

/// int sceGxmReserveVertexDefaultUniformBuffer(context, void **uniformBuffer)
/// Hand back a small real guest buffer so a guest read-back would be faithful;
/// the uniform values are captured from the sceGxmSetUniformDataF source anyway.
pub(super) fn reserve_uniforms(ctx: &mut GuestCtx, st: &mut VitaState) {
    let out = ctx.arg(1);
    let buf = st.galloc(256, 16);
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
    st.set_uniforms(values);
    ctx.ret(0);
}

/// void sceGxmSetVertexStream(context, unsigned int streamIndex, const void *data)
pub(super) fn set_vertex_stream(ctx: &mut GuestCtx, st: &mut VitaState) {
    let stream_index = ctx.arg(1);
    let data = ctx.arg(2);
    if stream_index == 0 {
        st.bind_stream0(data);
    }
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
pub(super) fn color_surface_get_format(st: &mut VitaState, surface: u32) -> u32 {
    st.color_surface(surface).map(|s| s.format).unwrap_or(0)
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
/// real shaders in the image). Per the public gxp layout the field carries both the
/// `SceGxmParameterSemantic` (high 4 bits) and its index (low 12 bits), e.g.
/// TEXCOORD3 -> semantic TEXCOORD, index 3. The exact bit-split is the documented
/// encoding but not yet cross-checked against a decoded attribute here; it feeds the
/// render frontier (attribute->semantic mapping), not the boot path, so a title that
/// only reflects on it during setup is unaffected either way. Validate the split when
/// the renderer consumes semantics.
fn param_semantic_word(ctx: &GuestCtx, param: u32) -> u32 {
    (ctx.read_u32(param.wrapping_add(4)) >> 16) & 0xffff
}

/// SceGxmParameterSemantic sceGxmProgramParameterGetSemantic(const SceGxmProgramParameter *param)
#[hostcall]
pub(super) fn param_get_semantic(ctx: &mut GuestCtx, _st: &mut VitaState, param: Ptr) -> u32 {
    (param_semantic_word(ctx, param.addr()) >> 12) & 0xf
}

/// unsigned int sceGxmProgramParameterGetSemanticIndex(const SceGxmProgramParameter *param)
#[hostcall]
pub(super) fn param_get_semantic_index(ctx: &mut GuestCtx, _st: &mut VitaState, param: Ptr) -> u32 {
    param_semantic_word(ctx, param.addr()) & 0xfff
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
