//! SceGxm: the graphics API. These handlers hand back opaque handles for GXM
//! objects, remember the surfaces and vertex-program layouts the guest sets up,
//! and record the per-frame draw stream (BeginScene to EndScene) with the vertex,
//! index, and uniform data snapshotted from guest memory. No GPU is emulated and
//! no pixel is drawn here; that is the renderer's job over this capture.

use crate::capture::{ColorSurface, VertexAttribute};
use crate::host::{GuestCtx, VitaState};
use crate::nid::gxm as nid;
use crate::SvcOutcome;

/// SceGxmInitializeParams: displayQueueCallback at offset 8, its data size at 12.
const INIT_CB_OFFSET: u32 = 8;
const INIT_CB_DATA_SIZE_OFFSET: u32 = 12;

pub fn try_dispatch(func_nid: u32, ctx: &mut GuestCtx, st: &mut VitaState) -> Option<SvcOutcome> {
    match func_nid {
        nid::INITIALIZE => initialize(ctx, st),
        nid::MAP_MEMORY | nid::FINISH | nid::PAD_HEARTBEAT | nid::DISPLAY_QUEUE_FINISH
        | nid::PROGRAM_CHECK | nid::DESTROY_CONTEXT
        | nid::DESTROY_RENDER_TARGET | nid::SHADER_PATCHER_DESTROY
        | nid::SHADER_PATCHER_UNREGISTER_PROGRAM | nid::SHADER_PATCHER_RELEASE_VERTEX_PROGRAM
        | nid::SHADER_PATCHER_RELEASE_FRAGMENT_PROGRAM | nid::DEPTH_STENCIL_SURFACE_INIT
        | nid::SET_FRAGMENT_PROGRAM => ok(ctx),
        nid::TERMINATE => {
            ctx.ret(0);
            if st.halt_on_terminate {
                return Some(SvcOutcome::Halt);
            }
        }
        nid::MAP_VERTEX_USSE_MEMORY | nid::MAP_FRAGMENT_USSE_MEMORY => map_usse(ctx),
        nid::CREATE_CONTEXT | nid::CREATE_RENDER_TARGET | nid::SHADER_PATCHER_CREATE => {
            out_handle(ctx, st, 1)
        }
        nid::SYNC_OBJECT_CREATE => out_handle(ctx, st, 0),
        nid::SHADER_PATCHER_REGISTER_PROGRAM => out_handle(ctx, st, 2),
        nid::PROGRAM_FIND_PARAMETER_BY_NAME => find_parameter(ctx, st),
        nid::COLOR_SURFACE_INIT => color_surface_init(ctx, st),
        nid::SHADER_PATCHER_CREATE_VERTEX_PROGRAM => create_vertex_program(ctx, st),
        nid::SHADER_PATCHER_CREATE_FRAGMENT_PROGRAM => out_handle(ctx, st, 6),
        nid::BEGIN_SCENE => begin_scene(ctx, st),
        nid::END_SCENE => end_scene(ctx, st),
        nid::SET_VERTEX_PROGRAM => set_vertex_program(ctx, st),
        nid::RESERVE_VERTEX_DEFAULT_UNIFORM_BUFFER => reserve_uniforms(ctx, st),
        nid::SET_UNIFORM_DATA_F => set_uniform_data_f(ctx, st),
        nid::SET_VERTEX_STREAM => set_vertex_stream(ctx, st),
        nid::DRAW => draw(ctx, st),
        nid::DISPLAY_QUEUE_ADD_ENTRY => display_queue_add_entry(ctx, st),
        _ => return None,
    }
    Some(SvcOutcome::Continue)
}

/// A call that succeeds with return code 0 and no out-params.
fn ok(ctx: &mut GuestCtx) {
    ctx.ret(0);
}

/// Map*UsseMemory(base, size, unsigned int *usseOffset): return offset 0.
fn map_usse(ctx: &mut GuestCtx) {
    let out = ctx.arg(2);
    ctx.write_u32(out, 0);
    ctx.ret(0);
}

/// A create-call that writes a fresh opaque handle to its out-pointer at
/// positional argument `out_arg`, returning 0.
fn out_handle(ctx: &mut GuestCtx, st: &mut VitaState, out_arg: usize) {
    let out = ctx.arg(out_arg);
    let handle = st.new_handle();
    ctx.write_u32(out, handle);
    ctx.ret(0);
}

/// int sceGxmInitialize(const SceGxmInitializeParams *params)
fn initialize(ctx: &mut GuestCtx, st: &mut VitaState) {
    let params = ctx.arg(0);
    st.display_queue_cb = ctx.read_u32(params + INIT_CB_OFFSET);
    st.display_queue_cb_data_size = ctx.read_u32(params + INIT_CB_DATA_SIZE_OFFSET);
    ctx.ret(0);
}

/// const SceGxmProgramParameter *sceGxmProgramFindParameterByName(program, name)
/// We have no real parameter table (placeholder shaders), so hand back a nonzero
/// token. The uniform write path uses the explicit component offset/count the
/// guest passes to sceGxmSetUniformDataF, so the token is never dereferenced.
fn find_parameter(ctx: &mut GuestCtx, st: &mut VitaState) {
    let token = st.new_handle();
    ctx.ret(token);
}

/// int sceGxmColorSurfaceInit(surface, format, type, scale, outputRegisterSize,
///     width, height, strideInPixels, data)  -- 9 args (5 on the stack).
fn color_surface_init(ctx: &mut GuestCtx, st: &mut VitaState) {
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
fn create_vertex_program(ctx: &mut GuestCtx, st: &mut VitaState) {
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

    let handle = st.new_handle();
    st.set_vertex_program(handle, attributes, stride);
    ctx.write_u32(out, handle);
    ctx.ret(0);
}

/// int sceGxmBeginScene(context, flags, renderTarget, validRegion,
///     vertexSyncObject, fragmentSyncObject, colorSurface, depthStencil) -- 8 args.
fn begin_scene(ctx: &mut GuestCtx, st: &mut VitaState) {
    let color_surface = ctx.arg(6);
    st.begin_scene(color_surface);
    ctx.ret(0);
}

/// int sceGxmEndScene(context, notification, notification2)
fn end_scene(ctx: &mut GuestCtx, st: &mut VitaState) {
    st.end_scene();
    ctx.ret(0);
}

/// void sceGxmSetVertexProgram(context, vertexProgram)
fn set_vertex_program(ctx: &mut GuestCtx, st: &mut VitaState) {
    let vp = ctx.arg(1);
    st.bind_vertex_program(vp);
    ctx.ret(0);
}

/// int sceGxmReserveVertexDefaultUniformBuffer(context, void **uniformBuffer)
/// Hand back a small real guest buffer so a guest read-back would be faithful;
/// the uniform values are captured from the sceGxmSetUniformDataF source anyway.
fn reserve_uniforms(ctx: &mut GuestCtx, st: &mut VitaState) {
    let out = ctx.arg(1);
    let buf = st.galloc(256, 16);
    ctx.write_u32(out, buf);
    ctx.ret(0);
}

/// int sceGxmSetUniformDataF(void *uniformBuffer, const SceGxmProgramParameter
///     *parameter, unsigned int componentOffset, unsigned int componentCount,
///     const float *sourceData)  -- 5 args (1 on the stack).
fn set_uniform_data_f(ctx: &mut GuestCtx, st: &mut VitaState) {
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
fn set_vertex_stream(ctx: &mut GuestCtx, st: &mut VitaState) {
    let stream_index = ctx.arg(1);
    let data = ctx.arg(2);
    if stream_index == 0 {
        st.bind_stream0(data);
    }
    ctx.ret(0);
}

/// int sceGxmDraw(context, SceGxmPrimitiveType primitive, SceGxmIndexFormat
///     indexType, const void *indexData, unsigned int indexCount)  -- 5 args.
fn draw(ctx: &mut GuestCtx, st: &mut VitaState) {
    let primitive = ctx.arg(1);
    let index_format = ctx.arg(2);
    let index_data = ctx.arg(3);
    let index_count = ctx.arg(4);
    st.record_draw(ctx, primitive, index_format, index_data, index_count);
    ctx.ret(0);
}

/// int sceGxmDisplayQueueAddEntry(oldBuffer, newBuffer, const void *callbackData)
/// The callback data's first field is the display buffer address to present.
fn display_queue_add_entry(ctx: &mut GuestCtx, st: &mut VitaState) {
    let callback_data = ctx.arg(2);
    let buffer = ctx.read_u32(callback_data);
    if buffer != 0 {
        st.present(buffer);
    }
    ctx.ret(0);
}
