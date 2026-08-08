//! SceGxm: the graphics API. These handlers hand back opaque handles for GXM
//! objects, remember the surfaces and vertex-program layouts the guest sets up,
//! and record the per-frame draw stream (BeginScene to EndScene) with the vertex,
//! index, and uniform data snapshotted from guest memory. No GPU is emulated and
//! no pixel is drawn here; that is the renderer's job over this capture.

use crate::capture::{ColorSurface, VertexAttribute};
use crate::host::{GuestCtx, VitaState, MAX_VERTEX_STREAMS};
use crate::render::f32_to_half;
use vitaslop_gxp_shader::ParamType;
use crate::hostcall;

/// SceGxmInitializeParams: displayQueueCallback at offset 8, its data size at 12.
const INIT_CB_OFFSET: u32 = 8;
const INIT_CB_DATA_SIZE_OFFSET: u32 = 12;

/// A call that succeeds with return code 0 and no out-params.
pub(super) fn ok(ctx: &mut GuestCtx) {
    ctx.ret(0);
}

/// int sceGxmMapMemory(void *base, SceSize size, SceGxmMemoryAttribFlags attr)
///
/// The guest's pages already ARE the memory the capture reads, so nothing is mapped -
/// but the RANGE is remembered, because `sceGxmUnmapMemory` is given only a base and
/// has to be able to invalidate what was cached from it.
pub(super) fn map_memory(ctx: &mut GuestCtx, st: &mut VitaState) {
    let base = ctx.arg(0);
    let size = ctx.arg(1);
    st.gxm_map(base, size);
    ctx.ret(0);
}

/// Map*UsseMemory(base, size, unsigned int *usseOffset): return offset 0.
pub(super) fn map_usse(ctx: &mut GuestCtx, st: &mut VitaState) {
    let base = ctx.arg(0);
    let size = ctx.arg(1);
    let out = ctx.arg(2);
    st.gxm_map(base, size);
    ctx.write_u32(out, 0);
    ctx.ret(0);
}

/// int sceGxmCreateRenderTarget(const SceGxmRenderTargetParams *params,
///     SceGxmRenderTarget **renderTarget)
///
/// Hands back an opaque handle like any other create call, and remembers two things
/// from the params block: the `driverMemBlock` UID at +0x10, so
/// `sceGxmRenderTargetGetDriverMemBlock` returns the block the guest actually
/// supplied rather than a guess, and the `width`/`height` at +0x04/+0x06, which are
/// the EXTENT every scene begun on this target rasterizes into (see
/// [`VitaState::render_target_extent`]).
///
/// `SceGxmRenderTargetParams` (vitasdk `gxm.h`, asserted 0x14 bytes): `uint32 flags`,
/// `uint16 width`, `uint16 height`, `uint16 scenesPerFrame`, `uint16 multisampleMode`,
/// `uint32 multisampleLocations`, `SceUID driverMemBlock`.
pub(super) fn create_render_target(ctx: &mut GuestCtx, st: &mut VitaState) {
    let params = ctx.arg(0);
    let out = ctx.arg(1);
    let handle = st.new_handle();
    let dims = ctx.read_u32(params + 0x04);
    let (width, height) = (dims & 0xFFFF, dims >> 16);
    tracing::debug!(
        target: "vitaslop::gxm",
        target_handle = format_args!("{handle:#x}"),
        width, height,
        params = format_args!("{params:#x}"),
        raw = format_args!(
            "{:#010x} {:#010x} {:#010x} {:#010x} {:#010x}",
            ctx.read_u32(params),
            ctx.read_u32(params + 4),
            ctx.read_u32(params + 8),
            ctx.read_u32(params + 12),
            ctx.read_u32(params + 16),
        ),
        caller = format_args!("{:#010x}", ctx.regs[14]),
        "createRenderTarget"
    );
    st.set_render_target_extent(handle, width, height);
    st.set_render_target_mem_block(handle, ctx.read_u32(params + 0x10));
    report_multisample_mode(handle, width, height, (ctx.read_u32(params + 0x08) >> 16) & 0xFFFF);
    // A render target whose extent decodes DEGENERATE is either a title creating a real 1x1
    // probe or this reader looking at the wrong bytes, and those need opposite responses. The
    // 20-byte params block is the only thing that tells them apart, so print it unconditionally
    // rather than behind a log level: on one retail title the world pass comes through a target
    // that decodes 1x1 while every draw in it sets a 960x544 viewport, and a 1x1 target with a
    // 1x1 valid region would clip that pass away entirely on hardware.
    {
        eprintln!(
            "gxm render target {handle:#x} extent {width}x{height}{} - raw \
             SceGxmRenderTargetParams at {params:#x}: {:#010x} {:#010x} {:#010x} {:#010x} \
             {:#010x} (flags, width|height<<16, scenesPerFrame|multisample<<16, \
             multisampleLocations, driverMemBlock), caller lr={:#010x}",
            if width <= 1 || height <= 1 { " DEGENERATE" } else { "" },
            ctx.read_u32(params),
            ctx.read_u32(params + 4),
            ctx.read_u32(params + 8),
            ctx.read_u32(params + 12),
            ctx.read_u32(params + 16),
            ctx.regs[14],
        );
        // Every render target here is created through ONE thin wrapper, so the immediate `lr`
        // is the same for the good sizes and the degenerate ones and says nothing. The chain
        // above it is what differs: the caller that computed 1x1 is the bug, and it is several
        // frames up. A stack SCAN, not a frame-pointer walk - ARM leaf frames often keep none -
        // so these are candidates ordered by depth, not proof.
        let sp = ctx.regs[13];
        let chain: Vec<String> = (0..192u32)
            .map(|i| ctx.read_u32(sp.wrapping_add(i * 4)))
            .filter(|v| (0x8100_0000..0x8200_0000).contains(v))
            .map(|v| format!("{v:#010x}"))
            .collect();
        eprintln!("gxm render target {handle:#x}: caller candidates [{}]", chain.join(" "));
    }
    ctx.write_u32(out, handle);
    ctx.ret(0);
}

/// Report - once per (target, mode) - the MULTISAMPLE mode a render target was created with,
/// and that we rasterize it at one sample regardless.
///
/// `SceGxmMultisampleMode` is `NONE`/`2X`/`4X`, and 4X means the hardware keeps 2x2 samples per
/// pixel - which is why a 960x544 colour surface on this hardware carries a **1920x1088** depth
/// surface, a pairing that otherwise reads as a decode error. Rendering it at one sample is an
/// approximation (the title is aliased relative to hardware), and an approximation says so.
/// The multisample mode a render target was created with, by handle. Recorded here rather than
/// in `VitaState` because the only consumer is diagnostic - but it has to be a RECORD and not a
/// re-read of the params struct, which is a caller stack frame that is gone by `beginScene`.
static MULTISAMPLE_BY_TARGET: std::sync::Mutex<Option<std::collections::HashMap<u32, u32>>> =
    std::sync::Mutex::new(None);

fn multisample_mode_of(handle: u32) -> u32 {
    let g = MULTISAMPLE_BY_TARGET.lock().unwrap_or_else(|e| e.into_inner());
    g.as_ref().and_then(|m| m.get(&handle).copied()).unwrap_or(0)
}

fn report_multisample_mode(handle: u32, width: u32, height: u32, mode: u32) {
    {
        let mut g = MULTISAMPLE_BY_TARGET.lock().unwrap_or_else(|e| e.into_inner());
        g.get_or_insert_with(std::collections::HashMap::new).insert(handle, mode);
    }
    if mode == 0 {
        return;
    }
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<(u32, u32)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.insert((handle, mode)) {
        let (name, dw, dh) = match mode {
            1 => ("2X", width * 2, height),
            2 => ("4X", width * 2, height * 2),
            _ => ("an unrecognised mode", width, height),
        };
        eprintln!(
            "gxm render target {handle:#x} ({width}x{height}) was created MULTISAMPLED ({name}), \
             so on hardware its depth surface is {dw}x{dh} samples - we rasterize it at ONE \
             sample, which is more aliased than the title intends"
        );
    }
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

/// Byte offset of the header's `default_uniform_buffer_count`, which counts 32-bit SA
/// registers (so the byte size is four times it).
pub(crate) const GXP_DEFAULT_UNIFORM_BUFFER_COUNT_OFF: u32 = 0x64;

/// Largest `default_uniform_buffer_count` taken at face value. Beyond this the header did
/// not resolve to a real program, and the size is clamped rather than used to size an
/// allocation.
///
/// Shared with the inline form ([`inline_op`]) rather than repeated there: the inline
/// path treats this as the boundary between "answer inline" and "let the handler decide",
/// so the two drifting apart would put the clamp in one place and not the other - which
/// reads as a title occasionally truncating its own uniform upload.
pub(crate) const DEFAULT_UNIFORM_BUFFER_MAX_WORDS: u32 = 4096;

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

/// Byte offset of the packed attribute word within a `SceGxmProgramParameter`.
const GXM_PARAM_WORD_OFF: u32 = 4;
/// Byte offset of a parameter's `array_size`.
const GXM_PARAM_ARRAY_SIZE_OFF: u32 = 8;
/// Byte offset of a parameter's `resource_index`.
const GXM_PARAM_RESOURCE_INDEX_OFF: u32 = 0xC;

/// The inline form of a GXM host import, for the pure reflection getters - or `None`
/// for every NID that has real behaviour and must stay a host call. Reached through
/// [`crate::vita::inline_op`], which owns the global on/off switch.
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
    use vitaslop_transpiler::InlineOp::{LoadScaled, LoadShiftMask};
    // The packed word's fields; `param_word` masks the word to 16 bits first, which
    // the 4-bit field masks below make redundant.
    let word = |shift| LoadShiftMask { offset: GXM_PARAM_WORD_OFF, shift, mask: 0xf };
    // A field of a texture's CONTROL WORD 0, at the pointer itself (offset 0). Every one of
    // these is now a plain field of the guest's own struct rather than an entry in a host-side
    // map, which is what makes inlining them possible at all - see [`texword0`].
    let tex = |(shift, mask): (u32, u32)| LoadShiftMask { offset: 0, shift, mask };
    // ...and for the two enums whose values are already IN control-word position, the answer is
    // the masked word with no shift, so the mask is the field in place. See
    // [`texture_get_mip_filter`].
    let tex_in_place = |(shift, mask): (u32, u32)| LoadShiftMask { offset: 0, shift: 0, mask: mask << shift };
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
        // The texture sampler getters. `sceGxmTextureGetLodBias` is the single hottest host call
        // one title makes - 71,298 in one profile window - and every one of these is one load,
        // one shift and one mask over a struct the guest already owns.
        g::TEXTURE_GET_LOD_BIAS => tex(texword0::LOD_BIAS),
        g::TEXTURE_GET_U_ADDR_MODE_SAFE => tex(texword0::UADDR_MODE),
        g::TEXTURE_GET_V_ADDR_MODE_SAFE => tex(texword0::VADDR_MODE),
        g::TEXTURE_GET_MIN_FILTER => tex(texword0::MIN_FILTER),
        g::TEXTURE_GET_MAG_FILTER => tex(texword0::MAG_FILTER),
        // Both the checked and unchecked spellings read the same field, exactly as the two
        // share one handler - see the note on `TEXTURE_GET_MIPMAP_COUNT` in `nid.rs`.
        g::TEXTURE_GET_MIPMAP_COUNT | g::TEXTURE_GET_MIPMAP_COUNT_UNSAFE => {
            tex(texword0::MIP_COUNT)
        }
        g::TEXTURE_GET_GAMMA_MODE => tex_in_place(texword0::GAMMA_MODE),
        // The two PROGRAM-pointer reads. Everything above is handed a parameter record;
        // these are handed the `SceGxmProgram` itself, which changes nothing about the
        // lowering - an inline form is defined by (pointer argument, offset), and which
        // structure the pointer names is not the emitter's business.
        //
        // Both are called per draw by a title that re-reflects its shader interface every
        // frame: 21,710 and 24,760 calls in one profile window.
        g::PROGRAM_GET_PARAMETER_COUNT => {
            LoadShiftMask { offset: GXP_PARAM_COUNT_OFF, shift: 0, mask: u32::MAX }
        }
        // `default_uniform_buffer_bytes` is `read(+0x64).min(4096) * 4`. The scale is a
        // shift; the CLAMP is not, so the inline form covers values at or below the cap
        // and hands the rest back - see `InlineOp::LoadScaled`. A program whose header we
        // failed to resolve is exactly the clamped case, so the handler keeps defining the
        // answer precisely where the answer is least trustworthy.
        g::PROGRAM_GET_DEFAULT_UNIFORM_BUFFER_SIZE => LoadScaled {
            offset: GXP_DEFAULT_UNIFORM_BUFFER_COUNT_OFF,
            max: DEFAULT_UNIFORM_BUFFER_MAX_WORDS,
            shl: 2,
        },
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
///
/// # This was memoized, and the memoization was REVERTED. Do not re-add it.
/// The obvious optimisation - build a name -> record map per program and hash into it -
/// was implemented, tested and MEASURED on 2026-08-08e, and it is worth nothing: over a
/// 2000-frame boot-inclusive window of a real title, 24,760 calls went from 0.58 to 0.57
/// us each, a saving of 0.2 ms in 19 s.
///
/// The reason is that the walk was never the cost. The line below it - reading the
/// REQUESTED name out of guest memory - happens on every call whatever the lookup does,
/// and it dominates. Caching removed the cheap half. In exchange it added a per-program
/// cache keyed by a guest ADDRESS, which a title may reuse for a different shader: two
/// programs agreeing in parameter count and parameters offset, and sharing a name, would
/// return the WRONG record - a uniform written to the wrong place, surfacing as a shading
/// bug nowhere near here. That is a real hazard bought for 0.01 us a call.
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
/// a retail racer does: it initialises its nine render targets into a static array and then
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
const CS_SCALE: u32 = 28;

/// Byte offset of the `SceGxmTexture backgroundTex` a `SceGxmColorSurface` ends with.
///
/// `SceGxmColorSurface` is **0x30 bytes**, not 0x20: `pbeSidebandWord` + `pbeEmitWords[6]` +
/// `outputRegisterSize` fill the first 32, and a whole 16-byte `SceGxmTexture` follows
/// (vitasdk `gxm.h`, `VITASDK_BUILD_ASSERT_EQ(0x30, SceGxmColorSurface)`).
const CS_BACKGROUND_TEX: u32 = 32;

/// Write `surface`'s fields into the guest struct so a copy of it stays resolvable, INCLUDING
/// the `backgroundTex` a real `sceGxmColorSurfaceInit` leaves at the end of it.
///
/// # Why `backgroundTex` is load-bearing and not an optional extra
/// It is how a title reads its own render target back - both its SIZE and its pixels. Leaving
/// it zeroed is not a missing convenience; it is a wrong ANSWER, because
/// `sceGxmTextureGetWidth` on sixteen zero bytes returns 1 (the control words store size-1).
/// MEASURED on one retail racer: it sizes every screen-sized render target from
/// `sceGxmTextureGetWidth(&surface->backgroundTex)`, so it created its world, glow and bloom
/// targets **1x1**, and derived every colour surface and valid region from that. The frame only
/// looked right because the renderer fell back to guessing each pass's extent from its
/// viewport. The same gap left twelve texture handles all-zero at bind time - the ones the
/// title builds from a colour surface - which is what made its whole bloom chain black.
///
/// An earlier reading of the same evidence from the other side is recorded above: writing a
/// ninth word here "corrupted whatever the guest had placed after the struct" and cost a
/// sampler binding. That was this texture, being overwritten with our field rather than filled.
fn write_color_surface(ctx: &mut GuestCtx, addr: u32, s: &ColorSurface) {
    ctx.write_u32(addr, COLOR_SURFACE_MAGIC);
    ctx.write_u32(addr + CS_FORMAT, s.format);
    ctx.write_u32(addr + CS_TYPE, s.surface_type);
    ctx.write_u32(addr + CS_WIDTH, s.width);
    ctx.write_u32(addr + CS_HEIGHT, s.height);
    ctx.write_u32(addr + CS_STRIDE, s.stride_pixels);
    ctx.write_u32(addr + CS_DATA, s.data_addr);
    ctx.write_u32(addr + CS_SCALE, s.scale_mode);
    write_background_tex(ctx, addr + CS_BACKGROUND_TEX, s);
}

/// Fill the 16-byte `SceGxmTexture` that describes `s` itself, at `addr`.
///
/// The layout is [`texture_init`]'s, so a title that binds this texture and one that binds a
/// texture it built by hand over the same memory go down exactly the same path. A surface whose
/// stride differs from its width is LINEAR_STRIDED (that is what the layout means); otherwise
/// plain LINEAR.
fn write_background_tex(ctx: &mut GuestCtx, addr: u32, s: &ColorSurface) {
    report_background_tex_had_data(ctx, addr);
    let tex_format = color_format_to_texture_format(s.format);
    let base_format = (tex_format >> 24) & 0xff;
    let swizzle = (tex_format >> 12) & 0x7;
    let type_field =
        if s.stride_pixels != 0 && s.stride_pixels != s.width { TYPE_LINEAR_STRIDED } else { TYPE_LINEAR };
    write_texture_control_words(
        ctx,
        addr,
        type_field,
        base_format,
        swizzle,
        s.width,
        s.height,
        s.data_addr,
        // A colour surface's background texture is a single image - a render target has no mip
        // chain - so one level, and `None` for the strided case where those bits are the stride.
        (type_field != TYPE_LINEAR_STRIDED).then_some(1),
    );
}

/// Report - once - that a colour surface's `backgroundTex` slot held data before we filled it.
///
/// It should not: `sceGxmColorSurfaceInit` owns those sixteen bytes, so anything already there
/// is either a title re-initialising a live surface (harmless, the contents are ours) or this
/// implementation writing over something the title put there, which is the failure the surface
/// size comment above records a previous session mis-diagnosing. Saying so makes the difference
/// between the two visible in any run rather than after a pixel diff.
fn report_background_tex_had_data(ctx: &mut GuestCtx, addr: u32) {
    let words = [
        ctx.read_u32(addr),
        ctx.read_u32(addr + 4),
        ctx.read_u32(addr + 8),
        ctx.read_u32(addr + 12),
    ];
    if words.iter().all(|w| *w == 0) {
        return;
    }
    use std::sync::atomic::{AtomicBool, Ordering};
    static SEEN: AtomicBool = AtomicBool::new(false);
    if SEEN.swap(true, Ordering::Relaxed) {
        return;
    }
    eprintln!(
        "gxm surface: the backgroundTex slot at {addr:#x} already held {:#010x} {:#010x} \
         {:#010x} {:#010x} before sceGxmColorSurfaceInit filled it",
        words[0], words[1], words[2], words[3]
    );
}

/// Write the four `SceGxmTexture` control words for a texture of this geometry.
///
/// Shared by `sceGxmTextureInit*` and by the `backgroundTex` a colour surface carries, so the
/// two can never drift. Note bit 31 of word 0: a base format above 0x7f does not fit the 5-bit
/// field in word 1 and its top bit lives there - without it a texture read back from its control
/// words alone (which is what happens once a title COPIES the struct) silently becomes a
/// different format, and every compressed format plus `U2F10F10F10` is above 0x7f.
#[allow(clippy::too_many_arguments)]
fn write_texture_control_words(
    ctx: &mut GuestCtx,
    addr: u32,
    type_field: u32,
    base_format: u32,
    swizzle: u32,
    width: u32,
    height: u32,
    data: u32,
    // `mipCount` as the guest passed it to `sceGxmTextureInit*`, or `None` for a
    // `LINEAR_STRIDED` texture, where the same argument is a byte stride and these bits belong
    // to the stride field instead (see [`texword0`]).
    mip_count: Option<u32>,
) {
    // Word 0's sampler fields are all left ZERO here, which is not a shortcut: GXM's defaults
    // for every one of them IS zero - REPEAT addressing, POINT filtering, mip filter disabled,
    // lod bias 0, gamma off - so a freshly initialised texture reads back the documented
    // defaults from its own control word with nothing else written.
    let mip = mip_count.map_or(0, |n| {
        let (shift, mask) = texword0::MIP_COUNT;
        (n & mask) << shift
    });
    let w0 = (((base_format >> 7) & 1) << 31) | mip;
    let w1 = (height.saturating_sub(1) & 0xfff)
        | ((width.saturating_sub(1) & 0xfff) << 12)
        | ((base_format & 0x1f) << 24)
        | (type_field << 29);
    ctx.write_u32(addr, w0);
    ctx.write_u32(addr + 4, w1);
    ctx.write_u32(addr + 8, data & 0xffff_fffc);
    ctx.write_u32(addr + 12, (swizzle & 0x7) << 29);
}

/// Map a `SceGxmColorFormat` to the `SceGxmTextureFormat` naming the same pixels.
///
/// The two enums are separate numberings of the same set, so this is a table, not arithmetic:
/// a colour format's BASE occupies bits 31:28 plus bits 24 and 23 (mask `0xF1800000`) and its
/// SWIZZLE bits 21:20, while a texture format's base is bits 31:24 and its swizzle bits 14:12.
/// Within the four- and three-component families the swizzle enumerations agree name for name,
/// so those shift straight across; the one- and two-component families do not, and a nonzero
/// swizzle there is REPORTED rather than translated into a channel order it might not be.
/// Both tables are from vitasdk `gxm.h`.
fn color_format_to_texture_format(color_format: u32) -> u32 {
    let base = color_format & 0xF180_0000;
    let swizzle = color_format & 0x0030_0000;
    // (colour base, texture base, does the swizzle enumeration carry across?)
    const TABLE: [(u32, u32, bool); 25] = [
        (0x0000_0000, 0x0C00_0000, true),  // U8U8U8U8
        (0x1000_0000, 0x9800_0000, true),  // U8U8U8
        (0x3000_0000, 0x0500_0000, true),  // U5U6U5
        (0x4000_0000, 0x0400_0000, true),  // U1U5U5U5
        (0x5000_0000, 0x0200_0000, true),  // U4U4U4U4
        (0x6000_0000, 0x0300_0000, true),  // U8U3U3U2
        (0xF000_0000, 0x0B00_0000, false), // F16
        (0x0080_0000, 0x1100_0000, false), // F16F16
        (0x1080_0000, 0x1200_0000, false), // F32
        (0x2080_0000, 0x0A00_0000, false), // S16
        (0x3080_0000, 0x1000_0000, false), // S16S16
        (0x4080_0000, 0x0900_0000, false), // U16
        (0x5080_0000, 0x0F00_0000, false), // U16U16
        (0x6080_0000, 0x0E00_0000, true),  // U2U10U10U10
        (0x8080_0000, 0x0000_0000, false), // U8
        (0x9080_0000, 0x0100_0000, false), // S8
        (0xA080_0000, 0x0600_0000, true),  // S5S5U6
        (0xB080_0000, 0x0700_0000, false), // U8U8
        (0xC080_0000, 0x0800_0000, false), // S8S8
        (0xD080_0000, 0x1400_0000, true),  // U8S8S8U8 -> X8S8S8U8
        (0xE080_0000, 0x0D00_0000, true),  // S8S8S8S8
        (0x0100_0000, 0x1B00_0000, true),  // F16F16F16F16
        (0x1100_0000, 0x1E00_0000, false), // F32F32
        (0x2100_0000, 0x1A00_0000, true),  // F11F11F10
        (0x3100_0000, 0x1900_0000, true),  // SE5M9M9M9
    ];
    // U2F10F10F10 is last so the table above stays one screen; kept separate only to keep the
    // array length honest with its declared size.
    if base == 0x4100_0000 {
        return 0x9A00_0000 | (swizzle >> 8);
    }
    let Some(&(_, tex_base, swizzle_carries)) = TABLE.iter().find(|(c, _, _)| *c == base) else {
        report_unmapped_color_format(color_format);
        // The geometry still has to be right: a title sizes its render targets from this
        // texture's width and height, and getting those wrong is a far larger error than a
        // wrong channel order. U8U8U8U8 is the only format every consumer here can size.
        return 0x0C00_0000;
    };
    if !swizzle_carries && swizzle != 0 {
        report_unmapped_color_format(color_format);
        return tex_base;
    }
    tex_base | (swizzle >> 8)
}

/// Report - once per format - a colour format whose texture equivalent is not established.
fn report_unmapped_color_format(color_format: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(color_format) {
        return;
    }
    eprintln!(
        "gxm surface: colour format {color_format:#010x} has no established SceGxmTextureFormat \
         equivalent, so the backgroundTex this surface carries names the right PIXELS at the \
         right size but may name the wrong channel order. A title sampling its own render \
         target through it will get its colours permuted."
    );
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
        scale_mode: ctx.read_u32(addr + CS_SCALE),
        // NOT read from guest memory: a `SceGxmColorSurface` is 32 bytes (eight control words)
        // and there is no ninth to keep this in. Writing one corrupted whatever the guest had
        // placed after the struct - measured, and it cost this title a sampler binding, which
        // surfaced as a shader falling back for a texture unit the guest had definitely bound.
        // The mode lives in the host-side table and is merged in by `resolve_color_surface`.
        gamma: 0,
    })
}

/// int sceGxmColorSurfaceInit(surface, format, type, scale, outputRegisterSize,
///     width, height, strideInPixels, data)  -- 9 args (5 on the stack).
pub(super) fn color_surface_init(ctx: &mut GuestCtx, st: &mut VitaState) {
    let surface = ctx.arg(0);
    let format = ctx.arg(1);
    let surface_type = ctx.arg(2);
    let scale_mode = ctx.arg(3);
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
        // The four stack arguments and the caller, because a surface that arrives
        // with a zero extent is either a title doing something unusual or this
        // handler reading the wrong stack slots, and only the raw words plus the
        // call site tell those apart.
        out_reg_size = ctx.arg(4),
        caller = format_args!("{:#010x}", ctx.regs[14]),
        "colorSurfaceInit"
    );
    let s = ColorSurface { format, surface_type, width, height, stride_pixels, data_addr, scale_mode, gamma: 0 };
    report_color_surface_scale_mode(data_addr, scale_mode);
    write_color_surface(ctx, surface, &s);
    st.set_color_surface(surface, s);
    ctx.ret(0);
}

/// Report - once per (surface data address, mode) - a colour surface created with a SCALE MODE
/// we do not honour.
///
/// `SCE_GXM_COLOR_SURFACE_SCALE_MSAA_DOWNSCALE` (1) means the pass RASTERISES at twice the
/// surface's resolution in each axis and the hardware resolves 2x2 samples into each stored
/// pixel. We rasterise at the surface's own resolution and store that, which is not a rounding
/// difference: everything the guest derives from the two resolutions - a post-process pass's
/// texel size, a screen-space scale/bias, a depth surface it then samples - is computed for a
/// buffer twice the size of the one we produced.
///
/// It is a report rather than a fix because honouring it means rasterising a pass at 2x and
/// resolving, which is a real piece of work; and an approximation that says nothing is
/// indistinguishable on screen from a faithful render, which is exactly how a wrong frame
/// survives being stared at. MEASURED on one retail racer: its world colour surface asks for
/// MSAA_DOWNSCALE and the depth surface the guest then samples is described as exactly 2x2 the
/// colour resolution - the two facts agree, and both disagree with what we render.
fn report_color_surface_scale_mode(data_addr: u32, scale_mode: u32) {
    if scale_mode == 0 {
        return;
    }
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((data_addr, scale_mode)) {
        return;
    }
    let name = match scale_mode {
        1 => "MSAA_DOWNSCALE (the guest rasterises this pass at 2x2 the stored resolution and \
              resolves into it)",
        _ => "an unrecognised scale mode",
    };
    eprintln!(
        "gxm surface: colour surface at {data_addr:#x} was created with {name} - we rasterise it \
         at the stored resolution and IGNORE the mode, so anything the guest derives from the \
         two resolutions is computed for a buffer twice the size of the one we produce"
    );
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
/// Report each DISTINCT `SceGxmBlendInfo` a title creates a fragment program with.
///
/// The blend equation is baked in here and never mentioned again, so this is the only place
/// it can be observed - and it decides whether a shader that outputs alpha 0 is invisible or
/// perfectly fine. A handful of distinct values covers a whole title, so printing each once is
/// cheap and says exactly what the renderer has to reproduce.
///
/// Keyed by the `SceGxmProgram*` as well as the decoded state, and it PRINTS that address: it is
/// the same address the shader dumps are named by (`frag_<header>.gxp`), so this line is what
/// ties "which blend equation" to "which blob" without another run. Deduping on the blend state
/// alone said a title used four equations and left no way to ask which program had which.
fn report_blend_info(program_header: u32, blend_info: u32, blend: crate::capture::BlendState) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<(u32, crate::capture::BlendState)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((program_header, blend)) {
        return;
    }
    eprintln!(
        "gxm blend: fragment program {program_header:#x} created with {} - mask={:#x} colorFunc={} alphaFunc={} \
         colorSrc={} colorDst={} alphaSrc={} alphaDst={} (blends={})",
        if blend_info == 0 { "a NULL blendInfo" } else { "a blendInfo" },
        blend.color_mask,
        blend.color_func,
        blend.alpha_func,
        blend.color_src,
        blend.color_dst,
        blend.alpha_src,
        blend.alpha_dst,
        blend.blends(),
    );
}

pub(super) fn create_fragment_program(ctx: &mut GuestCtx, st: &mut VitaState) {
    let program_id = ctx.arg(1);
    let blend_info = ctx.arg(4);
    let out = ctx.arg(6);
    let program_header = st.shader_program(program_id);
    let handle = st.new_handle();
    // The BLEND EQUATION arrives here and nowhere else - GXM has no runtime blend setter, so
    // a program created with a NULL `blendInfo` never blends and one created with an additive
    // info always does. Dropping this argument is what forced every renderer downstream to
    // guess the mode from the geometry, and a guess is wrong for whole classes of draw.
    let blend = match ctx.read_bytes(blend_info, 4) {
        b if blend_info != 0 && b.len() == 4 => {
            crate::capture::BlendState::from_bytes([b[0], b[1], b[2], b[3]])
        }
        _ => crate::capture::BlendState::default(),
    };
    report_blend_info(program_header, blend_info, blend);
    st.set_fragment_program(handle, program_header, blend);
    ctx.write_u32(out, handle);
    ctx.ret(0);
}

/// int sceGxmBeginScene(context, flags, renderTarget, validRegion,
///     vertexSyncObject, fragmentSyncObject, colorSurface, depthStencil) -- 8 args.
pub(super) fn begin_scene(ctx: &mut GuestCtx, st: &mut VitaState) {
    let render_target = ctx.arg(2);
    let valid_region = ctx.arg(3);
    let color_surface = ctx.arg(6);
    let depth_stencil = ctx.arg(7);
    // Resolve from the struct's own contents first (survives a copy), then from the
    // address table (covers a surface whose bytes a title has since overwritten).
    // `resolve_color_surface` also merges in the sticky gamma mode, which the 32-byte guest
    // struct cannot carry - and the scene's target is exactly where it has to arrive.
    let mut color = resolve_color_surface(ctx, st, color_surface);
    let surface_extent = color.as_ref().map(|c| (c.width, c.height));
    // The RENDER TARGET carries the extent this scene rasterizes into; the colour
    // surface only says where the pixels land. Taking the extent from the surface
    // works for the display buffers (a title initialises those with the real size)
    // and fails silently for render-to-texture, where a title may fill the surface
    // struct from a template: this one begins a 20,160-triangle pass on a 1024x1024
    // target through a colour surface initialised 1x1, and reading the surface made
    // that whole pass a single pixel. Where both are known the render target wins.
    if let (Some(c), Some((w, h))) = (color.as_mut(), st.render_target_extent(render_target)) {
        if w != 0 && h != 0 && (c.width, c.height) != (w, h) {
            tracing::debug!(
                target: "vitaslop::gxm",
                surface = format_args!("{color_surface:#x}"),
                surface_extent = format_args!("{}x{}", c.width, c.height),
                target_extent = format_args!("{w}x{h}"),
                "beginScene: taking the scene extent from the render target, not the \
                 colour surface"
            );
            c.width = w;
            c.height = h;
        }
    }
    if color.is_none() {
        tracing::debug!(
            target: "vitaslop::gxm",
            surface = format_args!("{color_surface:#x}"),
            "beginScene with an unrecognised colour surface - this scene's render target is \
             unknown, so a later pass sampling it cannot be chained to it"
        );
    }
    // Which RENDER TARGET this scene rasterises through, next to the SCALE MODE its colour
    // surface asks for. The two are one setting, not two: MSAA_DOWNSCALE means "rasterise at
    // 2x2 the stored resolution", which only makes sense against a multisampled target, and a
    // title that copies one surface template into several passes can pair them wrongly. Neither
    // fact is visible in the frame, and reading either alone has already produced a wrong
    // conclusion here.
    report_scene_target(
        color.as_ref().map(|c| c.data_addr).unwrap_or(0),
        render_target,
        st.render_target_extent(render_target),
        color.as_ref().map(|c| c.scale_mode).unwrap_or(0),
    );
    report_scene_extent_sources(
        color.as_ref().map(|c| c.data_addr).unwrap_or(0),
        surface_extent,
        st.render_target_extent(render_target),
        read_valid_region(ctx, valid_region),
        color.as_ref().map(|c| (c.width, c.height)),
    );
    let depth = read_depth_stencil_surface(ctx, depth_stencil);
    report_scene_depth(
        color.as_ref().map(|c| c.data_addr).unwrap_or(0),
        color_surface,
        depth_stencil,
        depth,
    );
    st.begin_scene(color, depth);
    ctx.ret(0);
}

/// Read a `SceGxmDepthStencilSurface` out of guest memory, or `None` for a null pointer.
///
/// The struct's layout IS published (see the constants above), so this reads the guest's own
/// fields rather than consulting a side table - which also means it survives the guest copying
/// the struct, exactly as the setters above assume.
fn read_depth_stencil_surface(
    ctx: &mut GuestCtx,
    surface: u32,
) -> Option<crate::capture::DepthSurface> {
    if surface == 0 {
        return None;
    }
    Some(crate::capture::DepthSurface {
        zls_control: ctx.read_u32(surface + DS_ZLS_CONTROL),
        depth_addr: ctx.read_u32(surface + DS_DEPTH_DATA),
        stencil_addr: ctx.read_u32(surface + DS_STENCIL_DATA),
        background_depth: ctx.read_u32(surface + DS_BACKGROUND_DEPTH),
    })
}

/// Report - once per distinct (colour target, render target) pairing - which render target a
/// scene rasterises through, its extent and multisample mode, and the scale mode the colour
/// surface asks for. See the call site for why those belong on one line.
fn report_scene_target(
    color_addr: u32,
    render_target: u32,
    extent: Option<(u32, u32)>,
    scale_mode: u32,
) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    // The EXTENT is in the key as well as the pairing: the same colour buffer goes through the
    // same render target at two different sizes over a run (a 1x1 dummy while a title is
    // loading, its real size in play), and a pairing-only dedup prints the boot one and hides
    // the one the frame is actually built from.
    static SEEN: Mutex<Option<HashSet<(u32, u32, u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let (w, h) = extent.unwrap_or((0, 0));
    if !g.get_or_insert_with(HashSet::new).insert((color_addr, render_target, w, h)) {
        return;
    }
    eprintln!(
        "gxm scene target: colour {color_addr:#x} rasterises through render target \
         {render_target:#x} ({w}x{h}, multisample {}) with colour scale mode {scale_mode}{}",
        multisample_mode_of(render_target),
        if scale_mode == 1 && multisample_mode_of(render_target) == 0 {
            "  <-- MSAA_DOWNSCALE on a target we read as NOT multisampled: one of those two \
             readings is wrong, and until it is settled the resolution this pass really \
             rasterises at is UNKNOWN"
        } else {
            ""
        }
    );
}

/// `SceGxmValidRegion { unsigned int xMax; unsigned int yMax; }` - `sceGxmBeginScene`'s
/// argument 3, the sub-rectangle of the render target this scene is allowed to touch.
/// `None` for the null pointer, which is what a title passes to mean "the whole target".
fn read_valid_region(ctx: &mut GuestCtx, valid_region: u32) -> Option<(u32, u32)> {
    if valid_region == 0 {
        return None;
    }
    Some((ctx.read_u32(valid_region), ctx.read_u32(valid_region + 4)))
}

/// Report - once per colour target - every INDEPENDENT statement of this scene's extent next to
/// the one we ended up rasterising at.
///
/// Three sources describe it and they can disagree: the colour surface's own width/height, the
/// render target's, and `sceGxmBeginScene`'s `validRegion`. Reading any one alone has already
/// produced a wrong answer here (a 1024x1024 pass came through a surface initialised 1x1), and
/// the current open question on one retail title is a world pass that lands at 960x544 while
/// BOTH the surface and the target read 1x1 - which no single-source reading can explain. Only
/// printing all of them together says which one the frame is actually built from.
fn report_scene_extent_sources(
    color_addr: u32,
    surface: Option<(u32, u32)>,
    target: Option<(u32, u32)>,
    valid_region: Option<(u32, u32)>,
    used: Option<(u32, u32)>,
) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    // Keyed on the VALUES, not just the address. A title reuses one colour buffer for a 1x1
    // dummy pass during boot and for its 960x544 world pass in play, and a per-address dedup
    // reports only the first - which reads as "that pass is 1x1" for the whole run and sent a
    // session hunting an extent bug that does not exist.
    #[allow(clippy::type_complexity)]
    static SEEN: Mutex<
        Option<HashSet<(u32, Option<(u32, u32)>, Option<(u32, u32)>, Option<(u32, u32)>)>>,
    > = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((color_addr, surface, target, valid_region)) {
        return;
    }
    let fmt = |e: Option<(u32, u32)>| match e {
        Some((w, h)) => format!("{w}x{h}"),
        None => "(none)".to_string(),
    };
    eprintln!(
        "gxm scene extent: colour {color_addr:#x} surface={} target={} validRegion={} -> \
         RASTERISED AT {}",
        fmt(surface),
        fmt(target),
        fmt(valid_region),
        fmt(used)
    );
}

/// Report - once per distinct (colour target, depth surface) pairing - where a scene puts its
/// depth.
///
/// This is the fact that tells a later pass sampling a depth buffer apart from one sampling a
/// colour buffer, and the two are allocated close enough together that an address-range match
/// silently resolves the wrong one. Printing the pairing makes that visible in any run.
fn report_scene_depth(
    color_addr: u32,
    color_surface: u32,
    depth_stencil: u32,
    depth: Option<crate::capture::DepthSurface>,
) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::HashSet<(u32, u32, u32)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    let d = depth.unwrap_or_default();
    if seen.insert((color_addr, d.depth_addr, d.stencil_addr)) {
        // The STRUCT addresses too, not just the pixel addresses they hold. On one retail racer
        // a scene reports its colour at `0x89204aa0` and its depth 256 bytes later, for buffers
        // that are two megabytes each - so they cannot both be where they say they are, and the
        // next question is always "were those pointers read out of the right structs". Only the
        // struct addresses let a `VITASLOP_PEEK` answer it. (A well-formed pass in the same
        // frame puts its depth just PAST its colour buffer, which is what the anomaly is
        // measured against.)
        eprintln!(
            "gxm scene depth: colour {color_addr:#x} (surface struct {color_surface:#x}) renders \
             depth into {:#x} (stencil {:#x}, zlsControl {:#010x}, background {}, depthStencil \
             struct {depth_stencil:#x})",
            d.depth_addr,
            d.stencil_addr,
            d.zls_control,
            f32::from_bits(d.background_depth)
        );
    }
}

/// int sceGxmEndScene(context, const SceGxmNotification *vertexNotification,
///     const SceGxmNotification *fragmentNotification)
///
/// Ending the scene is where the GPU's work for it finishes, so it is also where the
/// two optional notifications are signalled and where an occlusion query's counts land
/// in the guest's visibility buffer. All of that is synchronous here, which is why
/// `sceGxmNotificationWait` never actually has to wait.
pub(super) fn end_scene(ctx: &mut GuestCtx, st: &mut VitaState) {
    st.end_scene();
    st.flush_visibility(ctx);
    let (vertex_notification, fragment_notification) = (ctx.arg(1), ctx.arg(2));
    signal_notification(ctx, vertex_notification);
    signal_notification(ctx, fragment_notification);
    ctx.ret(0);
}

/// Write a `SceGxmNotification`'s `value` through its `address`, which is what the GPU
/// does when the work the notification was attached to completes. `{ volatile unsigned
/// int *address; unsigned int value; }`, per vitasdk `gxm.h`.
fn signal_notification(ctx: &mut GuestCtx, notification: u32) {
    if notification == 0 {
        return;
    }
    let address = ctx.read_u32(notification);
    let value = ctx.read_u32(notification + 4);
    if address != 0 {
        ctx.write_u32(address, value);
    }
}

/// int sceGxmNotificationWait(const SceGxmNotification *notification)
///
/// Block until `*notification->address == notification->value`. Every scene completes
/// synchronously here and signals its notifications at `sceGxmEndScene`, so by the time
/// a title waits the value is already there and this returns at once.
///
/// A notification that is NOT already signalled means it was never attached to a scene
/// that ended - waiting for it would hang forever, so it is signalled here instead, and
/// reported, because a wait that silently returns without its condition holding is the
/// kind of thing that surfaces thousands of frames away.
pub(super) fn notification_wait(ctx: &mut GuestCtx, _st: &mut VitaState) {
    let notification = ctx.arg(0);
    let address = ctx.read_u32(notification);
    let value = ctx.read_u32(notification + 4);
    if address != 0 && ctx.read_u32(address) != value {
        tracing::warn!(
            target: "vitaslop::gxm",
            address = format_args!("{address:#x}"),
            value,
            "notificationWait on a notification no ended scene signalled - signalling it \
             here rather than waiting forever"
        );
        ctx.write_u32(address, value);
    }
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
    let parameter = ctx.arg(1);
    let component_offset = ctx.arg(2);
    let component_count = ctx.arg(3);
    let source = ctx.arg(4);

    // `componentOffset` is relative to the PARAMETER, not to the buffer: where the parameter
    // itself starts is its `resource_index`, which for a UNIFORM is its 4-byte-register offset
    // into the default uniform buffer. Ignoring the parameter argument (as this once did) writes
    // every uniform at the top of the buffer, so any program whose second uniform is not at
    // register 0 reads a value the guest never put there - and the shader that multiplies by it
    // paints black without any error. The record layout is the parameter table's own (16 bytes:
    // name_rel, packed, array_size, resource_index), so the index is at +12.
    // The field is signed on disk, and only a non-negative register offset can name a slot in
    // this buffer - anything else is a record we are not reading correctly, so fall back to the
    // top of the buffer rather than write at a wild offset.
    let base = match parameter {
        0 => 0,
        p => (ctx.read_u32(p + 12) as i32).max(0) as u32,
    };
    // The parameter's own declared TYPE decides how wide a component is in the buffer. A
    // half-float uniform packs TWO components per 4-byte register, and the shader reads it
    // back that way - so writing one 4-byte float per component both puts every component at
    // the wrong offset AND stores a bit pattern the shader will unpack as two halves. That
    // is a silent corruption of every uniform after the first, which is why the width comes
    // from the record rather than being assumed. The record layout is the parameter table's
    // own 16 bytes (name_rel, packed, array_size, resource_index), and the type is the second
    // nibble of `packed` - the same field `SceGxmProgramParameter` reflection reads.
    let half = parameter != 0
        && matches!(ParamType::from_bits(((ctx.read_u32(parameter + 4) >> 4) & 0xf) as u8), ParamType::F16);
    let mut values = Vec::with_capacity(component_count as usize);
    for i in 0..component_count {
        values.push(ctx.read_f32(source + i * 4));
    }
    // Faithful copy into the reserved buffer (in case the guest reads it back, and because a
    // recompiled shader reads this buffer verbatim).
    for (i, v) in values.iter().enumerate() {
        let component = base * if half { 2 } else { 1 } + component_offset + i as u32;
        if half {
            // Two halves per register: read-modify-write the other half so a partial update
            // (the common `componentOffset` case) does not clear its neighbour.
            let addr = uniform_buffer + (component / 2) * 4;
            let word = ctx.read_u32(addr);
            let h = u32::from(f32_to_half(*v));
            let merged = if component % 2 == 0 { (word & 0xffff_0000) | h } else { (word & 0x0000_ffff) | (h << 16) };
            ctx.write_u32(addr, merged);
        } else {
            ctx.write_u32(uniform_buffer + component * 4, v.to_bits());
        }
    }
    report_uniform_write(ctx, uniform_buffer, parameter, base, component_offset, half, &values);
    tracing::trace!(
        target: "vitaslop::gxm",
        buffer = format_args!("{uniform_buffer:#x}"),
        parameter = format_args!("{parameter:#x}"),
        base,
        component_offset,
        component_count,
        half,
        "setUniformDataF"
    );
    if half {
        st.set_uniform_halves(base * 2 + component_offset, &values);
    } else {
        st.set_uniforms(base + component_offset, values);
    }
    ctx.ret(0);
}

/// `VITASLOP_UNIFORM_WATCH=<hex address>|<parameter name substring>[,...]`: report every
/// `sceGxmSetUniformDataF` that lands on one of those addresses OR writes a parameter whose
/// name matches, with the values the guest passed and the word left behind.
///
/// A name is usually the right handle. The uniform block a title bakes a material into is
/// heap, so its address moves between runs, and by the time a wrong value is noticed - in a
/// frame - the address is all that is left of it; the NAME is stable and is what the question
/// was actually about.
///
/// A guest-store watchpoint cannot see this. `sceGxmSetUniformDataF` takes an arbitrary buffer
/// pointer - a title bakes a material's uniform block once with it and memcpys the block per
/// draw - so the bytes a shader ends up reading may have been written by US, on the guest's
/// behalf, with no guest store anywhere near them. "No guest store ever writes that address"
/// then reads as "it must be static data" when the real answer is "a host call put it there"
/// (memory `vitaslop-host-call-reference-semantics`).
fn report_uniform_write(
    ctx: &GuestCtx,
    buffer: u32,
    parameter: u32,
    base: u32,
    component_offset: u32,
    half: bool,
    values: &[f32],
) {
    use std::sync::OnceLock;
    static WATCH: OnceLock<(Vec<u32>, Vec<String>)> = OnceLock::new();
    let (addrs, names) = WATCH.get_or_init(|| {
        let (mut a, mut n) = (Vec::new(), Vec::new());
        for t in std::env::var("VITASLOP_UNIFORM_WATCH").unwrap_or_default().split(',') {
            let t = t.trim();
            if t.is_empty() {
                continue;
            }
            match u32::from_str_radix(t.trim_start_matches("0x"), 16) {
                Ok(v) => a.push(v),
                Err(_) => n.push(t.to_string()),
            }
        }
        (a, n)
    });
    if (addrs.is_empty() && names.is_empty()) || values.is_empty() {
        return;
    }
    // The byte range this call touched, in the buffer's own addressing.
    let first = base * if half { 2 } else { 1 } + component_offset;
    let last = first + values.len() as u32 - 1;
    let (lo, hi) = match half {
        true => (buffer + (first / 2) * 4, buffer + (last / 2) * 4 + 3),
        false => (buffer + first * 4, buffer + last * 4 + 3),
    };
    // The parameter's name, which is what makes the line readable AND is the other way to
    // select one: an offset says where the write went, the name says what the guest thought it
    // was writing.
    let name = (parameter != 0)
        .then(|| {
            let rel = ctx.read_u32(parameter);
            ctx.read_cstr(parameter.wrapping_add(rel), 64)
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "<unnamed>".into());
    if !addrs.iter().any(|&a| a >= lo && a <= hi) && !names.iter().any(|n| name.contains(n.as_str()))
    {
        return;
    }
    eprintln!(
        "gxm uniform watch: sceGxmSetUniformDataF wrote {name} ({}) into {lo:#x}..={hi:#x} of \
         buffer {buffer:#x} - reg {base}, component offset {component_offset}, values {values:?}, \
         leaving {:08x} at {lo:#x}, from lr={:#010x}",
        if half { "F16" } else { "F32" },
        ctx.read_u32(lo),
        ctx.regs[14]
    );
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
pub(super) const TYPE_CUBE_ARBITRARY: u32 = 7; // 0xE000_0000 >> 29

/// void sceGxmSetFragmentTexture(context, unsigned int textureIndex, const
///     SceGxmTexture *texture)
pub(super) fn set_fragment_texture(ctx: &mut GuestCtx, st: &mut VitaState) {
    let unit = ctx.arg(1);
    let texture = ctx.arg(2);
    // Whether the control words are already zero AT BIND TIME. A texture that is live here and
    // zero at draw time is a LIFETIME problem (the guest reused or cleared the struct, or the
    // binding went stale); one that is zero here was never a texture at this address at all,
    // and the address itself is what is wrong. The two need opposite investigations.
    let live_at_bind = texture != 0
        && (ctx.read_u32(texture)
            | ctx.read_u32(texture + 4)
            | ctx.read_u32(texture + 8)
            | ctx.read_u32(texture + 12))
            != 0;
    // A non-null handle whose control words are all zero is the guest handing GXM a texture it
    // never initialised. Name the CALL SITE the first time each address does it: the binding
    // itself says nothing about why, and the caller is the only thing that can (see
    // `vitaslop-re-undocumented-nid-from-callsite`).
    if texture != 0 && !live_at_bind {
        use std::collections::HashSet;
        use std::sync::Mutex;
        static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
        let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if g.get_or_insert_with(HashSet::new).insert(texture) {
            eprintln!(
                "gxm texture: sceGxmSetFragmentTexture(unit {unit}, {texture:#x}) with ALL-ZERO \
                 control words, called from lr={:#010x}",
                ctx.regs[14]
            );
        }
    }
    st.bind_fragment_texture(ctx, unit, texture);
    st.note_direct_texture_bind();
    st.note_texture_live_at_bind(texture, live_at_bind);
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
    // The BYTE STRIDE stays in the host shadow. It is the one field of word 0 that cannot move
    // into the guest's words: a `LINEAR_STRIDED` texture spreads its stride over `stride_ext`,
    // `stride_low` and `stride`, and how those three COMPOSE is not published - the header names
    // them "stride extension" and "internal stride lower bits" and stops there. Packing it would
    // mean inventing the composition, so `sceGxmTextureGetStride` still reads what the guest
    // passed. `mipCount` DOES move, for every layout that has the field.
    let mip_or_stride = ctx.arg(5);
    let strided = type_field == TYPE_LINEAR_STRIDED;
    if strided {
        st.set_texture_stride(texture, mip_or_stride);
    }

    write_texture_control_words(
        ctx,
        texture,
        type_field,
        (tex_format >> 24) & 0xff,
        (tex_format >> 12) & 0x7,
        width,
        height,
        data,
        (!strided).then_some(mip_or_stride),
    );
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
    // Pace the caller to the scanout. On hardware this call is where a title waits for
    // the display, and without the wait the frame rate is "however fast the guest can
    // draw" - 216 fps on this title's front end against a 60 Hz panel. See
    // [`VitaState::pace_flip`]. Only in the preemptive model: a run-to-completion host
    // has no scheduler to park against and treats a flip as a plain continue.
    if st.is_preemptive() {
        st.pace_flip(FRAME_US);
    }
    ctx.ret(0);
}

/// One display period: the Vita panel is 60 Hz.
const FRAME_US: u64 = 1_000_000 / 60;

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
    report_region_clip_ignored(mode, [x_min, y_min, x_max, y_max]);
    0
}

/// Say so - once per distinct clip, unconditionally - that the guest set a REGION CLIP and no
/// renderer here consumes it.
///
/// `SceGxmRegionClip` is the hardware scissor. It is captured into
/// [`crate::capture::RenderState`] and then read by nobody: there is no `set_scissor` anywhere
/// in this project. That is invisible until a title uses the idiom where it matters - draw an
/// oversized triangle covering the whole viewport and SCISSOR it down to the rectangle you
/// actually want - at which point ignoring the scissor turns a small rectangle into a
/// fullscreen one. A black fullscreen triangle drawn that way covers the finished frame, which
/// is a black screen with the UI on top and no error anywhere.
///
/// Mode 0 is `SCE_GXM_REGION_CLIP_NONE` and is not worth reporting; it is the default and it
/// asks for nothing.
fn report_region_clip_ignored(mode: u32, rect: [u32; 4]) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    if mode == 0 {
        return;
    }
    static SEEN: Mutex<Option<HashSet<(u32, [u32; 4])>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((mode, rect)) {
        return;
    }
    eprintln!(
        "gxm: sceGxmSetRegionClip(mode={mode}, {},{} .. {},{}) - the guest asked for a SCISSOR \
         and NO renderer here applies one, so every draw under it rasterises over the whole \
         target. A title that draws an oversized triangle and scissors it to the rectangle it \
         wants gets a FULLSCREEN one instead.",
        rect[0], rect[1], rect[2], rect[3]
    );
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
    let mut s = read_color_surface(ctx, addr).or_else(|| st.color_surface(addr))?;
    // The gamma mode is sticky host-side state keyed by the SURFACE address, because the
    // 32-byte guest struct has nowhere to hold it. Merge it back in here so every consumer -
    // the getter, and the scene's render target - sees a complete surface.
    s.gamma = st.color_surface_gamma_mode(addr);
    Some(s)
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

/// The fields of `SceGxmTexture` CONTROL WORD 0, as `(byte offset, bit shift, mask)`.
///
/// Layout from vitasdk `gxm.h struct SceGxmTexture` (permissive, an approved reference). The
/// bitfields are declared LSB-first, so the shifts below are the running sum of the widths
/// above each one:
///
/// ```text
///  bit  0      unk0            bits 14-16  unk1
///  bits 1-2    stride_ext      bits 17-20  mip_count   (stride, if LINEAR_STRIDED)
///  bits 3-5    vaddr_mode      bits 21-26  lod_bias    (stride, if LINEAR_STRIDED)
///  bits 6-8    uaddr_mode      bits 27-28  gamma_mode
///  bit  9      mip_filter      bits 29-30  unk2        (stride_low, if LINEAR_STRIDED)
///  bits 10-11  min_filter      bit  31     format0
///  bits 12-13  mag_filter
/// ```
///
/// # Why these live in the guest's words and not in a host-side shadow
/// They used to live in a shadow map keyed by the texture struct's ADDRESS, on the stated
/// reasoning that a setter should not risk corrupting a struct the guest re-reads. That was
/// wrong in two ways that reinforce each other:
///
/// 1. **A `SceGxmTexture` is 16 bytes the guest owns and COPIES.** This codebase already knows
///    that - `sceGxmSetFragmentTexture` takes the words by value, and one title binds by-value
///    copies. Address-keyed state does not survive a copy, so a copied texture silently got
///    GXM defaults for its wrap modes and bias instead of the originals. A wrong wrap mode is
///    the "colours right, shapes wrong" failure, i.e. visible and hard to attribute.
/// 2. **A getter whose answer lives in host state cannot be inlined.** These are pure field
///    reads, which is exactly what [`vitaslop_transpiler::InlineOp::LoadShiftMask`] exists for,
///    and `sceGxmTextureGetLodBias` is the single hottest host call one title makes - 71,298 in
///    a profile window, 66,405 of them from one call site. Reading the guest's own word makes
///    it a load, a shift and a mask emitted straight into the guest code, crossing nothing.
///
/// The defaults are why the shadow worked for so long: GXM's defaults for every field here are
/// zero (REPEAT, POINT, no mip filter, bias 0, gamma off), and `write_texture_control_words`
/// leaves word 0 zero apart from the format bit - so the word and the shadow agreed on any
/// texture the guest never copied.
///
/// # What is NOT settled, stated rather than guessed
/// - `mip_count` is four bits and the header says only "Mip count". Whether the driver packs
///   the count or the count MINUS ONE (as it does for width and height, which this project
///   established from observed behaviour) is not stated by any clean source. It is also
///   unobservable through the API: every path that can see it goes through our own writer and
///   reader, and both choices satisfy `Set(n)` then `Get() == n` and survive a struct copy.
///   Stored raw, and this comment is the record of the ambiguity.
/// - `byte_stride` for a `LINEAR_STRIDED` texture is spread over `stride_ext`, `stride_low` and
///   `stride`, and how those three COMPOSE is not published ("stride extension", "internal
///   stride lower bits"). That one stays in the host shadow, because packing it would mean
///   inventing the composition - see [`VitaState::texture_stride`].
mod texword0 {
    /// `(shift, mask)` of each field of control word 0.
    pub const VADDR_MODE: (u32, u32) = (3, 0x7);
    pub const UADDR_MODE: (u32, u32) = (6, 0x7);
    pub const MIP_FILTER: (u32, u32) = (9, 0x1);
    pub const MIN_FILTER: (u32, u32) = (10, 0x3);
    pub const MAG_FILTER: (u32, u32) = (12, 0x3);
    pub const MIP_COUNT: (u32, u32) = (17, 0xf);
    pub const LOD_BIAS: (u32, u32) = (21, 0x3f);
    pub const GAMMA_MODE: (u32, u32) = (27, 0x3);
}

/// Read one field of a texture's control word 0.
fn tex_field(ctx: &GuestCtx, texture: u32, (shift, mask): (u32, u32)) -> u32 {
    (ctx.read_u32(texture) >> shift) & mask
}

/// Write one field of a texture's control word 0, leaving every other bit alone.
///
/// The value is MASKED, not asserted: the hardware field is this wide, so a caller passing a
/// wider value gets what the hardware would store. Faithfulness here is "the bits the hardware
/// keeps", not "reject what the guest asked for".
fn set_tex_field(ctx: &mut GuestCtx, texture: u32, (shift, mask): (u32, u32), value: u32) {
    let w0 = ctx.read_u32(texture);
    ctx.write_u32(texture, (w0 & !(mask << shift)) | ((value & mask) << shift));
}

/// int sceGxmTextureSetUAddrMode[Safe](SceGxmTexture *texture, SceGxmTextureAddrMode mode)
pub(super) fn texture_set_u_addr_mode(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let mode = ctx.arg(1);
    set_tex_field(ctx, texture, texword0::UADDR_MODE, mode);
    ctx.ret(0);
}

/// int sceGxmTextureSetVAddrMode[Safe](SceGxmTexture *texture, SceGxmTextureAddrMode mode)
pub(super) fn texture_set_v_addr_mode(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let mode = ctx.arg(1);
    set_tex_field(ctx, texture, texword0::VADDR_MODE, mode);
    ctx.ret(0);
}

/// int sceGxmTextureSetLodBias(SceGxmTexture *texture, unsigned int bias)
pub(super) fn texture_set_lod_bias(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let bias = ctx.arg(1);
    set_tex_field(ctx, texture, texword0::LOD_BIAS, bias);
    ctx.ret(0);
}

/// int sceGxmTextureSetMinFilter(SceGxmTexture *texture, SceGxmTextureFilter minFilter)
pub(super) fn texture_set_min_filter(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let filter = ctx.arg(1);
    set_tex_field(ctx, texture, texword0::MIN_FILTER, filter);
    ctx.ret(0);
}

/// int sceGxmTextureSetMagFilter(SceGxmTexture *texture, SceGxmTextureFilter magFilter)
pub(super) fn texture_set_mag_filter(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let filter = ctx.arg(1);
    set_tex_field(ctx, texture, texword0::MAG_FILTER, filter);
    ctx.ret(0);
}

/// int sceGxmTextureSetMipFilter(SceGxmTexture *texture, SceGxmTextureMipFilter mipFilter)
///
/// `SceGxmTextureMipFilter` is ALREADY SHIFTED into the control word
/// (`SCE_GXM_TEXTURE_MIP_FILTER_ENABLED = 0x00000200`, i.e. bit 9), unlike
/// `SceGxmTextureFilter`, whose values are a plain 0..3. Shifting a pre-shifted enum again
/// would mask 0x200 down to zero and store "disabled" for every call that asked for enabled -
/// silently, and visible only as absent mip filtering. The two neighbouring setters genuinely
/// do need the shift, which is what makes this worth spelling out.
pub(super) fn texture_set_mip_filter(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let filter = ctx.arg(1);
    let (shift, mask) = texword0::MIP_FILTER;
    set_tex_field(ctx, texture, texword0::MIP_FILTER, (filter >> shift) & mask);
    ctx.ret(0);
}

// NOTE `sceGxmTextureSetMipmapCount` and `sceGxmTextureGetMipFilter` exist in the API and are
// NOT implemented here, because no title in the corpus links them and this project does not
// hand-type a NID it has not verified against a real module. They are one line
// each over `texword0::MIP_COUNT` / `MIP_FILTER` the moment a title needs them, and until then
// an unregistered NID hard-fails at link, which is the correct outcome rather than a guess.

/// int sceGxmTextureSetGammaMode(SceGxmTexture *texture, SceGxmTextureGammaMode gammaMode)
///
/// The enum's values are already SHIFTED into place in the control word
/// (`SCE_GXM_TEXTURE_GAMMA_BGR = 0x08000000`, i.e. bit 27), so the argument is masked to the
/// field's two bits rather than shifted again.
pub(super) fn texture_set_gamma_mode(ctx: &mut GuestCtx, st: &mut VitaState) {
    let texture = ctx.arg(0);
    let gamma = ctx.arg(1);
    let (shift, mask) = texword0::GAMMA_MODE;
    set_tex_field(ctx, texture, texword0::GAMMA_MODE, (gamma >> shift) & mask);
    // The mode itself lives in the word now; this only reports the first one of a run, because a
    // gamma texture is sampled through an sRGB format.
    st.note_texture_gamma(texture, gamma);
    ctx.ret(0);
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
pub(super) fn texture_get_mipmap_count(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let v = tex_field(ctx, texture, texword0::MIP_COUNT);
    ctx.ret(v);
}

/// unsigned int sceGxmTextureGetLodBias(const SceGxmTexture *texture)
pub(super) fn texture_get_lod_bias(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let v = tex_field(ctx, texture, texword0::LOD_BIAS);
    ctx.ret(v);
}

/// SceGxmTextureAddrMode sceGxmTextureGetUAddrModeSafe(const SceGxmTexture *texture)
pub(super) fn texture_get_u_addr_mode(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let v = tex_field(ctx, texture, texword0::UADDR_MODE);
    ctx.ret(v);
}

/// SceGxmTextureAddrMode sceGxmTextureGetVAddrModeSafe(const SceGxmTexture *texture)
pub(super) fn texture_get_v_addr_mode(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let v = tex_field(ctx, texture, texword0::VADDR_MODE);
    ctx.ret(v);
}

/// SceGxmTextureFilter sceGxmTextureGetMinFilter(const SceGxmTexture *texture)
pub(super) fn texture_get_min_filter(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let v = tex_field(ctx, texture, texword0::MIN_FILTER);
    ctx.ret(v);
}

/// SceGxmTextureFilter sceGxmTextureGetMagFilter(const SceGxmTexture *texture)
pub(super) fn texture_get_mag_filter(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let v = tex_field(ctx, texture, texword0::MAG_FILTER);
    ctx.ret(v);
}

/// SceGxmTextureGammaMode sceGxmTextureGetGammaMode(const SceGxmTexture *texture)
///
/// `SceGxmTextureGammaMode`'s values are already in control-word position
/// (`GAMMA_R = 0x08000000`), so the answer is word 0 masked to the field, UNSHIFTED - the same
/// shape `sceGxmTextureGetMipFilter` would take. See [`texture_set_mip_filter`] for why the
/// neighbouring filter getters are different.
pub(super) fn texture_get_gamma_mode(ctx: &mut GuestCtx) {
    let texture = ctx.arg(0);
    let (shift, mask) = texword0::GAMMA_MODE;
    let v = ctx.read_u32(texture) & (mask << shift);
    ctx.ret(v);
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
    let data = resolve_color_surface(ctx, st, surface).map(|s| s.data_addr).unwrap_or(0);
    // A getter that answers zero for a surface the title is about to build a texture from makes
    // the title skip the build and bind an uninitialised `SceGxmTexture` - which then samples
    // as a 1x1 zero texel and blacks out whatever pass reads it, a long way from here. Say it
    // once per surface, because "the guest bound a null texture" and "we told the guest its
    // render target has no pixels" are the same event seen from two ends.
    if data == 0 {
        use std::collections::HashSet;
        use std::sync::Mutex;
        static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
        let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if g.get_or_insert_with(HashSet::new).insert(surface) {
            eprintln!(
                "gxm surface: sceGxmColorSurfaceGetData({surface:#x}) answered NULL - no colour                  surface is recorded at that address, called from lr={:#010x}",
                ctx.regs[14]
            );
        }
    }
    data
}

/// unsigned int sceGxmColorSurfaceGetStrideInPixels(const SceGxmColorSurface *surface)
#[hostcall]
pub(super) fn color_surface_get_stride_in_pixels(ctx: &mut GuestCtx, st: &mut VitaState, surface: u32) -> u32 {
    resolve_color_surface(ctx, st, surface).map(|s| s.stride_pixels).unwrap_or(0)
}

/// int sceGxmColorSurfaceSetGammaMode(SceGxmColorSurface *surface, SceGxmColorSurfaceGammaMode gammaMode)
#[hostcall]
pub(super) fn color_surface_set_gamma_mode(ctx: &mut GuestCtx, st: &mut VitaState, surface: u32, gamma: u32) -> i32 {
    st.set_color_surface_gamma(surface, gamma);
    // Write it into the guest-visible surface struct too, so a scene that resolves its target
    // through `read_color_surface` carries the mode with it. Keeping the mode only in a side
    // table keyed by the SURFACE address loses it the moment the scene is described by its
    // colour surface's CONTENTS - which is how the renderer sees it.
    if let Some(s) = read_color_surface(ctx, surface) {
        // Name the DATA address, not the surface struct: the renderer, the chain dump and every
        // diagnostic downstream identify a pass by where its pixels land.
        tracing::debug!(
            target: "vitaslop::gxm",
            surface = format_args!("{surface:#x}"),
            data = format_args!("{:#x}", s.data_addr),
            gamma = format_args!("{gamma:#x}"),
            size = format_args!("{}x{}", s.width, s.height),
            "colorSurfaceSetGammaMode"
        );
        eprintln!(
            "gxm surface: GAMMA-CORRECT writes on the surface at data {:#x} ({}x{}), mode              {gamma:#x}",
            s.data_addr, s.width, s.height
        );
        st.set_color_surface(surface, s);
    }
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
pub(super) fn precomputed_vertex_state_set_texture(ctx: &mut GuestCtx, st: &mut VitaState, state: u32, index: u32, texture: u32) -> i32 {
    st.precomputed_vertex_state_set_texture(ctx, state, index, texture);
    0
}

/// int sceGxmPrecomputedFragmentStateSetTexture(state, unsigned int textureIndex,
///     const SceGxmTexture *texture)
#[hostcall]
pub(super) fn precomputed_fragment_state_set_texture(ctx: &mut GuestCtx, st: &mut VitaState, state: u32, index: u32, texture: u32) -> i32 {
    st.precomputed_fragment_state_set_texture(ctx, state, index, texture);
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

// --- Depth/stencil surface ---------------------------------------------------
//
// Unlike a colour surface (whose emit words are opaque hardware state, so we keep our
// own tagged mirror), `SceGxmDepthStencilSurface`'s layout IS published in vitasdk
// `gxm.h`: `{ +0x00 zlsControl, +0x04 depthData, +0x08 stencilData, +0x0c
// backgroundDepth (float), +0x10 backgroundControl }`, 0x14 bytes. So these calls write
// the real fields, and a copy of the struct carries its own state with no side table.
//
// Only `zlsControl`'s bit layout is unpublished, and the two force-mode enums give us
// exactly the bits we need: GXM's enums are the raw register bits throughout (a depth
// func is 0x00C00000, a texture type is 0x60000000), and FORCE_LOAD_ENABLED is 0x2 with
// FORCE_STORE_ENABLED 0x4 - adjacent single bits, in the word that controls the Z Load
// Store unit. They are therefore written and read back in place.

const DS_ZLS_CONTROL: u32 = 0x00;
const DS_DEPTH_DATA: u32 = 0x04;
const DS_STENCIL_DATA: u32 = 0x08;
const DS_BACKGROUND_DEPTH: u32 = 0x0c;
const DS_BACKGROUND_CONTROL: u32 = 0x10;
/// `SceGxmDepthStencilForceLoadMode` / `ForceStoreMode` occupy these bits of `zlsControl`.
const DS_FORCE_LOAD_MASK: u32 = 0x0000_0002;
const DS_FORCE_STORE_MASK: u32 = 0x0000_0004;

/// int sceGxmDepthStencilSurfaceInit(SceGxmDepthStencilSurface *surface,
///     SceGxmDepthStencilFormat depthStencilFormat, SceGxmDepthStencilSurfaceType
///     surfaceType, unsigned int strideInSamples, void *depthData, void *stencilData)
///
/// Fills the published fields, with the GXM defaults for the two background values (a
/// depth clear of 1.0 - the far plane - and no background stencil). `format`,
/// `surfaceType` and `strideInSamples` belong to `zlsControl`, whose packing is not
/// published; they are folded in as the enum words they already are, which keeps the
/// force bits (the part that IS specified) exact. Nothing here reads that word back, and
/// every `Get` that would expose the packing is an unimplemented NID that hard-fails, so
/// a wrong guess cannot be observed as a wrong ANSWER - only as a loud missing call.
pub(super) fn depth_stencil_surface_init(ctx: &mut GuestCtx, _st: &mut VitaState) {
    let surface = ctx.arg(0);
    let format = ctx.arg(1);
    let surface_type = ctx.arg(2);
    let stride_in_samples = ctx.arg(3);
    let depth_data = ctx.arg(4);
    let stencil_data = ctx.arg(5);
    let zls = format | surface_type | (stride_in_samples & !(DS_FORCE_LOAD_MASK | DS_FORCE_STORE_MASK));
    ctx.write_u32(surface + DS_ZLS_CONTROL, zls);
    ctx.write_u32(surface + DS_DEPTH_DATA, depth_data);
    ctx.write_u32(surface + DS_STENCIL_DATA, stencil_data);
    ctx.write_u32(surface + DS_BACKGROUND_DEPTH, 1.0f32.to_bits());
    ctx.write_u32(surface + DS_BACKGROUND_CONTROL, 0);
    ctx.ret(0);
}

/// void sceGxmDepthStencilSurfaceSetBackgroundDepth(SceGxmDepthStencilSurface *surface,
///     float backgroundDepth)
#[hostcall]
pub(super) fn depth_stencil_surface_set_background_depth(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    surface: u32,
    background_depth: f32,
) -> i32 {
    ctx.write_u32(surface + DS_BACKGROUND_DEPTH, background_depth.to_bits());
    0
}

/// void sceGxmDepthStencilSurfaceSetBackgroundStencil(SceGxmDepthStencilSurface
///     *surface, unsigned char backgroundStencil)
///
/// The value lands in the low byte of `backgroundControl`. That byte position is an
/// inference (the struct field is published, its bit layout is not), but a stencil value
/// is eight bits wide and this word exists to carry it; the matching getter reads the
/// same byte, so a title that sets and reads back sees exactly what it wrote.
#[hostcall]
pub(super) fn depth_stencil_surface_set_background_stencil(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    surface: u32,
    background_stencil: u32,
) -> i32 {
    let w = ctx.read_u32(surface + DS_BACKGROUND_CONTROL);
    ctx.write_u32(surface + DS_BACKGROUND_CONTROL, (w & !0xff) | (background_stencil & 0xff));
    0
}

/// void sceGxmDepthStencilSurfaceSetForceLoadMode(SceGxmDepthStencilSurface *surface,
///     SceGxmDepthStencilForceLoadMode forceLoad)
#[hostcall]
pub(super) fn depth_stencil_surface_set_force_load_mode(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    surface: u32,
    force_load: u32,
) -> i32 {
    let w = ctx.read_u32(surface + DS_ZLS_CONTROL);
    ctx.write_u32(surface + DS_ZLS_CONTROL, (w & !DS_FORCE_LOAD_MASK) | (force_load & DS_FORCE_LOAD_MASK));
    0
}

/// void sceGxmDepthStencilSurfaceSetForceStoreMode(SceGxmDepthStencilSurface *surface,
///     SceGxmDepthStencilForceStoreMode forceStore)
#[hostcall]
pub(super) fn depth_stencil_surface_set_force_store_mode(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    surface: u32,
    force_store: u32,
) -> i32 {
    let w = ctx.read_u32(surface + DS_ZLS_CONTROL);
    ctx.write_u32(surface + DS_ZLS_CONTROL, (w & !DS_FORCE_STORE_MASK) | (force_store & DS_FORCE_STORE_MASK));
    0
}

// --- Back-face render state --------------------------------------------------

/// void sceGxmSetBackDepthWriteEnable(SceGxmContext *context, SceGxmDepthWriteMode enable)
#[hostcall]
pub(super) fn set_back_depth_write_enable(st: &mut VitaState, _context: u32, enable: u32) -> i32 {
    st.render_state_mut().back_depth_write = enable;
    0
}

/// void sceGxmSetBackPolygonMode(SceGxmContext *context, SceGxmPolygonMode mode)
#[hostcall]
pub(super) fn set_back_polygon_mode(st: &mut VitaState, _context: u32, mode: u32) -> i32 {
    st.render_state_mut().back_polygon_mode = mode;
    0
}

// --- Occlusion queries -------------------------------------------------------

/// int sceGxmSetVisibilityBuffer(SceGxmContext *context, void *bufferBase,
///     unsigned int stridePerCore)
#[hostcall]
pub(super) fn set_visibility_buffer(st: &mut VitaState, _context: u32, base: u32, stride_per_core: u32) -> i32 {
    st.set_visibility_buffer(base, stride_per_core);
    0
}

/// void sceGxmSetFrontVisibilityTestEnable(SceGxmContext *context,
///     SceGxmVisibilityTestMode enable)
#[hostcall]
pub(super) fn set_front_visibility_test_enable(st: &mut VitaState, _context: u32, enable: u32) -> i32 {
    st.render_state_mut().front_visibility_test_enable = enable;
    0
}

/// void sceGxmSetFrontVisibilityTestIndex(SceGxmContext *context, unsigned int index)
#[hostcall]
pub(super) fn set_front_visibility_test_index(st: &mut VitaState, _context: u32, index: u32) -> i32 {
    st.render_state_mut().front_visibility_test_index = index;
    0
}

/// void sceGxmSetFrontVisibilityTestOp(SceGxmContext *context, SceGxmVisibilityTestOp op)
#[hostcall]
pub(super) fn set_front_visibility_test_op(st: &mut VitaState, _context: u32, op: u32) -> i32 {
    st.render_state_mut().front_visibility_test_op = op;
    0
}

// --- Unmapping ---------------------------------------------------------------

/// int sceGxmUnmapMemory(void *base) / sceGxmUnmapVertexUsseMemory(void *base) /
/// sceGxmUnmapFragmentUsseMemory(void *base)
///
/// The guest's pages already ARE the memory the capture reads, so mapping is a no-op -
/// but unmapping is not, because the guest may now reuse those pages for anything, and a
/// texture snapshot cached against them would be sampled as if it were still a texture.
#[hostcall]
pub(super) fn unmap_memory(st: &mut VitaState, base: u32) -> i32 {
    st.gxm_unmap(base)
}

// --- Colour surface: scale mode + data rebind --------------------------------

/// SceGxmColorSurfaceScaleMode sceGxmColorSurfaceGetScaleMode(const SceGxmColorSurface *surface)
#[hostcall]
pub(super) fn color_surface_get_scale_mode(ctx: &mut GuestCtx, st: &mut VitaState, surface: u32) -> u32 {
    resolve_color_surface(ctx, st, surface).map(|s| s.scale_mode).unwrap_or(0)
}

/// int sceGxmColorSurfaceSetData(SceGxmColorSurface *surface, void *data)
///
/// Rebinds where the surface renders to. Written into BOTH the guest struct (so a copy
/// of the surface resolves to the new address) and the address table, because a scene
/// begun after this must be captured against the new buffer - a title double-buffers by
/// calling exactly this between frames, and missing it renders every frame into one.
#[hostcall]
pub(super) fn color_surface_set_data(ctx: &mut GuestCtx, st: &mut VitaState, surface: u32, data: u32) -> i32 {
    match resolve_color_surface(ctx, st, surface) {
        Some(mut s) => {
            s.data_addr = data;
            write_color_surface(ctx, surface, &s);
            st.set_color_surface(surface, s);
            0
        }
        None => {
            tracing::warn!(
                target: "vitaslop::gxm",
                surface = format_args!("{surface:#x}"),
                data = format_args!("{data:#x}"),
                "colorSurfaceSetData on a surface never initialised here - the new render \
                 target is NOT recorded"
            );
            // SCE_GXM_ERROR_INVALID_VALUE.
            0x8021_0000u32 as i32
        }
    }
}

// --- Program reflection: type + find-by-semantic ------------------------------

/// SceGxmProgramType sceGxmProgramGetType(const SceGxmProgram *program)
///
/// Bit 0 of the byte at header +0x14 selects the stage (set = fragment). This is the
/// same field, at the same offset, that the clean-room GXP container parser keys its
/// own vertex/fragment decision off, so the two cannot disagree.
#[hostcall]
pub(super) fn program_get_type(ctx: &mut GuestCtx, _st: &mut VitaState, program: u32) -> u32 {
    // SCE_GXM_VERTEX_PROGRAM = 0, SCE_GXM_FRAGMENT_PROGRAM = 1.
    (ctx.read_u32(program.wrapping_add(0x14)) & 1) as u32
}

/// const SceGxmProgramParameter *_sceGxmProgramFindParameterBySemantic(
///     const SceGxmProgram *program, SceGxmParameterSemantic semantic, unsigned int index)
///
/// The counterpart of `sceGxmProgramFindParameterByName` for a title that builds its
/// vertex-attribute array from semantics rather than names. Walks the same parameter
/// table and returns the first entry whose packed semantic word matches - the semantic
/// in the LOW byte, its index in the high one (see [`param_get_semantic`]). Null when
/// nothing matches, which is what the API returns and what a caller tests for.
#[hostcall]
pub(super) fn find_parameter_by_semantic(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    program: u32,
    semantic: u32,
    index: u32,
) -> u32 {
    let count = ctx.read_u32(program.wrapping_add(0x24));
    let base = program.wrapping_add(0x28).wrapping_add(ctx.read_u32(program.wrapping_add(0x28)));
    let want = (semantic & 0xff) | ((index & 0xff) << 8);
    (0..count)
        .map(|i| base.wrapping_add(i.wrapping_mul(16)))
        .find(|&p| (ctx.read_u32(p.wrapping_add(4)) >> 16) & 0xffff == want)
        .unwrap_or(0)
}

/// int sceGxmRenderTargetGetDriverMemBlock(const SceGxmRenderTarget *renderTarget,
///     SceUID *driverMemBlock)
///
/// Hands back the UID the guest supplied in `SceGxmRenderTargetParams::driverMemBlock`
/// (+0x10 of that struct), so a title that frees the block it allocated frees the right
/// one. A target created with `SCE_UID_INVALID_UID` (sceGxm allocates its own) reads
/// back exactly that, which is the signal not to free anything.
#[hostcall]
pub(super) fn render_target_get_driver_mem_block(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    render_target: u32,
    out: Ptr,
) -> i32 {
    if out.is_null() {
        // SCE_GXM_ERROR_INVALID_POINTER.
        0x8021_0004u32 as i32
    } else {
        ctx.write_u32(out.addr(), st.render_target_mem_block(render_target));
        0
    }
}

// --- Vertex-stage textures, cube-arbitrary init, palettes ---------------------

/// int _sceGxmSetVertexTexture(SceGxmContext *context, unsigned int textureIndex,
///     const SceGxmTexture *texture)
///
/// Binds a texture to a VERTEX-stage sampler (vertex texture fetch: displacement maps,
/// per-instance data tables, a vector canvas whose control points ARE texels).
///
/// The binding is recorded and travels with the draw in its own list, decoded by the same
/// `snapshot_bound_textures` the fragment stage uses so the two stages can never decode one
/// texture differently. It is a separate list because the two stages number their sampler
/// units INDEPENDENTLY - binding the fragment stage's texture here does not shade a surface
/// wrongly, it draws a different mesh.
pub(super) fn set_vertex_texture(ctx: &mut GuestCtx, st: &mut VitaState) {
    let unit = ctx.arg(1);
    let texture = ctx.arg(2);
    st.bind_vertex_texture(ctx, unit, texture);
    ctx.ret(0);
}

/// int sceGxmTextureSetPalette(SceGxmTexture *texture, const void *paletteData)
///
/// Points a paletted (P8/P4) texture at its colour table. The palette's position within
/// the 16-byte control words is not published, so it is kept beside them rather than
/// packed into a field whose neighbours ARE understood - guessing the packing would
/// corrupt the format and dimension fields that decode correctly today.
///
/// The capture's sampler does not expand palette indices to colours, so a paletted
/// texture still samples its INDEX as if it were a value. That is a real gap, and it
/// says so once rather than rendering wrong quietly.
#[hostcall]
pub(super) fn texture_set_palette(st: &mut VitaState, texture: u32, palette: u32) -> i32 {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            target: "vitaslop::gxm",
            texture = format_args!("{texture:#x}"),
            palette = format_args!("{palette:#x}"),
            "a texture palette was bound; the capture samples paletted formats as raw \
             indices, so this texture's colours are wrong until palette expansion lands"
        );
    }
    st.set_texture_palette(texture, palette);
    0
}

// --- Precomputed: whole-array setters and non-default uniform buffers ---------

/// int sceGxmPrecomputedDrawSetAllVertexStreams(SceGxmPrecomputedDraw *precomputedDraw,
///     const void *const *streamDataArray)
pub(super) fn precomputed_draw_set_all_vertex_streams(ctx: &mut GuestCtx, st: &mut VitaState) {
    let precomputed = ctx.arg(0);
    let array = ctx.arg(1);
    st.precomputed_draw_set_all_streams(ctx, precomputed, array);
    ctx.ret(0);
}

/// int sceGxmPrecomputedFragmentStateSetAllTextures(SceGxmPrecomputedFragmentState
///     *precomputedState, const SceGxmTexture *textureArray)
#[hostcall]
pub(super) fn precomputed_fragment_state_set_all_textures(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    state: u32,
    array: u32,
) -> i32 {
    st.precomputed_fragment_state_set_all_textures(ctx, state, array);
    0
}

/// int sceGxmPrecomputedVertexStateSetAllTextures(SceGxmPrecomputedVertexState
///     *precomputedState, const SceGxmTexture *textures)
#[hostcall]
pub(super) fn precomputed_vertex_state_set_all_textures(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    state: u32,
    array: u32,
) -> i32 {
    st.precomputed_vertex_state_set_all_textures(ctx, state, array);
    0
}

/// int sceGxmPrecomputed{Vertex,Fragment}StateSetUniformBuffer(state,
///     unsigned int bufferIndex, const void *bufferData)
/// int sceGxmPrecomputed{Vertex,Fragment}StateSetAllUniformBuffers(state,
///     const void *const *bufferDataArray)
///
/// These bind NON-default uniform buffers into a precomputed state - the same thing
/// `sceGxmSetVertexUniformBuffer` does on the direct path, and with the same limit: a
/// draw carries only the DEFAULT uniform buffer, so a shader reading buffer index N
/// reads nothing. Recording the pointer would not change that, so what matters is that
/// the gap is stated rather than the call quietly succeeding.
pub(super) fn precomputed_state_set_uniform_buffer(ctx: &mut GuestCtx, stage: &'static str, all: bool) {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            target: "vitaslop::gxm",
            stage,
            all,
            state = format_args!("{:#x}", ctx.arg(0)),
            "a precomputed state bound a non-default uniform buffer; the capture records \
             only the DEFAULT uniform buffer, so a shader reading that buffer index gets \
             nothing"
        );
    }
    ctx.ret(0);
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

    /// Guest address the synthetic `SceGxmProgram` header sits at. Far enough from
    /// [`PARAM`] that the two fixtures cannot overlap at any offset either one reads.
    const PROGRAM: u32 = 0x200;

    /// The program header fields the inline forms read, by byte offset. Only the read
    /// fields are set; everything else stays zero, which is what a header we failed to
    /// resolve looks like.
    fn program_header() -> Vec<(u32, u32)> {
        vec![
            (GXP_PARAM_COUNT_OFF, 11),
            // Comfortably under the clamp, so this is the INLINE case. The clamped case
            // is checked separately - it is the one the inline form must refuse.
            (GXP_DEFAULT_UNIFORM_BUFFER_COUNT_OFF, 37),
        ]
    }

    /// A guest image holding BOTH fixtures: the parameter record at [`PARAM`] and the
    /// program header at [`PROGRAM`].
    ///
    /// One image rather than two, because an inline form is defined by (pointer argument,
    /// offset) and nothing else - which structure the pointer names is not something the
    /// lowering knows or needs to. Keeping both in one image lets the same check cover a
    /// parameter getter and a program getter without a second harness.
    fn fixture_image() -> Vec<u8> {
        let mut bytes = vec![0u8; 4096];
        for (i, w) in param_record().iter().enumerate() {
            let off = PARAM as usize + i * 4;
            bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }
        for (off, w) in program_header() {
            let at = PROGRAM as usize + off as usize;
            bytes[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
        bytes
    }

    /// Run a NID through the real dispatch over the fixture image, with `ptr` in r0, and
    /// return the r0 the guest would see.
    fn handler_result_at(func_nid: u32, ptr: u32) -> u32 {
        let mut regs = [0u32; REG_COUNT];
        regs[0] = ptr;
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = fixture_image();
        let mut st = VitaState::new(0, 4096, Box::new(DeterministicWorld::default()));
        let mut mem = SliceMemory(&mut bytes);
        let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        super::super::dispatch(crate::nid::lib::SCE_GXM, func_nid, &mut ctx, &mut st);
        regs[0]
    }

    /// The word at `ptr + offset` in the fixture image - the word the inline form reads.
    fn fixture_word(ptr: u32, offset: u32) -> u32 {
        let bytes = fixture_image();
        let at = (ptr + offset) as usize;
        u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes"))
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
            let offset = op.offset().expect("a GXM getter reads through its pointer argument");
            assert_eq!(
                op.eval(word_at(offset)),
                handler_result(func_nid),
                "inline form of {} disagrees with its handler",
                crate::nid::name(func_nid)
            );
        }
    }

    /// The same equivalence for the getters handed a PROGRAM rather than a parameter
    /// record. Split from the test above only because the pointer differs; the obligation
    /// is identical, and `every_inlined_nid_is_covered` holds both lists together.
    #[test]
    fn program_inline_ops_match_their_handlers() {
        for func_nid in COVERED_PROGRAM {
            let op = inline_op(func_nid).expect("listed NID has an inline form");
            let offset = op.offset().expect("a program getter reads through its pointer argument");
            let word = fixture_word(PROGRAM, offset);
            assert!(
                !op.falls_back(word),
                "{} fixture must exercise the INLINE arm",
                crate::nid::name(func_nid)
            );
            assert_eq!(
                op.eval(word),
                handler_result_at(func_nid, PROGRAM),
                "inline form of {} disagrees with its handler",
                crate::nid::name(func_nid)
            );
        }
    }

    /// The clamped default-uniform-buffer case must NOT be answered inline.
    ///
    /// This is the whole reason `LoadScaled` carries a value guard. Inline, an unresolved
    /// header's count would be shifted straight into a huge size; the handler clamps it.
    /// A title uses this number as the length of its own uniform upload, so a wrong one
    /// does not fail - it truncates or overruns the block, far from here.
    #[test]
    fn an_unresolved_uniform_buffer_count_falls_back_to_the_handler() {
        let op = inline_op(g::PROGRAM_GET_DEFAULT_UNIFORM_BUFFER_SIZE).expect("has an inline form");
        assert!(
            op.falls_back(DEFAULT_UNIFORM_BUFFER_MAX_WORDS + 1),
            "a count past the clamp must reach the handler"
        );
        assert!(
            !op.falls_back(DEFAULT_UNIFORM_BUFFER_MAX_WORDS),
            "the clamp itself is still answerable inline"
        );
        // ...and where it IS answerable, the two agree at the boundary.
        assert_eq!(
            op.eval(DEFAULT_UNIFORM_BUFFER_MAX_WORDS),
            DEFAULT_UNIFORM_BUFFER_MAX_WORDS * 4,
            "the scale is four bytes per SA register"
        );
    }

    /// The NIDs the parameter-record test checks.
    const COVERED: [u32; 6] = [
        g::PROGRAM_PARAMETER_GET_CATEGORY,
        g::PROGRAM_PARAMETER_GET_TYPE,
        g::PROGRAM_PARAMETER_GET_COMPONENT_COUNT,
        g::PROGRAM_PARAMETER_GET_CONTAINER_INDEX,
        g::PROGRAM_PARAMETER_GET_ARRAY_SIZE,
        g::PROGRAM_PARAMETER_GET_RESOURCE_INDEX,
    ];

    /// The NIDs the program-header test checks.
    const COVERED_PROGRAM: [u32; 2] =
        [g::PROGRAM_GET_PARAMETER_COUNT, g::PROGRAM_GET_DEFAULT_UNIFORM_BUFFER_SIZE];

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
        for &nid in COVERED.iter().chain(COVERED_PROGRAM.iter()) {
            assert!(
                inline_op(nid).is_some(),
                "{} is listed as covered but has no inline form",
                crate::nid::name(nid)
            );
        }
        for nid in [
            g::DRAW,                       // records a whole draw into the scene
            g::END_SCENE,                  // completes and folds a frame
            g::SET_VERTEX_PROGRAM,         // updates the bound program
            g::PROGRAM_PARAMETER_GET_NAME, // returns a pointer, not a bitfield
            g::PROGRAM_GET_PARAMETER,      // computes an address from two reads
            // A string search over the whole parameter table: pure, but not ONE read.
            // Memoizing it was tried and reverted - see `find_parameter`.
            g::PROGRAM_FIND_PARAMETER_BY_NAME,
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
