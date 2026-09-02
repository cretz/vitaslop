//! SceGxm: the graphics API. These handlers hand back opaque handles for GXM
//! objects, remember the surfaces and vertex-program layouts the guest sets up,
//! and record the per-frame draw stream (BeginScene to EndScene) with the vertex,
//! index, and uniform data snapshotted from guest memory. No GPU is emulated and
//! no pixel is drawn here; that is the renderer's job over this capture.

use super::gxmctx;
use super::gxmprog;
use crate::capture::{ColorSurface, VertexAttribute};
use crate::host::{GuestCtx, VitaState, MAX_VERTEX_STREAMS};
use crate::render::f32_to_half;
use vitaslop_gxp_shader::ParamType;
use crate::hostcall;

/// SceGxmInitializeParams: `flags` at 0, `displayQueueMaxPendingCount` at 4,
/// `displayQueueCallback` at 8, its data size at 12.
const INIT_MAX_PENDING_OFFSET: u32 = 4;
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

/// Report - once per (target, mode) - the MULTISAMPLE mode a render target was created with.
///
/// `SceGxmMultisampleMode` is `NONE`/`2X`/`4X`, and 4X means the hardware keeps 2x2 samples per
/// pixel - which is why a 960x544 colour surface on this hardware carries a **1920x1088** depth
/// surface, a pairing that otherwise reads as a decode error.
///
/// This line reports the REQUEST only. It used to end "we rasterize it at ONE sample", which
/// stopped being true when the renderer began honouring the mode; the answer now belongs to
/// the renderer, which reports a grant per target and WARNS on a refusal. A diagnostic that
/// states someone else's behaviour goes stale the moment that behaviour changes, and this one
/// did.
/// The multisample mode a render target was created with, by handle. Recorded here rather than
/// in `VitaState` because it is read at `beginScene` to tell the renderer how many samples the
/// pass wants - and it has to be a RECORD and not a re-read of the params struct, which is a
/// caller stack frame that is gone by then.
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
             so on hardware its depth surface is {dw}x{dh} samples. This is the REQUEST; whether \
             a pass through this target got it is the renderer's own MULTISAMPLE granted/REFUSED \
             line, and a refusal is a warning"
        );
    }
}

/// int sceGxmCreateContext(const SceGxmContextParams *params, SceGxmContext **context)
///
/// The context IS the guest's `params->hostMem`. On hardware GXM builds its context inside
/// that buffer - the guest allocates it, passes it, and the returned `SceGxmContext *` points
/// into it - and every `sceGxmSet*` writes one field of the structure there. Putting the
/// sticky state where the hardware puts it is what lets those setters be INLINED into guest
/// code instead of crossing the host boundary 1,240 times a frame; see [`super::gxmctx`].
///
/// # Why this refuses rather than falls back
/// `hostMem` is not optional and `SCE_GXM_MINIMUM_CONTEXT_HOST_MEM_SIZE` is 2 KB, so a title
/// that passes null or a short buffer is one GXM itself would reject. Handing back an opaque
/// handle anyway would leave every later setter writing to an address that is not a context
/// and every draw reading a state nobody wrote - a wrong picture with no error, thousands of
/// frames from the cause.
pub(super) fn create_context(ctx: &mut GuestCtx, st: &mut VitaState) -> crate::SvcOutcome {
    let params = ctx.arg(0);
    let out = ctx.arg(1);
    let host_mem = ctx.read_u32(params + CTX_PARAMS_HOST_MEM);
    let host_mem_size = ctx.read_u32(params + CTX_PARAMS_HOST_MEM_SIZE);
    if host_mem == 0 || host_mem_size < gxmctx::BYTES {
        return crate::SvcOutcome::Fatal(format!(
            "sceGxmCreateContext with hostMem={host_mem:#x} size={host_mem_size} - the context \
             state lives in hostMem (as it does on hardware) and needs {} bytes of the {} GXM \
             itself requires",
            gxmctx::BYTES,
            gxmctx::MINIMUM_HOST_MEM
        ));
    }
    gxmctx::init(ctx, host_mem);
    st.adopt_gxm_context(host_mem);
    // The default-uniform ring is attached HERE rather than lazily at the first reserve,
    // because the emitted inline reserve reads a ring that is already there and hands the
    // call back to the host when it is not. Doing it once at create means the very first
    // reserve of a run is the only one that could ever have needed the host, and even that
    // one does not.
    st.attach_uniform_ring(ctx, host_mem);
    ctx.write_u32(out, host_mem);
    ctx.ret(0);
    crate::SvcOutcome::Continue
}

/// `SceGxmContextParams` field offsets (vitasdk `gxm.h`): `void *hostMem` then
/// `SceSize hostMemSize`.
const CTX_PARAMS_HOST_MEM: u32 = 0x00;
const CTX_PARAMS_HOST_MEM_SIZE: u32 = 0x04;

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
    // How many frames the title may have in flight. Ignoring it forced every title to a
    // depth of one, which costs a self-pacing title a whole display period per frame -
    // see `VitaState::pace_flip`.
    st.set_display_queue_depth(ctx.read_u32(params + INIT_MAX_PENDING_OFFSET));
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
    use gxmctx::off as ctxoff;
    use vitaslop_transpiler::InlineOp::{
        BindPrecomputedState, CopyArgIndexed, LoadScaled, LoadShiftMask, ReserveUniformBuffer,
        SetUniformData, StoreArg, StoreArgField, StoreArgFieldInPlace, StoreArgIndexed,
        StoreArgRun, StoreVfpRun,
    };
    // A `void sceGxmSet*(SceGxmContext *context, uint32 value)`: one word of the context
    // block, at the offset its handler writes.
    let store = |offset| StoreArg { offset };
    // The packed word's fields; `param_word` masks the word to 16 bits first, which
    // the 4-bit field masks below make redundant.
    let word = |shift| LoadShiftMask { offset: GXM_PARAM_WORD_OFF, shift, mask: 0xf, plus: 0 };
    // A field of a texture's CONTROL WORD 0, at the pointer itself (offset 0). Every one of
    // these is now a plain field of the guest's own struct rather than an entry in a host-side
    // map, which is what makes inlining them possible at all - see [`texword0`].
    let tex = |(shift, mask): (u32, u32)| LoadShiftMask { offset: 0, shift, mask, plus: 0 };
    // ...and for the two enums whose values are already IN control-word position, the answer is
    // the masked word with no shift, so the mask is the field in place. See
    // [`texture_get_mip_filter`].
    let tex_in_place =
        |(shift, mask): (u32, u32)| LoadShiftMask { offset: 0, shift: 0, mask: mask << shift, plus: 0 };
    // A SETTER of a control-word-0 field: the read-modify-write twin of `tex`, over the SAME
    // `(shift, mask)` pair, so a setter and its getter cannot disagree about where the field
    // is. Only the setters that store their argument AS PASSED belong here - see
    // [`texture_set_mip_filter`] for the one that does not, and why that matters.
    let tex_set = |(shift, mask): (u32, u32)| StoreArgField { offset: 0, shift, mask };
    // The setter twin of `tex_in_place`: for an enum whose values are ALREADY in control-word
    // position, the stored bits are the argument masked to the field WHERE IT IS. Given the
    // same `(shift, mask)` pair the getter uses, so the two still cannot disagree about the
    // field - the shift is folded into the mask here rather than applied to the value.
    let tex_set_in_place =
        |(shift, mask): (u32, u32)| StoreArgFieldInPlace { offset: 0, mask: mask << shift };
    // sceGxmTextureGetWidth/Height: the 12-bit SIZE-MINUS-ONE field of control word 1, plus
    // one - see [`texture_get_dim`], whose `+ 1` this `plus` is.
    let tex_dim = |shift| LoadShiftMask { offset: 4, shift, mask: 0xfff, plus: 1 };
    // A `sceGxmReserve*DefaultUniformBuffer(context, void **out)`: a bump of the ring in the
    // context block, recorded into one stage's three-word slot. Which stage is the only
    // difference between the two, and it is carried by the record's offset.
    let reserve = |record| ReserveUniformBuffer { layout: uniform_ring_layout(record) };
    Some(match func_nid {
        g::PROGRAM_PARAMETER_GET_CATEGORY => word(0),
        g::PROGRAM_PARAMETER_GET_TYPE => word(4),
        g::PROGRAM_PARAMETER_GET_COMPONENT_COUNT => word(8),
        g::PROGRAM_PARAMETER_GET_CONTAINER_INDEX => word(12),
        g::PROGRAM_PARAMETER_GET_ARRAY_SIZE => {
            LoadShiftMask { offset: GXM_PARAM_ARRAY_SIZE_OFF, shift: 0, mask: u32::MAX, plus: 0 }
        }
        g::PROGRAM_PARAMETER_GET_RESOURCE_INDEX => {
            LoadShiftMask { offset: GXM_PARAM_RESOURCE_INDEX_OFF, shift: 0, mask: u32::MAX, plus: 0 }
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
        // The three control-word-1 reads. `GetType` is the top three bits put back where they
        // came from (`((w1 >> 29) & 7) << 29`), which is the whole word masked - the same
        // shape `tex_in_place` has, one word over.
        g::TEXTURE_GET_TYPE => LoadShiftMask { offset: 4, shift: 0, mask: 0x7 << 29, plus: 0 },
        g::TEXTURE_GET_WIDTH => tex_dim(12),
        g::TEXTURE_GET_HEIGHT => tex_dim(0),
        // The control-word-0 SETTERS whose handler is `set_tex_field` and nothing else. Each
        // is a read-modify-write of ONE field, which is why they need their own form: a whole
        // word store would clear the seven settings packed beside the one being set, and the
        // result is a texture that samples wrongly rather than an error anyone can see.
        //
        // `...SetGammaMode` is deliberately absent: it also REPORTS to the host, and an inline
        // form would silently stop that happening.
        g::TEXTURE_SET_U_ADDR_MODE | g::TEXTURE_SET_U_ADDR_MODE_SAFE => {
            tex_set(texword0::UADDR_MODE)
        }
        g::TEXTURE_SET_V_ADDR_MODE | g::TEXTURE_SET_V_ADDR_MODE_SAFE => {
            tex_set(texword0::VADDR_MODE)
        }
        g::TEXTURE_SET_MIN_FILTER => tex_set(texword0::MIN_FILTER),
        g::TEXTURE_SET_MAG_FILTER => tex_set(texword0::MAG_FILTER),
        // The same shape as the four above and simply MISSED when they were listed - its
        // handler is `set_tex_field(.., LOD_BIAS, bias)` and nothing else, and its GETTER has
        // been inlined here since the block was written. MEASURED on one of them's
        // on-track run:
        // **465.6 calls a frame**, every one of them a boundary crossing to do a
        // read-modify-write of six bits the guest already owns.
        g::TEXTURE_SET_LOD_BIAS => tex_set(texword0::LOD_BIAS),
        // ...and the one whose enum is ALREADY in control-word position, so its handler masks
        // the argument in place rather than shifting it up. That is a different program, which
        // is why it could not use `tex_set` - see [`texture_set_mip_filter`], and
        // [`vitaslop_transpiler::InlineOp::StoreArgFieldInPlace`] for the form that matches it.
        // Another **465.6 calls a frame** on the same title, from the same loop.
        g::TEXTURE_SET_MIP_FILTER => tex_set_in_place(texword0::MIP_FILTER),
        // `sceGxmTextureSetData(texture, data)`: control word 2 is the data address with the
        // two low LOD bits packed under it, and the handler is exactly "keep the two low
        // bits, store the aligned pointer, return 0" - which IS the in-place field store,
        // with the field being every bit but those two. The argument arrives already in
        // field position (a pointer's low two bits are alignment, not payload), so the
        // in-place form is the right one for the same reason MIP_FILTER's is. MEASURED at
        // 32 calls per frame on a racing title's race - a texture data swap per animated
        // texture per frame - each one a full crossing to mask one word.
        g::TEXTURE_SET_DATA => StoreArgFieldInPlace { offset: 8, mask: 0xffff_fffc },
        // The two PROGRAM-pointer reads. Everything above is handed a parameter record;
        // these are handed the `SceGxmProgram` itself, which changes nothing about the
        // lowering - an inline form is defined by (pointer argument, offset), and which
        // structure the pointer names is not the emitter's business.
        //
        // Both are called per draw by a title that re-reflects its shader interface every
        // frame: 21,710 and 24,760 calls in one profile window.
        g::PROGRAM_GET_PARAMETER_COUNT => {
            LoadShiftMask { offset: GXP_PARAM_COUNT_OFF, shift: 0, mask: u32::MAX, plus: 0 }
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
        // The CONTEXT STATE SETTERS - the largest block of host calls a real title makes in
        // steady gameplay, and the reason `StoreArg` exists at all. Measured on a retail title:
        // the first five below are 248 calls a frame EACH, and eight draw-state calls share
        // the single call site every `sceGxmDrawPrecomputed` comes from - nine crossings per
        // draw, eight of them one-word writes into a structure the hardware keeps in guest
        // memory anyway.
        //
        // Each is `set(context, value)` and its handler is `gxmctx::set(...)` and nothing
        // else, which is exactly what `StoreArg` emits. See [`gxmctx`] for the layout these
        // offsets come from - they are the same constants the handlers use, not a second
        // copy, so the two cannot drift.
        g::SET_VERTEX_PROGRAM => store(ctxoff::VERTEX_PROGRAM),
        g::SET_FRAGMENT_PROGRAM => store(ctxoff::FRAGMENT_PROGRAM),
        g::SET_CULL_MODE => store(ctxoff::CULL_MODE),
        g::SET_FRONT_DEPTH_FUNC => store(ctxoff::FRONT_DEPTH_FUNC),
        g::SET_FRONT_DEPTH_WRITE_ENABLE => store(ctxoff::FRONT_DEPTH_WRITE),
        g::SET_TWO_SIDED_ENABLE => store(ctxoff::TWO_SIDED),
        g::SET_BACK_DEPTH_FUNC => store(ctxoff::BACK_DEPTH_FUNC),
        g::SET_BACK_DEPTH_WRITE_ENABLE => store(ctxoff::BACK_DEPTH_WRITE),
        g::SET_FRONT_FRAGMENT_PROGRAM_ENABLE => store(ctxoff::FRONT_FRAGMENT_PROGRAM_ENABLE),
        g::SET_BACK_FRAGMENT_PROGRAM_ENABLE => store(ctxoff::BACK_FRAGMENT_PROGRAM_ENABLE),
        g::SET_FRONT_POLYGON_MODE => store(ctxoff::FRONT_POLYGON_MODE),
        g::SET_BACK_POLYGON_MODE => store(ctxoff::BACK_POLYGON_MODE),
        g::SET_FRONT_POINT_LINE_WIDTH => store(ctxoff::FRONT_POINT_LINE_WIDTH),
        g::SET_FRONT_STENCIL_REF => store(ctxoff::FRONT_STENCIL_REF),
        g::SET_VIEWPORT_ENABLE => store(ctxoff::VIEWPORT_ENABLE),
        // `sceGxmSetViewport(context, 6 floats)`: the handler stores the six argument
        // floats' raw bits into six consecutive context words and returns 0, which is the
        // VFP run form exactly - the hardfloat AAPCS carries them in s0..s5. ~12 calls per
        // frame on a racing title's race, each a full crossing to move 24 bytes.
        g::SET_VIEWPORT => StoreVfpRun { offset: ctxoff::VIEWPORT, count: 6 },
        // `sceGxmSetRegionClip(context, mode, xMin, yMin, xMax, yMax)`: five argument words
        // stored as passed into the five consecutive words at REGION_CLIP_MODE (the mode,
        // then the four bounds at REGION_CLIP = REGION_CLIP_MODE + 4). The last two
        // arguments ride the guest stack, which the run form reads exactly where
        // `GuestCtx::arg` does. Another ~12 calls per frame from the same draw loop.
        g::SET_REGION_CLIP => StoreArgRun { offset: ctxoff::REGION_CLIP_MODE, count: 5 },
        // `sceGxmSetFrontDepthBias(context, factor, units)`: two argument words stored as
        // passed into two consecutive context words. The handler casts each `i32 as u32`,
        // which is the identity on the bits the register already holds, so the run form
        // stores exactly what it stores.
        //
        // MEASURED on a retail sports title, 100 display frames of live play: **271 calls a
        // frame**, the second-hottest host call it makes after `sceGxmDraw` itself, 18,971
        // of them from a single call site in its draw loop. On the browser a host call is
        // 91% marshalling ([[vitaslop-browser-host-call-cost]]), so this is 271 crossings a
        // frame to move eight bytes into the guest's own context block.
        g::SET_FRONT_DEPTH_BIAS => StoreArgRun { offset: ctxoff::FRONT_DEPTH_BIAS_FACTOR, count: 2 },
        // `sceGxmSet{Front,Back}StencilFunc(context, func, fail, depthFail, depthPass,
        // compareMask, writeMask)`: six argument words stored as passed into six consecutive
        // context words (the last three ride the guest stack, which the run form reads
        // exactly where `GuestCtx::arg` does). These used to sit on `NOT_INLINABLE` because
        // the handler masked the two `unsigned char` words; the narrowing moved to the
        // READ-BACK (`gxmctx::render_state`), so the handler now stores as passed and the
        // run form is exact. MEASURED on a retail sports title's round in a browser:
        // sceGxmSetFrontStencilFunc is its third-hottest host call, ~35 a frame.
        g::SET_FRONT_STENCIL_FUNC => StoreArgRun { offset: ctxoff::FRONT_STENCIL_FUNC, count: 6 },
        g::SET_BACK_STENCIL_FUNC => StoreArgRun { offset: ctxoff::BACK_STENCIL_FUNC, count: 6 },
        // `sceGxmSet{Vertex,Fragment}UniformBuffer(context, index, data)`: a bounded indexed
        // pointer store, exactly the `SET_VERTEX_STREAM` shape - in range the store IS the
        // whole call, and an index past `SCE_GXM_MAX_UNIFORM_BUFFERS` falls back to the
        // handler, which is where the report of it lives. MEASURED on the same round: the
        // vertex form is the second-hottest host call, ~46 a frame.
        g::SET_VERTEX_UNIFORM_BUFFER => StoreArgIndexed {
            offset: ctxoff::VERTEX_UNIFORM_BUFFERS,
            count: gxmctx::MAX_UNIFORM_BUFFERS as u32,
        },
        g::SET_FRAGMENT_UNIFORM_BUFFER => StoreArgIndexed {
            offset: ctxoff::FRAGMENT_UNIFORM_BUFFERS,
            count: gxmctx::MAX_UNIFORM_BUFFERS as u32,
        },
        // `sceGxmDepthStencilSurfaceSetForce{Load,Store}Mode(surface, mode)`: an in-place
        // masked field update of the surface's own `zlsControl` word - the argument is the
        // pre-positioned enum bit, so the store needs no shift, which is the
        // `StoreArgFieldInPlace` shape exactly (same as `sceGxmTextureSetData`). The
        // surface struct lives in guest memory, so the store IS the whole call.
        g::DEPTH_STENCIL_SURFACE_SET_FORCE_LOAD_MODE => StoreArgFieldInPlace {
            offset: DS_ZLS_CONTROL,
            mask: DS_FORCE_LOAD_MASK,
        },
        g::DEPTH_STENCIL_SURFACE_SET_FORCE_STORE_MODE => StoreArgFieldInPlace {
            offset: DS_ZLS_CONTROL,
            mask: DS_FORCE_STORE_MASK,
        },
        // The two per-draw-loop state binds: a copy between two guest structures now that
        // the precomputed state lives in guest memory (`vita::gxmstate`). 24 calls per
        // frame EACH on a racing title's race - the last non-draw GXM crossings it makes.
        g::SET_PRECOMPUTED_VERTEX_STATE => BindPrecomputedState { layout: bind_state_layout(false) },
        g::SET_PRECOMPUTED_FRAGMENT_STATE => BindPrecomputedState { layout: bind_state_layout(true) },
        g::SET_FRONT_VISIBILITY_TEST_ENABLE => store(ctxoff::FRONT_VISIBILITY_TEST_ENABLE),
        g::SET_FRONT_VISIBILITY_TEST_INDEX => store(ctxoff::FRONT_VISIBILITY_TEST_INDEX),
        g::SET_FRONT_VISIBILITY_TEST_OP => store(ctxoff::FRONT_VISIBILITY_TEST_OP),
        // The one INDEXED setter that qualifies: a stream binding is a plain pointer, so
        // storing it is the whole call. An index past the end still reaches the handler,
        // which is where the report of it lives.
        g::SET_VERTEX_STREAM => StoreArgIndexed {
            offset: ctxoff::STREAMS,
            count: gxmctx::MAX_VERTEX_STREAMS as u32,
        },
        // The one COPY: a fragment texture binding is the four control words as they read at
        // bind time, because GXM copies them by value. That is why this cannot be a store form
        // - see `CopyArgIndexed` - and it is the largest single block of host calls left in
        // steady gameplay, at 1,275 crossings a display frame over a live race.
        //
        // The handler still runs for every case it defines: an out-of-range unit, and a null
        // texture (which UNBINDS rather than copying).
        g::SET_FRAGMENT_TEXTURE => CopyArgIndexed {
            offset: ctxoff::TEXTURES,
            stride: gxmctx::TEXTURE_STRIDE,
            count: gxmctx::MAX_TEXTURE_UNITS as u32,
            words: gxmctx::TEXTURE_CONTROL_WORDS,
        },
        // The two RESERVES. Not a setter and not a getter: an allocation, which is why this
        // block carried a written-down reason for staying on the host until the facts it
        // needs were moved to where the hardware keeps them. See
        // [`vitaslop_transpiler::InlineOp::ReserveUniformBuffer`] for the argument, and
        // [`gxmprog`] for the size it reads.
        //
        // Together the largest remaining item in a gameplay frame's host-call budget:
        // MEASURED at 1,189 crossings a frame on one title (53% of everything it calls) and
        // 601 on another, at ~1.4 us of browser marshalling each.
        g::RESERVE_VERTEX_DEFAULT_UNIFORM_BUFFER => reserve(ctxoff::VERTEX_UNIFORM),
        g::RESERVE_FRAGMENT_DEFAULT_UNIFORM_BUFFER => reserve(ctxoff::FRAGMENT_UNIFORM),
        // ...and the call that is left once those are gone: **1,106 a frame on one retail
        // racer's race, 58% of every host call it still makes.** Two byte copies - into the
        // buffer the guest names and into the fallback SA bank - over a parameter record the
        // guest already holds. See [`vitaslop_transpiler::InlineOp::SetUniformData`] for what
        // it refuses (F16, an unreadable record, a write past the bank) and why each refusal
        // is a case the handler defines rather than a corner cut here.
        g::SET_UNIFORM_DATA_F => SetUniformData { layout: uniform_data_layout() },
        _ => return None,
    })
}

/// The layout an [`vitaslop_transpiler::InlineOp::SetUniformData`] reads.
///
/// Every number is the one the HANDLER uses: the GXM parameter record's own field offsets
/// from the top of this module, the F16 type nibble from the same `ParamType` decode
/// `set_uniform_data_f` calls, and the bank's layout and ceiling from
/// [`crate::host`]. `the_uniform_data_layout_is_closed` holds them together.
fn uniform_data_layout() -> vitaslop_transpiler::UniformDataLayout {
    vitaslop_transpiler::UniformDataLayout {
        bank_slot: super::mirror::SLOT_SA_BANK,
        bank_len_at: 0,
        bank_data_at: crate::host::SA_BANK_DATA,
        param_packed_at: GXM_PARAM_WORD_OFF,
        type_shift: 4,
        type_mask: 0xf,
        f16_type: F16_TYPE_BITS,
        param_index_at: GXM_PARAM_RESOURCE_INDEX_OFF,
        max_regs: crate::host::MAX_DEFAULT_UNIFORM_REGS,
    }
}

/// The `packed` type nibble that means F16 - the one component width
/// [`set_uniform_data_f`] does not write four bytes for, and therefore the one case the
/// inline form hands back.
///
/// Named here rather than written as `1`, and PINNED against `ParamType::from_bits` by
/// `the_uniform_data_layout_is_closed`: the emitted code compares a raw nibble, so a
/// renumbering in the decoder would otherwise leave the two disagreeing about which
/// parameters are half-precision - and the symptom is every F16 uniform after the first
/// landing at the wrong offset, which reads as a shader bug.
const F16_TYPE_BITS: u32 = 1;

/// The layout an [`vitaslop_transpiler::InlineOp::ReserveUniformBuffer`] for `record` reads.
///
/// Every number is the constant the HANDLER uses, taken from [`gxmctx`] and [`gxmprog`]
/// rather than written out again, so the emitted code and the fallback cannot disagree about
/// where a word lives. `the_uniform_reserve_layout_is_closed` holds the two record offsets to
/// the shape the emitter assumes (`[buffer, size, header]`, in that order, contiguous).
fn uniform_ring_layout(record: u32) -> vitaslop_transpiler::UniformRingLayout {
    vitaslop_transpiler::UniformRingLayout {
        ctx_magic_at: gxmctx::off::MAGIC,
        ctx_magic: gxmctx::MAGIC,
        ctx_program: match record {
            r if r == gxmctx::off::FRAGMENT_UNIFORM => gxmctx::off::FRAGMENT_PROGRAM,
            _ => gxmctx::off::VERTEX_PROGRAM,
        },
        ctx_ring_base: gxmctx::off::UNIFORM_RING_BASE,
        ctx_ring_end: gxmctx::off::UNIFORM_RING_END,
        ctx_ring_cursor: gxmctx::off::UNIFORM_RING_CURSOR,
        record,
        prog_magic_at: gxmprog::off::MAGIC,
        prog_magic: gxmprog::MAGIC,
        prog_size: gxmprog::off::UNIFORM_SIZE,
        prog_alloc: gxmprog::off::UNIFORM_ALLOC,
        prog_header: gxmprog::off::HEADER,
        align: gxmctx::UNIFORM_ALIGN,
    }
}

/// Where `InlineOp::BindPrecomputedState` finds everything, per stage - the one place the
/// context, state-struct and arrays-block offsets meet the emitter. Pinned to the handlers
/// by the `precomputed_state_binds` tests.
fn bind_state_layout(fragment: bool) -> vitaslop_transpiler::BindStateLayout {
    use crate::vita::gxmstate;
    vitaslop_transpiler::BindStateLayout {
        ctx_magic_at: gxmctx::off::MAGIC,
        ctx_magic: gxmctx::MAGIC,
        st_magic_at: gxmstate::off::MAGIC,
        st_magic: if fragment { gxmstate::MAGIC_FRAGMENT } else { gxmstate::MAGIC_VERTEX },
        st_block_at: gxmstate::off::BLOCK,
        st_buf_at: gxmstate::off::BUF,
        st_size_at: gxmstate::off::SIZE,
        st_header_at: gxmstate::off::HEADER,
        st_handle_at: gxmstate::off::HANDLE,
        ctx_record: if fragment { gxmctx::off::FRAGMENT_UNIFORM } else { gxmctx::off::VERTEX_UNIFORM },
        copy_dst: if fragment { gxmctx::off::TEXTURES } else { gxmctx::off::VERTEX_UNIFORM_BUFFERS },
        copy_bytes: if fragment {
            // The TEXTURE array alone. The fragment block's uniform-buffer table sits behind
            // it and goes somewhere else entirely in the context block, so it is applied by
            // the handler's own loop rather than by this copy.
            gxmstate::TEXTURE_ARRAY_BYTES
        } else {
            // The vertex block's TABLE half only - the recorded (never-applied) textures
            // behind it stay behind, see `gxmstate::VERTEX_BLOCK_BYTES`.
            gxmstate::VERTEX_BLOCK_TEXTURES
        },
        // The fragment copy is the texture array, and an empty slot there is this engine's own
        // zero rather than the guest's - see `BindStateLayout::copy_slot_stride` and the
        // handler's matching loop in `VitaState::bind_precomputed_fragment_state`. The vertex
        // copy is the uniform-buffer table, where a zero entry IS the guest's "no buffer".
        copy_slot_stride: if fragment { gxmctx::TEXTURE_STRIDE } else { 0 },
        ctx_prog: gxmctx::off::FRAGMENT_PROGRAM,
        has_prog: fragment,
    }
}

/// Why the remaining `sceGxmSet*` calls are NOT inlined. Kept as code rather than a comment
/// so a future reader adding one has to answer the same question, and so `only_pure_setters_
/// are_inlined` can walk the list.
///
/// One reason survives:
/// - **The handler does something else.** `sceGxmSetVisibilityBuffer` clears the accumulated
///   occlusion counts. Inlining silently deletes that, because the handler simply never runs.
///
/// "**More than one value word**" used to head this list (`sceGxmSetViewport` six floats,
/// `sceGxmSetRegionClip` five, `sceGxmSet*StencilFunc` six) - a fact about the FORMS
/// available, not about the calls, and the run forms (`StoreVfpRun` / `StoreArgRun`) closed
/// every one of them.
///
/// `sceGxmReserve*DefaultUniformBuffer` used to be on this list, under a third reason that
/// read "the call is not a store at all - it allocates and sizes a buffer from the bound
/// program's reflected interface, and no amount of moving state into the guest changes
/// that." The last clause was simply wrong, and it cost the frame 1,189 crossings while it
/// stood: the SIZE is fixed when the program is created and now lives in the handle
/// ([`gxmprog`]), and the allocation is a bump of a ring that GXM keeps in the guest's own
/// memory anyway, so both halves of "real work" turned out to be facts in the wrong place.
/// See `InlineOp::ReserveUniformBuffer`.
///
/// `sceGxmSetFragmentTexture` used to be on this list, for a reason that was correct about
/// the call and wrong about the conclusion: it copies a texture's control words BY VALUE
/// (`vitaslop-texture-binding-by-value`), so storing the POINTER would be a different
/// program. The answer was a COPY form rather than a store form - see
/// `InlineOp::CopyArgIndexed` - which does what the hardware does, at bind time. "It is not a
/// plain store" is a reason to widen the closed set by one proven operation, not a reason to
/// keep crossing 1,275 times a frame.
#[cfg(test)]
const NOT_INLINABLE: &[(u32, &str)] = &[
    // `SET_VIEWPORT` and `SET_REGION_CLIP` used to sit here under "six value words" /
    // "five value words" - which was a fact about the FORMS available, not about the
    // calls, and the run forms (`StoreVfpRun` / `StoreArgRun`) closed it.
    // `SET_FRONT_STENCIL_FUNC` used to sit here too, under "masks two of its six value
    // words (`& 0xff`), which a run form stores as passed" - a fact about WHERE the
    // narrowing lived, not about the call. It lives at the read-back now
    // (`gxmctx::render_state`), the handler stores as passed, and the run form is exact.
    (crate::nid::gxm::SET_VISIBILITY_BUFFER, "clears the occlusion counters"),
];

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
    ctx.write_u32(addr + 12, (swizzle & 0x7) << texword3::SWIZZLE_SHIFT);
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

/// `SCE_GXM_ERROR_INVALID_POINTER` (`psp2/gxm.h`).
const SCE_GXM_ERROR_INVALID_POINTER: i32 = 0x805B_0004u32 as i32;

/// int sceGxmColorSurfaceInitDisabled(SceGxmColorSurface *surface)
///
/// Initialise a colour surface as DISABLED: a render target that writes no colour at all.
/// A title uses this for a depth-only pass - a shadow map or a z-prepass - where the depth
/// surface is the whole output and a colour attachment would only cost bandwidth.
///
/// # Why this writes a full surface rather than just zeroing
/// The struct still has to be a RECOGNISABLE colour surface: it is passed to
/// `sceGxmBeginScene` like any other, and the reader must be able to tell "this is a
/// surface that is disabled" from "this is uninitialised memory", which is exactly the
/// distinction [`COLOR_SURFACE_MAGIC`] exists to make. So the magic is stamped and every
/// field is zeroed - `data_addr == 0` being the thing that marks it disabled, and the same
/// thing `sceGxmColorSurfaceIsEnabled` would report on.
#[hostcall]
pub(super) fn color_surface_init_disabled(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    surface: Ptr,
) -> i32 {
    do_color_surface_init_disabled(ctx, st, surface.addr())
}

/// See [`color_surface_init_disabled`]. A `#[hostcall]` body cannot early-return.
fn do_color_surface_init_disabled(ctx: &mut GuestCtx, st: &mut VitaState, addr: u32) -> i32 {
    if addr == 0 {
        return SCE_GXM_ERROR_INVALID_POINTER;
    }
    let s = ColorSurface {
        format: 0,
        surface_type: 0,
        width: 0,
        height: 0,
        stride_pixels: 0,
        // The disabled marker. Nothing may be rendered into a surface with no memory
        // behind it, and a later `IsEnabled` reads this same word.
        data_addr: 0,
        scale_mode: 0,
        gamma: 0,
    };
    tracing::debug!(
        target: "vitaslop::gxm",
        surface = format_args!("{addr:#x}"),
        caller = format_args!("{:#010x}", ctx.regs[14]),
        "colorSurfaceInitDisabled"
    );
    write_color_surface(ctx, addr, &s);
    st.set_color_surface(addr, s);
    0
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
    let handle = st.new_program_handle(ctx, program_header);
    st.set_vertex_program(handle, attributes, streams, program_header);
    // Remember the program itself, not just the binding. A title that creates its FRAGMENT
    // programs with a NULL `vertexProgram` names no pair anywhere, and the only material left
    // to build one from is the two lists of programs it created - see
    // `VitaState::note_vertex_program_created`.
    st.note_vertex_program_created(ctx, program_header);
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

/// The blend a fragment program performs ITSELF, as a [`crate::capture::BlendState`], for a
/// program GXM was given no `SceGxmBlendInfo` for.
///
/// The GXM enums this builds are the ones the same title passes explicitly for its other
/// shaders, so the two routes produce the same value for the same equation and everything
/// downstream - the pipeline cache key included - stays one representation.
///
/// It REPORTS, always. A frame whose blending came from the shader bytes rather than from the
/// API argument must not be indistinguishable from one where the guest asked for it: that is the
/// difference between reading a blob correctly and having guessed well.
fn program_rop_blend(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    program_header: u32,
) -> Option<crate::capture::BlendState> {
    use vitaslop_gxp_shader::RopDstFactor;
    let blob = st.program_blob(ctx, program_header);
    let rop = vitaslop_gxp_shader::rop_blend(&blob)?;
    // `SceGxmBlendFactor`: 1 = ONE, 4 = SRC_ALPHA, 5 = ONE_MINUS_SRC_ALPHA.
    // `SceGxmBlendFunc`: 0 = NONE, 1 = ADD.
    let blend = crate::capture::BlendState {
        color_mask: 0xf,
        color_func: 1,
        alpha_func: 1,
        color_src: 4,
        color_dst: match rop.dst {
            RopDstFactor::One => 1,
            RopDstFactor::OneMinusSrcAlpha => 5,
        },
        alpha_src: 1,
        alpha_dst: 0,
    };
    report_rop_blend(program_header, rop, blend);
    Some(blend)
}

/// Report - once per program - that a blend came from the SHADER rather than from GXM.
fn report_rop_blend(
    program_header: u32,
    rop: vitaslop_gxp_shader::RopBlend,
    blend: crate::capture::BlendState,
) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(program_header) {
        return;
    }
    eprintln!(
        "gxm blend: fragment program {program_header:#x} was given a NULL blendInfo but its own \
         epilogue SOP2 blends ({rop:?}) - using colorSrc={} colorDst={} from the SHADER",
        blend.color_src, blend.color_dst,
    );
}

pub(super) fn create_fragment_program(ctx: &mut GuestCtx, st: &mut VitaState) {
    let program_id = ctx.arg(1);
    let blend_info = ctx.arg(4);
    // `const SceGxmProgram *vertexProgram` - the program this fragment program will be PAIRED
    // with, which GXM needs here to patch the varying linkage. It names the pair long before any
    // draw does, and that is the whole reason this call can prepare a shader at all.
    let vertex_program = ctx.arg(5);
    let out = ctx.arg(6);
    let program_header = st.shader_program(program_id);
    let handle = st.new_program_handle(ctx, program_header);
    // The BLEND EQUATION arrives here - GXM has no runtime blend setter, so a program created
    // with an additive info always does. Dropping this argument is what forced every renderer
    // downstream to guess the mode from the geometry, and a guess is wrong for whole classes of
    // draw.
    //
    // >>> A NULL `blendInfo` DOES NOT MEAN "never blends". It means the DRIVER patches nothing
    // >>> in - and a fragment program whose offline compiler already compiled the blend into its
    // >>> epilogue needs nothing patched. Such a program ends in a group-0x80 SOP2 whose second
    // >>> source is the OUTPUT register, which at the end of a fragment program is the
    // >>> destination colour the ROP feeds back. Reading NULL as REPLACE drew one title's entire
    // >>> UI - every glyph, every alpha-cut sprite - as opaque rectangles. See
    // >>> `vitaslop_gxp_shader::rop_blend` for the field evidence and for the frame that pins
    // >>> the operator. No blob in any corpus here carries both a real `blendInfo` and an
    // >>> epilogue SOP2, so this is consulted ONLY when GXM supplied nothing and the two can
    // >>> never compound.
    let blend = match ctx.read_bytes(blend_info, 4) {
        b if blend_info != 0 && b.len() == 4 => {
            crate::capture::BlendState::from_bytes([b[0], b[1], b[2], b[3]])
        }
        _ => program_rop_blend(ctx, st, program_header)
            .unwrap_or_else(crate::capture::BlendState::default),
    };
    report_blend_info(program_header, blend_info, blend);
    st.set_fragment_program(handle, program_header, blend);
    // >>> PREPARE THE SHADER HERE, WHERE THE HARDWARE DOES.
    //
    // A `.gxp` holds USSE machine code the SDK compiled offline, so the device's shader patcher
    // has only to patch and link at this call - and titles make it while a loading screen is
    // up. Our recompiler instead has to produce WGSL and have the driver compile it, and doing
    // that lazily at the first DRAW puts the whole cost inside a gameplay frame: MEASURED on
    // a retail race, 931 ms of WGSL compile and 449 ms of pipeline creation, 160
    // pipelines built ACROSS the race, with single frames spending 50-100 ms building 2-6 of
    // them. That is not just slow, it is a different SHAPE from the hardware.
    st.queue_shader_precompile(ctx, vertex_program, program_header);
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
    st.begin_scene(ctx, color, depth, multisample_mode_of(render_target));
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
///
/// Stores the HANDLE in the context block. Resolving it to a `SceGxmProgram *` header is
/// the reader's job (`VitaState::bound_vertex_program`), which moves that lookup from once
/// per bind to once per draw and leaves this call as one guest store - the shape an inline
/// form can take over. See [`gxmctx`].
#[hostcall]
pub(super) fn set_vertex_program(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, vertex_program: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::VERTEX_PROGRAM, vertex_program);
    0
}

/// void sceGxmSetFragmentProgram(context, fragmentProgram)
///
/// As [`set_vertex_program`]: the handle goes in the block and the header + blend state are
/// derived from it at draw time. The blend equation is baked into the fragment program at
/// creation (see [`crate::capture::BlendState`]), so it is a pure function of this handle
/// and needs no separate record.
#[hostcall]
pub(super) fn set_fragment_program(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, fragment_program: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::FRAGMENT_PROGRAM, fragment_program);
    0
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
    report_uniform_write(ctx, uniform_buffer, parameter, base, component_offset, half, &values, source);
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
        st.set_uniform_halves(ctx, base * 2 + component_offset, &values);
    } else {
        st.set_uniforms(ctx, base + component_offset, &values);
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
    // The guest address the values were READ FROM, which is where the next question goes: a
    // uniform whose VALUE is wrong was handed that value by whoever filled this struct, and a
    // store watch cannot be pointed at it without the address. The write DESTINATION is already
    // on the line and is a different address entirely.
    source: u32,
) {
    use std::sync::OnceLock;
    static WATCH: OnceLock<(Vec<u32>, Vec<String>)> = OnceLock::new();
    let (addrs, names) = WATCH.get_or_init(|| {
        let (mut a, mut n) = (Vec::new(), Vec::new());
        // Through the knob table, not `std::env::var`. This watch names the guest code that
        // WROTE a uniform, and the uniform this project needs it for (`screenTintColour`, the
        // white-out) is written on the BROWSER and never on the desktop - so the one engine
        // that can answer could not set the knob at all.
        for t in crate::knobs::var("VITASLOP_UNIFORM_WATCH").unwrap_or_default().split(',') {
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
    // At WARN on `vitaslop::gxm`, not `eprintln!`: a browser has no stderr the panel can show,
    // and this watch exists to be read on the engine where the write happens.
    tracing::warn!(
        target: "vitaslop::gxm",
        "gxm uniform watch: sceGxmSetUniformDataF wrote {name} ({}) into {lo:#x}..={hi:#x} of \
         buffer {buffer:#x} - reg {base}, component offset {component_offset}, values {values:?} \
         READ FROM {source:#x}, \
         leaving {:08x} at {lo:#x}, from lr={:#010x}",
        if half { "F16" } else { "F32" },
        ctx.read_u32(lo),
        ctx.regs[14]
    );
}

/// void sceGxmSetVertexStream(context, unsigned int streamIndex, const void *data)
///
/// A stream binding is a POINTER on hardware - unlike a texture binding, which GXM copies
/// by value - so it lives in the context block like the rest of the sticky state.
#[hostcall]
pub(super) fn set_vertex_stream(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, stream_index: u32, data: u32) -> i32 {
    gxmctx::set_stream(ctx, context, stream_index, data);
    0
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
    // NOTE the "all-zero control words at bind time" report that used to live here is GONE,
    // and deliberately so rather than by omission. It compared the words at the bind against
    // the words at the draw to tell a texture that was never initialised from one whose
    // struct was reused afterwards. Now that a binding IS a by-value copy taken at the bind,
    // the draw-time report can make that distinction from the binding alone - all-zero copied
    // words mean they were zero when they were copied - so a second, bind-time report would
    // be a worse-informed duplicate. It also could not have survived: this handler now runs
    // for a few binds a run, not 1,275 a frame, so the report would have gone quiet and read
    // as "this stopped happening".
    // The context the GUEST named, recorded before the handler discards it - see
    // `TextureSlotWrites::last_bind_context` for the divergence this exists to catch.
    st.note_bind_context(ctx.arg(0));
    st.bind_fragment_texture(ctx, unit, texture);
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
    // Diagnostic (`RUST_LOG=vitaslop::display=trace`): WHICH BUFFER REACHES THE PANEL AT
    // EACH VSYNC is the whole question when a title's picture strobes, and it cannot be
    // read off the scene list - the renderer presents the last scene's TARGET, which is a
    // different statement from the guest's own flip. Logged with the clock and the vblank
    // grid position so a flip can be placed against the vsync it latches at.
    tracing::trace!(
        target: "vitaslop::display",
        thid = st.current_thread(),
        buffer = format_args!("{buffer:#010x}"),
        old = format_args!("{:#010x}", ctx.arg(0)),
        new = format_args!("{:#010x}", ctx.arg(1)),
        us = st.now_us(),
        vcount = super::display::vcount(st),
        // How many entries are waiting for the guest's display-queue callback. GXM runs that
        // callback once per entry; a depth that GROWS is this engine failing to keep up, and
        // the ring of callback-data slots is finite, so a growing depth eventually hands the
        // callback the wrong entry's data.
        cb_pending = st.display_callback_backlog(),
        "queue add entry"
    );
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
// Each `sceGxmSet*` below writes ONE WORD of the sticky GXM context state, which lives in
// the guest's own context memory exactly as it does on hardware - see [`gxmctx`]. The state
// is read back and snapshotted into every draw at record time (`VitaState::record_draw`).
//
// The single-word shape is deliberate and load-bearing: it is what
// `InlineOp::StoreArg { offset }` can emit into guest code, which is how the eight-crossings-
// per-draw block that used to dominate a gameplay frame stops crossing at all. A setter that
// wrote a host field could not be inlined however cheap its handler was.
//
// All return `void` on the Vita - the `-> i32` (0) here just parks a defined value in r0 the
// caller ignores. The enum arguments are stored verbatim as their raw GXM words.

/// void sceGxmSetCullMode(SceGxmContext *context, SceGxmCullMode mode)
#[hostcall]
pub(super) fn set_cull_mode(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, mode: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::CULL_MODE, mode);
    0
}

/// void sceGxmSetTwoSidedEnable(SceGxmContext *context, SceGxmTwoSidedMode enable)
#[hostcall]
pub(super) fn set_two_sided_enable(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, enable: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::TWO_SIDED, enable);
    0
}

/// void sceGxmSetFrontDepthFunc(SceGxmContext *context, SceGxmDepthFunc depthFunc)
#[hostcall]
pub(super) fn set_front_depth_func(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, func: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::FRONT_DEPTH_FUNC, func);
    0
}

/// void sceGxmSetFrontDepthBias(SceGxmContext *context, int factor, int units)
///
/// The polygon offset applied to front faces: the depth of every fragment is nudged by
/// `factor` times the primitive's depth slope plus `units` times the depth buffer's own
/// resolution unit. A title sets it to lift a decal - a skid mark, a shadow blob - off the
/// surface it lies on, so that the two do not z-fight.
///
/// Recorded as SIGNED words: a negative bias (pull toward the viewer) is the common case
/// and reading it unsigned would turn a small negative offset into an enormous positive
/// one.
#[hostcall]
pub(super) fn set_front_depth_bias(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, factor: i32, units: i32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::FRONT_DEPTH_BIAS_FACTOR, factor as u32);
    gxmctx::set(ctx, context, gxmctx::off::FRONT_DEPTH_BIAS_UNITS, units as u32);
    0
}

/// void sceGxmSetBackDepthFunc(SceGxmContext *context, SceGxmDepthFunc depthFunc)
#[hostcall]
pub(super) fn set_back_depth_func(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, func: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::BACK_DEPTH_FUNC, func);
    0
}

/// void sceGxmSetFrontDepthWriteEnable(SceGxmContext *context, SceGxmDepthWriteMode enable)
#[hostcall]
pub(super) fn set_front_depth_write_enable(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, enable: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::FRONT_DEPTH_WRITE, enable);
    0
}

/// void sceGxmSetFrontFragmentProgramEnable(SceGxmContext *context, SceGxmFragmentProgramMode enable)
#[hostcall]
pub(super) fn set_front_fragment_program_enable(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, enable: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::FRONT_FRAGMENT_PROGRAM_ENABLE, enable);
    0
}

/// void sceGxmSetBackFragmentProgramEnable(SceGxmContext *context, SceGxmFragmentProgramMode enable)
#[hostcall]
pub(super) fn set_back_fragment_program_enable(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, enable: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::BACK_FRAGMENT_PROGRAM_ENABLE, enable);
    0
}

/// void sceGxmSetFrontPointLineWidth(SceGxmContext *context, unsigned int width)
#[hostcall]
pub(super) fn set_front_point_line_width(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, width: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::FRONT_POINT_LINE_WIDTH, width);
    0
}

/// void sceGxmSetFrontPolygonMode(SceGxmContext *context, SceGxmPolygonMode mode)
#[hostcall]
pub(super) fn set_front_polygon_mode(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, mode: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::FRONT_POLYGON_MODE, mode);
    0
}

/// void sceGxmSetFrontStencilRef(SceGxmContext *context, unsigned int sref)
#[hostcall]
pub(super) fn set_front_stencil_ref(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, sref: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::FRONT_STENCIL_REF, sref);
    0
}

/// void sceGxmSetFrontStencilFunc(SceGxmContext *context, SceGxmStencilFunc func,
///     SceGxmStencilOp stencilFail, SceGxmStencilOp depthFail, SceGxmStencilOp
///     depthPass, unsigned char compareMask, unsigned char writeMask)
#[hostcall]
pub(super) fn set_front_stencil_func(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    context: u32,
    func: u32,
    stencil_fail: u32,
    depth_fail: u32,
    depth_pass: u32,
    compare_mask: u32,
    write_mask: u32,
) -> i32 {
    use gxmctx::off;
    // The two mask words are stored AS PASSED, not `& 0xff` as they used to be: the byte
    // narrowing lives at the READ-BACK (`gxmctx::render_state`) instead, so the inline run
    // form (`StoreArgRun`, below in `inline_op`) and this handler write identical words.
    // AAPCS already zero-extends an `unsigned char` argument at the call site, so the
    // narrowing is belt-and-braces either way - but it has to live on ONE side, and the
    // read side covers both paths.
    gxmctx::set(ctx, context, off::FRONT_STENCIL_FUNC, func);
    gxmctx::set(ctx, context, off::FRONT_STENCIL_OP_FAIL, stencil_fail);
    gxmctx::set(ctx, context, off::FRONT_STENCIL_OP_DEPTH_FAIL, depth_fail);
    gxmctx::set(ctx, context, off::FRONT_STENCIL_OP_DEPTH_PASS, depth_pass);
    gxmctx::set(ctx, context, off::FRONT_STENCIL_COMPARE_MASK, compare_mask);
    gxmctx::set(ctx, context, off::FRONT_STENCIL_WRITE_MASK, write_mask);
    0
}

/// void sceGxmSetBackStencilFunc(SceGxmContext *context, SceGxmStencilFunc func,
///     SceGxmStencilOp stencilFail, SceGxmStencilOp depthFail, SceGxmStencilOp
///     depthPass, unsigned char compareMask, unsigned char writeMask)
///
/// The two-sided counterpart of [`set_front_stencil_func`], stored in the back-face half
/// of the context block. Recorded whether or not two-sided rendering is currently enabled:
/// a title commonly sets both faces once at start-up and turns `sceGxmSetTwoSidedEnable`
/// on later, so state dropped at the moment it was set is missing when it starts to matter.
#[hostcall]
pub(super) fn set_back_stencil_func(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    context: u32,
    func: u32,
    stencil_fail: u32,
    depth_fail: u32,
    depth_pass: u32,
    compare_mask: u32,
    write_mask: u32,
) -> i32 {
    use gxmctx::off;
    // Stored AS PASSED, masks included - the byte narrowing lives at the read-back so this
    // handler and the inline run form write identical words. See `set_front_stencil_func`.
    gxmctx::set(ctx, context, off::BACK_STENCIL_FUNC, func);
    gxmctx::set(ctx, context, off::BACK_STENCIL_OP_FAIL, stencil_fail);
    gxmctx::set(ctx, context, off::BACK_STENCIL_OP_DEPTH_FAIL, depth_fail);
    gxmctx::set(ctx, context, off::BACK_STENCIL_OP_DEPTH_PASS, depth_pass);
    gxmctx::set(ctx, context, off::BACK_STENCIL_COMPARE_MASK, compare_mask);
    gxmctx::set(ctx, context, off::BACK_STENCIL_WRITE_MASK, write_mask);
    0
}

/// void sceGxmSetViewport(SceGxmContext *context, float xOffset, float xScale,
///     float yOffset, float yScale, float zOffset, float zScale)
#[hostcall]
pub(super) fn set_viewport(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    context: u32,
    x_offset: f32,
    x_scale: f32,
    y_offset: f32,
    y_scale: f32,
    z_offset: f32,
    z_scale: f32,
) -> i32 {
    for (i, v) in [x_offset, x_scale, y_offset, y_scale, z_offset, z_scale].iter().enumerate() {
        gxmctx::set(ctx, context, gxmctx::off::VIEWPORT + i as u32 * 4, v.to_bits());
    }
    0
}

/// void sceGxmSetViewportEnable(SceGxmContext *context, SceGxmViewportMode enable)
#[hostcall]
pub(super) fn set_viewport_enable(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, enable: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::VIEWPORT_ENABLE, enable);
    0
}

/// void sceGxmSetRegionClip(SceGxmContext *context, SceGxmRegionClipMode mode,
///     unsigned int xMin, unsigned int yMin, unsigned int xMax, unsigned int yMax)
#[hostcall]
pub(super) fn set_region_clip(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    context: u32,
    mode: u32,
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::REGION_CLIP_MODE, mode);
    for (i, v) in [x_min, y_min, x_max, y_max].iter().enumerate() {
        gxmctx::set(ctx, context, gxmctx::off::REGION_CLIP + i as u32 * 4, *v);
    }
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
/// Control word 3, from vitasdk `gxm.h struct SceGxmTexture` (an approved permissive
/// reference), declared LSB-first:
///
/// ```text
///  bits 0-25   palette_addr      bits 28-30  swizzle_format
///  bits 26-27  lod_min1          bit  31     normalize_mode
/// ```
///
/// # This was off by one, and it was invisible in exactly one direction
/// Both the writer and the reader used bit 29, so a texture WE initialised round-tripped
/// perfectly - `Set` then `Get` agreed, and every test passed. The error only shows on the
/// other path: a texture the GUEST built its own control words for, which is where
/// `report_texture_resolved_from_control_words` fires. There, reading bits 29:31 of a word
/// whose swizzle really sits at 28:30 returns `(swizzle >> 1) | (normalize_mode << 2)` - so
/// ABGR and ARGB both read as ABGR, RGBA and BGRA both read as ARGB, and a set
/// `normalize_mode` adds 4 to whatever came out.
///
/// A wrong SWIZZLE does not drop a texture or make it obviously garbage: it renders the real
/// image with its channels permuted, which is the failure this codebase already warns about at
/// `report_texture_resolved_from_control_words`. It is also the reason to fix the WRITER and
/// not only the reader - the guest may read back the words we wrote, and it should find the
/// field where the hardware puts it.
pub(crate) mod texword3 {
    /// Low bit of the 3-bit `swizzle_format` field.
    pub const SWIZZLE_SHIFT: u32 = 28;
    pub const SWIZZLE_MASK: u32 = 0x7;
}

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
/// BOTH stages' bindings are sticky context state, exactly like a stream pointer, and a draw
/// whose program declares a non-default uniform buffer reads the address back to snapshot the
/// buffer's bytes. There are two ways a program then reads them, and the blob says which:
/// through MEMORY LOADS chasing the bound pointer (`VitaState::capture_mem_window`), or out of
/// the SA register file the driver copies the buffer into (`Program::sa_uniform_buffers`).
///
/// The fragment side used to be dropped with a warning, on the reading that a fragment program
/// reading a bound buffer must be doing it with a memory load and would refuse to link
/// (`LinkError::FragmentMemLoad`). That reading was incomplete: the SA-resident shape needs no
/// load at all, and one retail title's fragment programs keep their whole fog/material block
/// there. Recording the address costs one guest word and is what lets such a draw be fed.
pub(super) fn set_uniform_buffer(ctx: &mut GuestCtx, stage: &'static str) {
    let context = ctx.arg(0);
    let index = ctx.arg(1);
    let data = ctx.arg(2);
    if stage == "vertex" {
        gxmctx::set_vertex_uniform_buffer(ctx, context, index, data);
    } else {
        gxmctx::set_fragment_uniform_buffer(ctx, context, index, data);
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
        // >>> SAY WHICH WAY THE MODE WENT.
        // This line used to read "GAMMA-CORRECT writes on the surface ..." for every call,
        // including the ones passing mode 0 - which is the guest turning gamma OFF. So the
        // most common reading of this diagnostic (`mode 0x0` beside the word GAMMA-CORRECT)
        // asserted the opposite of what had happened, on the exact question - does this
        // surface hold encoded bytes - that decides how everything sampling it must decode.
        eprintln!(
            "gxm surface: {} on the surface at data {:#x} ({}x{}), mode {gamma:#x}",
            if gamma != 0 { "GAMMA-CORRECT writes ENABLED" } else { "gamma-correct writes DISABLED (linear)" },
            s.data_addr,
            s.width,
            s.height
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

/// void sceGxmPrecomputedDrawSetParamsInstanced(SceGxmPrecomputedDraw *precomputedDraw,
///     SceGxmPrimitiveType primType, SceGxmIndexFormat indexType, const void *indexData,
///     unsigned int indexCount, unsigned int indexWrap)
///
/// The instanced form of [`precomputed_draw_set_params`], carrying the same extra
/// argument `sceGxmDrawInstanced` takes. The bundle stores the base geometry exactly as
/// the non-instanced call does; the wrap is recorded in the log and not applied, which is
/// the same coverage [`draw_instanced`] has - the first instance renders, and a title that
/// instances its scenery says so in the log rather than silently drawing one copy.
#[hostcall]
pub(super) fn precomputed_draw_set_params_instanced(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    precomputed: u32,
    prim_type: u32,
    index_type: u32,
    index_data: u32,
    index_count: u32,
    index_wrap: u32,
) -> i32 {
    tracing::debug!(
        target: "vitaslop::gxm",
        index_count, index_wrap,
        instances = if index_wrap > 0 { index_count / index_wrap } else { 1 },
        "precomputedDrawSetParamsInstanced"
    );
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
// into a guest struct the game builds once and binds per draw (one title draws
// almost entirely through this path - `sceGxmSetUniformDataF` is never called). The
// state lives IN the guest struct plus a guest-heap arrays block (`vita::gxmstate`),
// so a state the title `memcpy`s keeps working and the per-draw binds are inlinable
// (`InlineOp::BindPrecomputedState`). `Init`/`SetDefaultUniformBuffer`/`SetTexture`
// write the bundle; `sceGxmSetPrecomputed{Vertex,Fragment}State` applies it to the
// context block so `record_draw` snapshots the same uniforms and textures the direct
// path would.

/// unsigned int sceGxmGetPrecomputedVertexStateSize(const SceGxmVertexProgram *program)
/// The size the guest allocates for the state's memBlock. The public struct is
/// SCE_GXM_PRECOMPUTED_VERTEX_STATE_WORD_COUNT (7) u32 words = 0x1C bytes; the state
/// lives in the STRUCT plus an arrays block this engine allocates from the guest heap
/// (`vita::gxmstate`), so the guest's memBlock is bookkeeping we do not consume - its
/// real-driver size is not ours to define, which is why the arrays do not live there.
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
pub(super) fn precomputed_vertex_state_init(ctx: &mut GuestCtx, st: &mut VitaState, state: u32, vertex_program: u32, _mem_block: u32) -> i32 {
    st.precomputed_vertex_state_init(ctx, state, vertex_program);
    0
}

/// int sceGxmPrecomputedFragmentStateInit(SceGxmPrecomputedFragmentState *state,
///     const SceGxmFragmentProgram *fragmentProgram, void *memBlock)
#[hostcall]
pub(super) fn precomputed_fragment_state_init(ctx: &mut GuestCtx, st: &mut VitaState, state: u32, fragment_program: u32, _mem_block: u32) -> i32 {
    st.precomputed_fragment_state_init(ctx, state, fragment_program);
    0
}

/// void sceGxmPrecomputedVertexStateSetDefaultUniformBuffer(state, void *defaultBuffer)
#[hostcall]
pub(super) fn precomputed_vertex_state_set_default_uniform_buffer(ctx: &mut GuestCtx, st: &mut VitaState, state: u32, buffer: u32) -> i32 {
    st.precomputed_vertex_state_set_uniform_buffer(ctx, state, buffer);
    0
}

/// void sceGxmPrecomputedFragmentStateSetDefaultUniformBuffer(state, void *defaultBuffer)
#[hostcall]
pub(super) fn precomputed_fragment_state_set_default_uniform_buffer(ctx: &mut GuestCtx, st: &mut VitaState, state: u32, buffer: u32) -> i32 {
    st.precomputed_fragment_state_set_uniform_buffer(ctx, state, buffer);
    0
}

/// void *sceGxmPrecomputedVertexStateGetDefaultUniformBuffer(const ...State *state)
#[hostcall]
pub(super) fn precomputed_vertex_state_get_default_uniform_buffer(ctx: &mut GuestCtx, st: &mut VitaState, state: u32) -> u32 {
    st.precomputed_vertex_state_uniform_buffer(ctx, state)
}

/// void *sceGxmPrecomputedFragmentStateGetDefaultUniformBuffer(const ...State *state)
#[hostcall]
pub(super) fn precomputed_fragment_state_get_default_uniform_buffer(ctx: &mut GuestCtx, st: &mut VitaState, state: u32) -> u32 {
    st.precomputed_fragment_state_uniform_buffer(ctx, state)
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
pub(super) fn set_back_depth_write_enable(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, enable: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::BACK_DEPTH_WRITE, enable);
    0
}

/// void sceGxmSetBackPolygonMode(SceGxmContext *context, SceGxmPolygonMode mode)
#[hostcall]
pub(super) fn set_back_polygon_mode(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, mode: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::BACK_POLYGON_MODE, mode);
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
pub(super) fn set_front_visibility_test_enable(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, enable: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::FRONT_VISIBILITY_TEST_ENABLE, enable);
    0
}

/// void sceGxmSetFrontVisibilityTestIndex(SceGxmContext *context, unsigned int index)
#[hostcall]
pub(super) fn set_front_visibility_test_index(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, index: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::FRONT_VISIBILITY_TEST_INDEX, index);
    0
}

/// void sceGxmSetFrontVisibilityTestOp(SceGxmContext *context, SceGxmVisibilityTestOp op)
#[hostcall]
pub(super) fn set_front_visibility_test_op(ctx: &mut GuestCtx, _st: &mut VitaState, context: u32, op: u32) -> i32 {
    gxmctx::set(ctx, context, gxmctx::off::FRONT_VISIBILITY_TEST_OP, op);
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

/// unsigned int sceGxmProgramGetSize(const SceGxmProgram *program)
///
/// The container header's own `size` field (+0x08), which is the length in bytes of the
/// program blob - what a title reads before copying a program it has just been handed.
/// Not derived from anything we know about the program: the header states it, and a
/// value computed some other way could disagree with the bytes the guest owns.
#[hostcall]
pub(super) fn program_get_size(ctx: &mut GuestCtx, _st: &mut VitaState, program: u32) -> u32 {
    ctx.read_u32(program.wrapping_add(0x08))
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
/// Points a paletted (P8/P4) texture at its colour table.
///
/// # The pointer is written into CONTROL WORD 3, where the hardware keeps it
/// vitasdk's `SceGxmTexture` declares control word 3's low field as `palette_addr : 26` and
/// its `sceGxmTextureGetPalette` shifts that field left by six - the palette is 64-byte
/// aligned, so the six bits are not lost. Keeping the pointer ONLY in a host-side map keyed by
/// the `SceGxmTexture *` was the shape that could not follow a struct COPY, and copying the
/// struct is how the title that needed this binds every one of its paletted textures: the
/// capture then saw a paletted format with no table and dropped the binding.
///
/// The host map is kept beside it, because it is what a `Get` can answer for a texture whose
/// words were never ours to write.
#[hostcall]
pub(super) fn texture_set_palette(ctx: &mut GuestCtx, st: &mut VitaState, texture: u32, palette: u32) -> i32 {
    report_unaligned_palette(texture, palette);
    if texture != 0 {
        let w3 = ctx.read_u32(texture.wrapping_add(12));
        let field = (palette >> 6) & 0x03ff_ffff;
        ctx.write_u32(texture.wrapping_add(12), (w3 & !0x03ff_ffffu32) | field);
    }
    st.set_texture_palette(texture, palette);
    0
}

/// `void *sceGxmTextureGetPalette(const SceGxmTexture *texture)`
///
/// The inverse of [`texture_set_palette`], and it reads the same two places that one
/// writes, in the same order of trust: the host-side map first, because it holds the
/// pointer EXACTLY as the guest gave it, and control word 3 otherwise, because a texture
/// whose words the guest wrote itself (or COPIED from another struct) was never in the map.
/// The word's field is `palette_addr : 26` for a 64-byte-aligned table, so recovering the
/// address is a shift back up by six - which is why an unaligned pointer is reported when
/// it is SET rather than silently rounded here.
#[hostcall]
pub(super) fn texture_get_palette(ctx: &mut GuestCtx, st: &mut VitaState, texture: u32) -> u32 {
    let recorded = st.texture_palette(texture);
    if recorded != 0 {
        recorded
    } else if texture != 0 {
        (ctx.read_u32(texture.wrapping_add(12)) & 0x03ff_ffff) << 6
    } else {
        0
    }
}

/// Say - once - that a palette pointer is not 64-byte aligned, so control word 3's 26-bit
/// field cannot carry it exactly. GXM requires the alignment; a title that broke it would have
/// its low bits silently dropped, which is worth a line rather than a wrong table.
fn report_unaligned_palette(texture: u32, palette: u32) {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if palette % 64 == 0 || REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        texture = format_args!("{texture:#x}"),
        palette = format_args!("{palette:#x}"),
        "sceGxmTextureSetPalette was given a palette that is NOT 64-byte aligned - control          word 3's 26-bit field cannot represent its low six bits, so a copy of this texture          will read a palette up to 63 bytes below the one that was set"
    );
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
pub(super) fn precomputed_state_set_uniform_buffer(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    stage: &'static str,
    all: bool,
) {
    // Both stages' bindings are recorded and applied at bind time, because a recompiled
    // program reads them either through MEMORY LOADS chasing the pointer
    // (`VitaState::capture_mem_window`) or straight out of the SA register file the driver
    // copies the buffer into (`Program::sa_uniform_buffers`). The ALL form reads one pointer
    // per possible index; a slot the guest's (shorter) array did not cover is only ever
    // consumed if the program declares that buffer index, in which case the array covered it.
    let state = ctx.arg(0);
    if all {
        let array = ctx.arg(1);
        for i in 0..gxmctx::MAX_UNIFORM_BUFFERS as u32 {
            let data = ctx.read_u32(array.wrapping_add(i * 4));
            st.precomputed_state_set_nondefault_uniform_buffer(ctx, state, stage, i, data);
        }
    } else {
        let (index, data) = (ctx.arg(1), ctx.arg(2));
        st.precomputed_state_set_nondefault_uniform_buffer(ctx, state, stage, index, data);
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
    pub(crate) const PARAM: u32 = 0x40;

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

    /// The reserve HANDLER must bump the ring exactly as the emitted form does.
    ///
    /// The three-part obligation this closes: `the_uniform_reserve_layout_is_closed` proves
    /// the two read the same WORDS, `inline_imports.rs`'s reserve tests prove the emitted
    /// code performs the bump, and this proves the handler performs the same one. Without
    /// the third the handler could drift and nothing would notice - it runs a handful of
    /// times a run (a ring not yet attached, a scene that overran it), so a difference would
    /// surface as one draw in ten thousand reading another's uniforms.
    #[test]
    fn the_reserve_handler_bumps_the_ring_as_the_inline_form_does() {
        use crate::vita::{gxmctx, gxmprog};
        const CONTEXT: u32 = 0x400;
        const VHANDLE: u32 = 0x800;
        const FHANDLE: u32 = 0x840;
        const OUT: u32 = 0x880;
        const RING: u32 = 0x1000;
        const RING_BYTES: u32 = 0x800;
        // Deliberately not a multiple of the alignment, so a handler that forgot to align
        // hands the second reserve out at an address the emitted form would never produce.
        const VSIZE: u32 = 0x24;
        const VHEADER: u32 = 0x8100_0000;
        const FHEADER: u32 = 0x8200_0000;

        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 0x4000];
        let mut st = VitaState::new(0, 0x4000, Box::new(DeterministicWorld::default()));
        let mut mem = SliceMemory(&mut bytes);
        let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
        gxmctx::init(&mut ctx, CONTEXT);
        // The ring is placed by hand rather than allocated, so this test is about the BUMP
        // and not about the arena.
        gxmctx::set_uniform_ring(&mut ctx, CONTEXT, RING, RING_BYTES);
        gxmprog::init(&mut ctx, VHANDLE, VSIZE, VHEADER);
        gxmprog::init(&mut ctx, FHANDLE, 0, FHEADER);
        gxmctx::set(&mut ctx, CONTEXT, gxmctx::off::VERTEX_PROGRAM, VHANDLE);
        gxmctx::set(&mut ctx, CONTEXT, gxmctx::off::FRAGMENT_PROGRAM, FHANDLE);
        st.adopt_gxm_context(CONTEXT);

        // What the EMITTED form would produce, computed from the layout it was given
        // rather than from the handler's code.
        let align = |a: u32| (a + gxmctx::UNIFORM_ALIGN - 1) & !(gxmctx::UNIFORM_ALIGN - 1);
        let valloc = VSIZE.max(gxmctx::UNIFORM_MIN_ALLOC);
        let falloc = gxmctx::UNIFORM_MIN_ALLOC;
        let expect = [
            (g::RESERVE_VERTEX_DEFAULT_UNIFORM_BUFFER, gxmctx::off::VERTEX_UNIFORM, align(RING), VSIZE, VHEADER, valloc),
            (g::RESERVE_FRAGMENT_DEFAULT_UNIFORM_BUFFER, gxmctx::off::FRAGMENT_UNIFORM, align(RING) + valloc, 0, FHEADER, falloc),
            // A second vertex reserve, to prove the cursor really moved rather than being
            // rewritten to the same place.
            (g::RESERVE_VERTEX_DEFAULT_UNIFORM_BUFFER, gxmctx::off::VERTEX_UNIFORM, align(RING) + valloc + falloc, VSIZE, VHEADER, valloc),
        ];
        for (nid, record, buf, size, header, alloc) in expect {
            regs[0] = CONTEXT;
            regs[1] = OUT;
            let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            super::super::dispatch(crate::nid::lib::SCE_GXM, nid, &mut ctx, &mut st);
            let name = crate::nid::name(nid);
            assert_eq!(regs[0], 0, "{name} succeeds");
            let ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            assert_eq!(ctx.read_u32(OUT), buf, "{name} hands back the aligned block");
            assert_eq!(
                gxmctx::uniform_binding(&ctx, CONTEXT, record),
                gxmctx::UniformBinding { buf, size, header },
                "{name} records what it handed out"
            );
            assert_eq!(
                gxmctx::uniform_ring(&ctx, CONTEXT).2,
                buf + alloc,
                "{name} leaves the cursor past the block it handed out"
            );
        }
    }

    /// The emitted `sceGxmSetUniformDataF` and the handler behind it must read the SAME
    /// parameter record and write the SAME bank.
    ///
    /// Every number the emitted form is given is a raw offset or a raw nibble - it has no
    /// access to `ParamType` or to the bank's Rust type - so this is the only place the two
    /// can be held together. The half-precision nibble is the one that matters most: get it
    /// wrong and the inline form writes four bytes per component for a uniform the shader
    /// unpacks as two halves, which is a silent corruption of every component after the
    /// first and reads as a shader bug.
    #[test]
    fn the_uniform_data_layout_is_closed() {
        let op = inline_op(g::SET_UNIFORM_DATA_F).expect("sceGxmSetUniformDataF has an inline form");
        let vitaslop_transpiler::InlineOp::SetUniformData { layout: l } = op else {
            panic!("it must lower to the uniform-copy form, got {op:?}");
        };
        assert_eq!(l.param_packed_at, GXM_PARAM_WORD_OFF, "the packed word the handler reads");
        assert_eq!(l.param_index_at, GXM_PARAM_RESOURCE_INDEX_OFF, "...and the resource index");
        assert_eq!(l.bank_data_at, crate::host::SA_BANK_DATA, "the bank's first float");
        assert_eq!(l.bank_len_at, 0, "the high-water word sits before it");
        assert_eq!(l.max_regs, crate::host::MAX_DEFAULT_UNIFORM_REGS, "the same ceiling both sides");
        assert_eq!(l.bank_slot, super::super::mirror::SLOT_SA_BANK, "the slot the host publishes");
        // The type field, decoded by the SAME function the handler decodes it with. The
        // emitted code compares a raw nibble, so this is where a renumbering is caught.
        assert!(
            matches!(
                ParamType::from_bits(l.f16_type as u8),
                ParamType::F16
            ),
            "the nibble the inline form refuses must be the one `ParamType` calls F16"
        );
        assert!(
            !matches!(ParamType::from_bits(0), ParamType::F16),
            "...and F32, the case it DOES answer, must not be"
        );
        assert_eq!((l.type_shift, l.type_mask), (4, 0xf), "the handler's own shift and mask");
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
    fn only_calls_with_no_other_behaviour_are_inlined() {
        for &nid in COVERED.iter().chain(COVERED_PROGRAM.iter()).chain(COVERED_SETTERS.iter().map(|(n, _)| n))
        {
            assert!(
                inline_op(nid).is_some(),
                "{} is listed as covered but has no inline form",
                crate::nid::name(nid)
            );
        }
        assert!(
            inline_op(g::SET_VERTEX_STREAM).is_some(),
            "the indexed setter has an inline form"
        );
        for nid in [
            g::DRAW,                       // records a whole draw into the scene
            g::END_SCENE,                  // completes and folds a frame
            g::PROGRAM_PARAMETER_GET_NAME, // returns a pointer, not a bitfield
            g::PROGRAM_GET_PARAMETER,      // computes an address from two reads
            // A string search over the whole parameter table: pure, but not ONE read.
            // Memoizing it was tried and reverted - see `find_parameter`.
            g::PROGRAM_FIND_PARAMETER_BY_NAME,
        ] {
            assert!(
                inline_op(nid).is_none(),
                "{} does more than one memory access and must not be inlined",
                crate::nid::name(nid)
            );
        }
        // The setters that were CONSIDERED and refused, each with the reason on the entry.
        // Listed rather than assumed, because "we did not get round to it" and "it must not
        // be" look identical from outside `inline_op`.
        for &(nid, why) in super::NOT_INLINABLE {
            assert!(
                inline_op(nid).is_none(),
                "{} must NOT be inlined: it {why}",
                crate::nid::name(nid)
            );
        }
    }

    // --- The context-state SETTERS -------------------------------------------
    //
    // The obligation is the mirror image of the getters': a getter's inline form must
    // compute what its handler returns, a setter's must WRITE what its handler writes,
    // where its handler writes it. Both are second implementations of the same call that
    // nothing else in the system compares, so both are pinned here.

    /// Guest address of the synthetic context block. Not `PARAM`'s neighbourhood, and not
    /// zero, so a form that ignored r0 would read a zeroed page instead of agreeing.
    const CONTEXT: u32 = 0x400;

    /// The value the setter is asked to store. Chosen so no default and no offset could
    /// coincide with it.
    const SET_VALUE: u32 = 0xA5A5_1234;

    /// Run a setter NID through the real dispatch with `(r0, r1, r2) = (CONTEXT, a, b)` over
    /// an initialised context block, and return the whole block afterwards.
    fn setter_block(func_nid: u32, a: u32, b: u32) -> Vec<u8> {
        let mut regs = [0u32; REG_COUNT];
        regs[0] = CONTEXT;
        regs[1] = a;
        regs[2] = b;
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 4096];
        let mut st = VitaState::new(0, 4096, Box::new(DeterministicWorld::default()));
        let mut mem = SliceMemory(&mut bytes);
        {
            let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            // Start from the GXM defaults, so a handler that writes NOTHING is caught by the
            // field still holding its default rather than by it holding zero - which is the
            // default for several of these fields.
            gxmctx::init(&mut ctx, CONTEXT);
            super::super::dispatch(crate::nid::lib::SCE_GXM, func_nid, &mut ctx, &mut st);
        }
        bytes
    }

    /// The word at `CONTEXT + offset` of a block.
    fn block_word(block: &[u8], offset: u32) -> u32 {
        let at = (CONTEXT + offset) as usize;
        u32::from_le_bytes(block[at..at + 4].try_into().expect("4 bytes"))
    }

    /// EVERY inlined setter must write exactly the word its inline form claims, at exactly
    /// the offset its inline form claims, and NOTHING ELSE in the block.
    ///
    /// The "nothing else" half is the one that matters. A setter is a second implementation
    /// of a one-word write; if the handler also touched a neighbouring field, the inline
    /// form would silently stop doing so, and the guest would read a stale value out of a
    /// field nobody thought was involved.
    #[test]
    fn inlined_setters_write_what_their_inline_forms_claim() {
        let untouched = setter_block(g::SET_CULL_MODE, 0, 0);
        for &(func_nid, name) in COVERED_SETTERS {
            let op = inline_op(func_nid).expect("listed NID has an inline form");
            let offset = op.store_offset().expect("a context setter writes through its pointer");
            let block = setter_block(func_nid, SET_VALUE, 0);
            assert_eq!(
                block_word(&block, offset),
                SET_VALUE,
                "{name}: the handler must write the value at the offset its inline form uses"
            );
            assert_eq!(op.eval(0), 0, "{name}: a void setter returns the success code");
            // Every OTHER word of the block must be exactly what a no-op left.
            for w in 0..(gxmctx::BYTES / 4) {
                let at = w * 4;
                if at == offset {
                    continue;
                }
                assert_eq!(
                    block_word(&block, at),
                    block_word(&untouched, at),
                    "{name} changed the word at {at:#x}, which its inline form would not"
                );
            }
        }
    }

    /// The indexed setter, over every index in range and one past the end.
    ///
    /// The out-of-range index is the arm the inline form hands BACK to the handler, so what
    /// is checked there is that the handler declines to write - if it wrote past the array,
    /// the guarded inline form would be the one behaving differently, not the handler.
    #[test]
    fn the_indexed_setter_writes_the_slot_its_inline_form_claims() {
        let op = inline_op(g::SET_VERTEX_STREAM).expect("has an inline form");
        let offset = op.store_offset().expect("writes through its pointer");
        let untouched = setter_block(g::SET_CULL_MODE, 0, 0);
        for index in 0..gxmctx::MAX_VERTEX_STREAMS as u32 {
            assert!(!op.falls_back_on_index(index), "index {index} is in range");
            let block = setter_block(g::SET_VERTEX_STREAM, index, SET_VALUE);
            assert_eq!(block_word(&block, offset + index * 4), SET_VALUE, "stream {index}");
        }
        let past = gxmctx::MAX_VERTEX_STREAMS as u32;
        assert!(op.falls_back_on_index(past), "one past the end is the handler's case");
        let block = setter_block(g::SET_VERTEX_STREAM, past, SET_VALUE);
        for w in 0..(gxmctx::BYTES / 4) {
            assert_eq!(
                block_word(&block, w * 4),
                block_word(&untouched, w * 4),
                "an out-of-range stream index must write NOTHING, not the word past the array"
            );
        }
    }

    /// The inlined setters, and the field each one owns. Written out rather than derived
    /// from `inline_op`, so adding a setter there without a line here leaves
    /// `only_calls_with_no_other_behaviour_are_inlined` failing.
    ///
    /// `SET_VERTEX_STREAM` is deliberately absent - it is indexed, and it has its own test.
    const COVERED_SETTERS: &[(u32, &str)] = &[
        (g::SET_VERTEX_PROGRAM, "sceGxmSetVertexProgram"),
        (g::SET_FRAGMENT_PROGRAM, "sceGxmSetFragmentProgram"),
        (g::SET_CULL_MODE, "sceGxmSetCullMode"),
        (g::SET_FRONT_DEPTH_FUNC, "sceGxmSetFrontDepthFunc"),
        (g::SET_FRONT_DEPTH_WRITE_ENABLE, "sceGxmSetFrontDepthWriteEnable"),
        (g::SET_TWO_SIDED_ENABLE, "sceGxmSetTwoSidedEnable"),
        (g::SET_BACK_DEPTH_FUNC, "sceGxmSetBackDepthFunc"),
        (g::SET_BACK_DEPTH_WRITE_ENABLE, "sceGxmSetBackDepthWriteEnable"),
        (g::SET_FRONT_FRAGMENT_PROGRAM_ENABLE, "sceGxmSetFrontFragmentProgramEnable"),
        (g::SET_BACK_FRAGMENT_PROGRAM_ENABLE, "sceGxmSetBackFragmentProgramEnable"),
        (g::SET_FRONT_POLYGON_MODE, "sceGxmSetFrontPolygonMode"),
        (g::SET_BACK_POLYGON_MODE, "sceGxmSetBackPolygonMode"),
        (g::SET_FRONT_POINT_LINE_WIDTH, "sceGxmSetFrontPointLineWidth"),
        (g::SET_FRONT_STENCIL_REF, "sceGxmSetFrontStencilRef"),
        (g::SET_VIEWPORT_ENABLE, "sceGxmSetViewportEnable"),
        (g::SET_FRONT_VISIBILITY_TEST_ENABLE, "sceGxmSetFrontVisibilityTestEnable"),
        (g::SET_FRONT_VISIBILITY_TEST_INDEX, "sceGxmSetFrontVisibilityTestIndex"),
        (g::SET_FRONT_VISIBILITY_TEST_OP, "sceGxmSetFrontVisibilityTestOp"),
    ];
}

#[cfg(test)]
mod run_setter_tests {
    //! The two RUN setters against their handlers: `sceGxmSetViewport` (six VFP argument
    //! floats into six context words) and `sceGxmSetRegionClip` (five core-register/stack
    //! argument words into five context words). The obligation is the store-form one -
    //! the inline form must WRITE what the handler writes, bit for bit, and nothing else -
    //! held here by dispatching the real handler and checking the words against the very
    //! values the emitted form would store.

    use super::*;
    use crate::nid::gxm as g;
    use crate::{DeterministicWorld, SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::{REG_COUNT, SP};
    use vitaslop_transpiler::InlineOp;

    const CTX: u32 = 0x100;
    const STACK: u32 = 0x800;

    /// Dispatch `func_nid` over a zeroed 4 KB guest image with the given registers, VFP
    /// bits and stack words, and hand the memory back.
    fn dispatch_over(
        func_nid: u32,
        regs: &mut [u32; REG_COUNT],
        vfp: &mut [u32; VFP_ARG_COUNT],
        stack: &[u32],
    ) -> Vec<u8> {
        let mut bytes = vec![0u8; 4096];
        regs[SP] = STACK;
        for (i, w) in stack.iter().enumerate() {
            let off = STACK as usize + i * 4;
            bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }
        let mut st = VitaState::new(0, 4096, Box::new(DeterministicWorld::default()));
        let mut mem = SliceMemory(&mut bytes);
        {
            let mut ctx = crate::host::GuestCtx::new(regs, vfp, &mut mem, 0);
            crate::vita::dispatch(crate::nid::lib::SCE_GXM, func_nid, &mut ctx, &mut st);
        }
        bytes
    }

    fn word(bytes: &[u8], addr: u32) -> u32 {
        let a = addr as usize;
        u32::from_le_bytes(bytes[a..a + 4].try_into().expect("4 bytes"))
    }

    /// The viewport handler stores the six argument floats' RAW BITS - a signalling-NaN
    /// pattern included, so a form that round-tripped through a float op would show - and
    /// the inline form claims exactly that run.
    #[test]
    fn set_viewport_stores_the_six_vfp_argument_bit_patterns() {
        let op = inline_op(g::SET_VIEWPORT).expect("has an inline form");
        assert_eq!(
            op,
            InlineOp::StoreVfpRun { offset: gxmctx::off::VIEWPORT, count: 6 },
            "sceGxmSetViewport lowers to the six-word VFP run at the viewport block"
        );
        let bits: [u32; 6] = [
            0x3f80_0000, // 1.0
            0xbf00_0000, // -0.5
            0x7fa0_0001, // a signalling NaN pattern - bit-exactness or nothing
            0x0000_0001, // a denormal
            0x4479_c000, // 999.0
            0x8000_0000, // -0.0
        ];
        let mut regs = [0u32; REG_COUNT];
        regs[0] = CTX;
        let mut vfp = [0u32; VFP_ARG_COUNT];
        vfp[..6].copy_from_slice(&bits);
        let bytes = dispatch_over(g::SET_VIEWPORT, &mut regs, &mut vfp, &[]);
        for (i, &b) in bits.iter().enumerate() {
            assert_eq!(
                word(&bytes, CTX + gxmctx::off::VIEWPORT + i as u32 * 4),
                b,
                "viewport word {i} must be the raw bits of s{i}"
            );
        }
        assert_eq!(regs[0], 0, "the handler returns success");
        assert_eq!(op.eval(0), 0, "the inline form returns the same success code");
    }

    /// The region-clip handler stores its five argument words AS PASSED - the last two off
    /// the guest stack - into the five consecutive words starting at the mode. The
    /// adjacency the run form rests on is asserted, not assumed.
    #[test]
    fn set_region_clip_stores_the_five_argument_words_from_registers_and_stack() {
        assert_eq!(
            gxmctx::off::REGION_CLIP,
            gxmctx::off::REGION_CLIP_MODE + 4,
            "the run form stores mode + bounds as ONE contiguous run"
        );
        let op = inline_op(g::SET_REGION_CLIP).expect("has an inline form");
        assert_eq!(
            op,
            InlineOp::StoreArgRun { offset: gxmctx::off::REGION_CLIP_MODE, count: 5 },
            "sceGxmSetRegionClip lowers to the five-word argument run at the mode word"
        );
        // Values wider than any plausible clip bound, so a handler that masked one would
        // disagree with the run form and this test is what would say so.
        let args = [0xAAAA_0001u32, 0xBBBB_0002, 0xCCCC_0003, 0xDDDD_0004, 0xEEEE_0005];
        let mut regs = [0u32; REG_COUNT];
        regs[0] = CTX;
        regs[1] = args[0]; // mode
        regs[2] = args[1]; // xMin
        regs[3] = args[2]; // yMin
        let mut vfp = [0u32; VFP_ARG_COUNT];
        // xMax and yMax are AAPCS stack arguments.
        let bytes = dispatch_over(g::SET_REGION_CLIP, &mut regs, &mut vfp, &args[3..]);
        for (i, &v) in args.iter().enumerate() {
            assert_eq!(
                word(&bytes, CTX + gxmctx::off::REGION_CLIP_MODE + i as u32 * 4),
                v,
                "region-clip word {i} must be argument {} as passed",
                i + 1
            );
        }
        assert_eq!(regs[0], 0, "the handler returns success");
        assert_eq!(op.eval(0), 0, "the inline form returns the same success code");
    }

    /// The depth-bias handler stores its two argument words AS PASSED into two consecutive
    /// context words. Both are SIGNED on the guest side and the handler casts them, so the
    /// values below are chosen NEGATIVE - a form that sign-extended, masked or clamped
    /// either one would disagree with the run form here and nowhere else.
    #[test]
    fn set_front_depth_bias_stores_its_two_argument_words_as_passed() {
        assert_eq!(
            gxmctx::off::FRONT_DEPTH_BIAS_UNITS,
            gxmctx::off::FRONT_DEPTH_BIAS_FACTOR + 4,
            "the run form stores factor + units as ONE contiguous run"
        );
        let op = inline_op(g::SET_FRONT_DEPTH_BIAS).expect("has an inline form");
        assert_eq!(
            op,
            InlineOp::StoreArgRun { offset: gxmctx::off::FRONT_DEPTH_BIAS_FACTOR, count: 2 },
            "sceGxmSetFrontDepthBias lowers to the two-word argument run at the factor word"
        );
        let args = [(-3i32) as u32, (-129i32) as u32];
        let mut regs = [0u32; REG_COUNT];
        regs[0] = CTX;
        regs[1] = args[0]; // factor
        regs[2] = args[1]; // units
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let bytes = dispatch_over(g::SET_FRONT_DEPTH_BIAS, &mut regs, &mut vfp, &[]);
        for (i, &v) in args.iter().enumerate() {
            assert_eq!(
                word(&bytes, CTX + gxmctx::off::FRONT_DEPTH_BIAS_FACTOR + i as u32 * 4),
                v,
                "depth-bias word {i} must be argument {} as passed",
                i + 1
            );
        }
        assert_eq!(regs[0], 0, "the handler returns success");
        assert_eq!(op.eval(0), 0, "the inline form returns the same success code");
    }

    /// The two stencil-func handlers store their six argument words AS PASSED - the last
    /// three off the guest stack - into six consecutive context words. The mask words are
    /// deliberately WIDER than a byte here: the `& 0xff` narrowing lives at the read-back
    /// (`gxmctx::render_state`), so a handler that masked on the store path would disagree
    /// with the run form and this test is what would say so.
    #[test]
    fn the_stencil_funcs_store_their_six_argument_words_as_passed() {
        for (nid, base, name) in [
            (g::SET_FRONT_STENCIL_FUNC, gxmctx::off::FRONT_STENCIL_FUNC, "front"),
            (g::SET_BACK_STENCIL_FUNC, gxmctx::off::BACK_STENCIL_FUNC, "back"),
        ] {
            for k in 1..6 {
                assert_eq!(
                    base + k * 4,
                    [
                        base,
                        base + 4,
                        base + 8,
                        base + 12,
                        base + 16,
                        base + 20
                    ][k as usize],
                    "the six {name}-stencil words are ONE contiguous run"
                );
            }
            let op = inline_op(nid).expect("has an inline form");
            assert_eq!(
                op,
                InlineOp::StoreArgRun { offset: base, count: 6 },
                "sceGxmSet{name}StencilFunc lowers to the six-word argument run"
            );
            let args =
                [0xAAAA_0001u32, 0xBBBB_0002, 0xCCCC_0003, 0xDDDD_0004, 0xEEEE_01FE, 0xFFFF_02FD];
            let mut regs = [0u32; REG_COUNT];
            regs[0] = CTX;
            regs[1] = args[0]; // func
            regs[2] = args[1]; // stencilFail
            regs[3] = args[2]; // depthFail
            let mut vfp = [0u32; VFP_ARG_COUNT];
            // depthPass, compareMask and writeMask are AAPCS stack arguments.
            let bytes = dispatch_over(nid, &mut regs, &mut vfp, &args[3..]);
            for (i, &v) in args.iter().enumerate() {
                assert_eq!(
                    word(&bytes, CTX + base + i as u32 * 4),
                    v,
                    "{name}-stencil word {i} must be argument {} as passed",
                    i + 1
                );
            }
            assert_eq!(regs[0], 0, "the handler returns success");
            assert_eq!(op.eval(0), 0, "the inline form returns the same success code");
        }
    }

    /// The two depth-stencil force-mode setters against their handlers: an in-place masked
    /// field update of the surface's `zlsControl` word, preserving every other bit. The
    /// argument sweep includes a value with dirty bits outside the mask, so a form that
    /// stored it unmasked - or masked the wrong bit - would show.
    #[test]
    fn the_force_modes_update_their_zls_bit_in_place() {
        for (nid, mask, name) in [
            (g::DEPTH_STENCIL_SURFACE_SET_FORCE_LOAD_MODE, super::DS_FORCE_LOAD_MASK, "load"),
            (g::DEPTH_STENCIL_SURFACE_SET_FORCE_STORE_MODE, super::DS_FORCE_STORE_MASK, "store"),
        ] {
            let op = inline_op(nid).expect("has an inline form");
            assert_eq!(
                op,
                InlineOp::StoreArgFieldInPlace { offset: super::DS_ZLS_CONTROL, mask },
                "force-{name} lowers to the in-place field store of its zlsControl bit"
            );
            for value in [0u32, mask, 0xFFFF_FFFF] {
                let before = 0xA5A5_A5A9u32;
                let mut regs = [0u32; REG_COUNT];
                regs[0] = CTX;
                regs[1] = value;
                let mut vfp = [0u32; VFP_ARG_COUNT];
                let mut bytes = vec![0u8; 4096];
                let zls_at = (CTX + super::DS_ZLS_CONTROL) as usize;
                bytes[zls_at..zls_at + 4].copy_from_slice(&before.to_le_bytes());
                let mut st = VitaState::new(0, 4096, Box::new(DeterministicWorld::default()));
                let mut mem = SliceMemory(&mut bytes);
                {
                    let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
                    crate::vita::dispatch(crate::nid::lib::SCE_GXM, nid, &mut ctx, &mut st);
                }
                let after = u32::from_le_bytes(bytes[zls_at..zls_at + 4].try_into().unwrap());
                assert_eq!(
                    after,
                    (before & !mask) | (value & mask),
                    "force-{name} with argument {value:#x} must move ONLY its bit"
                );
                assert_eq!(regs[0], 0, "the handler returns success");
            }
            assert_eq!(op.eval(0), 0, "the inline form returns the same success code");
        }
    }

    /// The two uniform-buffer binds, over every index in range and one past the end -
    /// the `SET_VERTEX_STREAM` obligations, held for the two setters that share its shape.
    /// The out-of-range index is the arm the inline form hands BACK to the handler, which
    /// declines to write and reports.
    #[test]
    fn the_uniform_buffer_binds_write_the_slot_their_inline_forms_claim() {
        for (nid, base, name) in [
            (g::SET_VERTEX_UNIFORM_BUFFER, gxmctx::off::VERTEX_UNIFORM_BUFFERS, "vertex"),
            (g::SET_FRAGMENT_UNIFORM_BUFFER, gxmctx::off::FRAGMENT_UNIFORM_BUFFERS, "fragment"),
        ] {
            let op = inline_op(nid).expect("has an inline form");
            assert_eq!(
                op,
                InlineOp::StoreArgIndexed {
                    offset: base,
                    count: gxmctx::MAX_UNIFORM_BUFFERS as u32
                },
                "sceGxmSet{name}UniformBuffer lowers to the bounded indexed store"
            );
            const ADDR: u32 = 0xA5A5_1234;
            for index in 0..gxmctx::MAX_UNIFORM_BUFFERS as u32 {
                assert!(!op.falls_back_on_index(index), "index {index} is in range");
                let mut regs = [0u32; REG_COUNT];
                regs[0] = CTX;
                regs[1] = index;
                regs[2] = ADDR;
                let mut vfp = [0u32; VFP_ARG_COUNT];
                let bytes = dispatch_over(nid, &mut regs, &mut vfp, &[]);
                assert_eq!(
                    word(&bytes, CTX + base + index * 4),
                    ADDR,
                    "{name} uniform buffer {index}"
                );
            }
            let past = gxmctx::MAX_UNIFORM_BUFFERS as u32;
            assert!(
                op.falls_back_on_index(past),
                "index {past} must fall back to the handler"
            );
            let mut regs = [0u32; REG_COUNT];
            regs[0] = CTX;
            regs[1] = past;
            regs[2] = ADDR;
            let mut vfp = [0u32; VFP_ARG_COUNT];
            let bytes = dispatch_over(nid, &mut regs, &mut vfp, &[]);
            assert_eq!(
                word(&bytes, CTX + base + past * 4),
                0,
                "an out-of-range {name} index must write NOTHING, not the word past the array"
            );
        }
    }
}

#[cfg(test)]
mod precomputed_state_binds {
    //! The GUEST-RESIDENT precomputed state (`vita::gxmstate`) against the binds that
    //! apply it - and the inline layout against both, so the emitted form, the host
    //! handler and the state writers cannot disagree about where a fact lives.

    use super::*;
    use crate::nid::gxm as g;
    use crate::vita::gxmstate;
    use crate::{DeterministicWorld, SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::REG_COUNT;

    const CTX: u32 = 0x400;
    const VSTATE: u32 = 0x200;
    const FSTATE: u32 = 0x240;
    const COPY: u32 = 0x280;
    const UB: u32 = 0x3000;
    const TEXTURE: u32 = 0x3800;

    struct Rig {
        st: VitaState,
        bytes: Vec<u8>,
        regs: [u32; REG_COUNT],
        vfp: [u32; VFP_ARG_COUNT],
    }

    impl Rig {
        fn new() -> Rig {
            // Big enough that `galloc` (whose cursor starts at base + 1 MB) lands inside
            // the slice - the state's arrays block comes from the guest heap.
            let mut r = Rig {
                st: VitaState::new(0, 0x20_0000, Box::new(DeterministicWorld::default())),
                bytes: vec![0u8; 0x20_0000],
                regs: [0u32; REG_COUNT],
                vfp: [0u32; VFP_ARG_COUNT],
            };
            // A context block the binds can write. `gxm_context` is what the handlers use.
            {
                let mut mem = SliceMemory(&mut r.bytes);
                let mut ctx = crate::host::GuestCtx::new(&mut r.regs, &mut r.vfp, &mut mem, 0);
                gxmctx::init(&mut ctx, CTX);
            }
            r.st.adopt_gxm_context(CTX);
            r
        }

        fn call(&mut self, nid: u32, args: &[u32]) {
            self.regs[..4].fill(0);
            for (i, a) in args.iter().enumerate() {
                self.regs[i] = *a;
            }
            let mut mem = SliceMemory(&mut self.bytes);
            let mut ctx = crate::host::GuestCtx::new(&mut self.regs, &mut self.vfp, &mut mem, 0);
            crate::vita::dispatch(crate::nid::lib::SCE_GXM, nid, &mut ctx, &mut self.st);
        }

        fn word(&self, addr: u32) -> u32 {
            let a = addr as usize;
            u32::from_le_bytes(self.bytes[a..a + 4].try_into().expect("4 bytes"))
        }
    }

    /// The vertex bind: the state's uniform-buffer table lands over the context's,
    /// wholesale, and the record carries the struct's memoised words - through the real
    /// dispatch, over a state built by the real setters.
    #[test]
    fn vertex_bind_replaces_the_table_and_record_from_the_guest_state() {
        let mut r = Rig::new();
        r.call(g::PRECOMPUTED_VERTEX_STATE_INIT, &[VSTATE, 0, 0]);
        assert_eq!(r.word(VSTATE + gxmstate::off::MAGIC), gxmstate::MAGIC_VERTEX);
        let block = r.word(VSTATE + gxmstate::off::BLOCK);
        assert_ne!(block, 0, "Init attaches an arrays block from the guest heap");
        r.call(g::PRECOMPUTED_VERTEX_STATE_SET_DEFAULT_UNIFORM_BUFFER, &[VSTATE, UB]);
        r.call(g::PRECOMPUTED_VERTEX_STATE_SET_UNIFORM_BUFFER, &[VSTATE, 3, 0xAB00_0000]);
        // A stale direct binding in a slot the state does NOT declare must not survive.
        {
            let mut mem = SliceMemory(&mut r.bytes);
            let mut ctx = crate::host::GuestCtx::new(&mut r.regs, &mut r.vfp, &mut mem, 0);
            gxmctx::set_vertex_uniform_buffer(&mut ctx, CTX, 5, 0xDEAD_0000);
        }
        r.call(g::SET_PRECOMPUTED_VERTEX_STATE, &[CTX, VSTATE]);
        for i in 0..gxmctx::MAX_UNIFORM_BUFFERS as u32 {
            let want = if i == 3 { 0xAB00_0000 } else { 0 };
            assert_eq!(
                r.word(CTX + gxmctx::off::VERTEX_UNIFORM_BUFFERS + i * 4),
                want,
                "table slot {i} after the bind"
            );
        }
        assert_eq!(r.word(CTX + gxmctx::off::VERTEX_UNIFORM), UB, "record: buffer");
        assert_eq!(
            r.word(CTX + gxmctx::off::VERTEX_UNIFORM + 8),
            r.word(VSTATE + gxmstate::off::HEADER),
            "record: header comes from the struct"
        );
    }

    /// The fragment bind: the 16-slot texture array lands wholesale (a unit bound directly
    /// beforehand does not survive), the program handle is bound, and the record follows.
    #[test]
    fn fragment_bind_replaces_the_texture_array_program_and_record() {
        let mut r = Rig::new();
        // A texture struct with distinctive control words for the state to copy by value.
        for (k, w) in [0x1111_2222u32, 0x3333_4444, 0x5555_6666, 0x7777_0004].iter().enumerate() {
            let at = (TEXTURE as usize) + k * 4;
            r.bytes[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
        r.call(g::PRECOMPUTED_FRAGMENT_STATE_INIT, &[FSTATE, 0x77, 0]);
        r.call(g::PRECOMPUTED_FRAGMENT_STATE_SET_DEFAULT_UNIFORM_BUFFER, &[FSTATE, UB]);
        r.call(g::PRECOMPUTED_FRAGMENT_STATE_SET_TEXTURE, &[FSTATE, 2, TEXTURE]);
        // >>> BIND UNIT 7 DIRECTLY, THEN APPLY THE STATE. THE DIRECT BIND MUST SURVIVE, AND
        // >>> THIS TEST ASSERTED THE OPPOSITE UNTIL A SHIPPING TITLE PROVED IT WRONG.
        //
        // The old rule was "the state does not declare unit 7, so the bind must clear it" -
        // written from the plausible reading that binding a precomputed state REPLACES the
        // stage's whole texture array. But the array it replaces from is a block THIS ENGINE
        // allocates and zeroes, and only `PrecomputedFragmentStateSetTexture` ever fills a slot
        // in it, so an empty slot is an unwritten value rather than an unbind the guest asked
        // for [[vitaslop-poison-separates-a-guest-zero-from-an-unwritten-one]].
        //
        // MEASURED on PCSE00120: 19,603 direct `sceGxmSetFragmentTexture` binds, a texture put
        // INTO a precomputed state **0 times** (a complete count - that setter is never
        // inlined), ~1,286 state binds a frame. A store watchpoint on unit 0's slot caught the
        // sequence: the guest binds its sprite texture, twenty state binds follow, and the
        // immediate `sceGxmDraw` that samples it reaches the renderer with all sixteen slots
        // zero. Its title screen could not draw. With empty slots skipped the title runs the
        // whole recipe with the recompiler STRICT and no fallback at all.
        {
            let mut mem = SliceMemory(&mut r.bytes);
            let mut ctx = crate::host::GuestCtx::new(&mut r.regs, &mut r.vfp, &mut mem, 0);
            gxmctx::set_texture_binding(
                &mut ctx,
                CTX,
                7,
                gxmctx::TexBinding { addr: 0x1234, words: [1, 2, 3, 4], from_precomputed: false },
            );
        }
        r.call(g::SET_PRECOMPUTED_FRAGMENT_STATE, &[CTX, FSTATE]);
        let slot = CTX + gxmctx::off::TEXTURES + 2 * gxmctx::TEXTURE_STRIDE;
        assert_eq!(r.word(slot), TEXTURE, "unit 2: the bound texture's address");
        assert_eq!(r.word(slot + 4), 0x1111_2222, "unit 2: control word 0, copied BY VALUE");
        assert_eq!(r.word(slot + 20), 1, "unit 2: marked from_precomputed");
        let undeclared = CTX + gxmctx::off::TEXTURES + 7 * gxmctx::TEXTURE_STRIDE;
        assert_eq!(
            r.word(undeclared),
            0x1234,
            "unit 7: the state carries nothing for this unit, so the DIRECT bind must survive"
        );
        assert_eq!(r.word(CTX + gxmctx::off::FRAGMENT_PROGRAM), 0x77, "the program handle");
        assert_eq!(r.word(CTX + gxmctx::off::FRAGMENT_UNIFORM), UB, "record: buffer");
    }

    /// A state the guest `memcpy`s keeps working - the fidelity half of moving the state
    /// into guest memory (the copy aliases the same arrays block, as on hardware). The old
    /// address-keyed table failed exactly this.
    #[test]
    fn a_memcpyd_state_binds_like_the_original() {
        let mut r = Rig::new();
        r.call(g::PRECOMPUTED_FRAGMENT_STATE_INIT, &[FSTATE, 0x77, 0]);
        r.call(g::PRECOMPUTED_FRAGMENT_STATE_SET_DEFAULT_UNIFORM_BUFFER, &[FSTATE, UB]);
        let (from, to) = (FSTATE as usize, COPY as usize);
        let bytes: Vec<u8> = r.bytes[from..from + gxmstate::off::BYTES as usize].to_vec();
        r.bytes[to..to + bytes.len()].copy_from_slice(&bytes);
        r.call(g::SET_PRECOMPUTED_FRAGMENT_STATE, &[CTX, COPY]);
        assert_eq!(r.word(CTX + gxmctx::off::FRAGMENT_UNIFORM), UB, "the copy binds");
        assert_eq!(r.word(CTX + gxmctx::off::FRAGMENT_PROGRAM), 0x77);
    }

    /// The inline layout names exactly the offsets the state writers and binds use - the
    /// one place the emitted form could silently drift from the handlers.
    #[test]
    fn the_inline_layout_matches_the_guest_structures() {
        for (nid, fragment) in
            [(g::SET_PRECOMPUTED_VERTEX_STATE, false), (g::SET_PRECOMPUTED_FRAGMENT_STATE, true)]
        {
            let op = inline_op(nid).expect("the bind has an inline form");
            let vitaslop_transpiler::InlineOp::BindPrecomputedState { layout: l } = op else {
                panic!("{} must lower to a state bind", crate::nid::name(nid));
            };
            assert_eq!(l, super::bind_state_layout(fragment));
            assert_eq!(l.ctx_magic, gxmctx::MAGIC);
            assert_eq!(l.st_magic_at, gxmstate::off::MAGIC);
            assert_eq!(l.st_block_at, gxmstate::off::BLOCK);
            assert_eq!(
                (l.st_buf_at, l.st_size_at, l.st_header_at, l.st_handle_at),
                (gxmstate::off::BUF, gxmstate::off::SIZE, gxmstate::off::HEADER, gxmstate::off::HANDLE)
            );
            if fragment {
                assert_eq!(l.st_magic, gxmstate::MAGIC_FRAGMENT);
                assert_eq!(l.copy_dst, gxmctx::off::TEXTURES);
                assert_eq!(l.copy_bytes, gxmstate::TEXTURE_ARRAY_BYTES);
                assert!(l.has_prog, "the fragment bind stores the program handle");
                assert_eq!(l.ctx_prog, gxmctx::off::FRAGMENT_PROGRAM);
                assert_eq!(l.ctx_record, gxmctx::off::FRAGMENT_UNIFORM);
            } else {
                assert_eq!(l.st_magic, gxmstate::MAGIC_VERTEX);
                assert_eq!(l.copy_dst, gxmctx::off::VERTEX_UNIFORM_BUFFERS);
                assert_eq!(l.copy_bytes, gxmstate::VERTEX_BLOCK_TEXTURES);
                assert!(!l.has_prog, "the vertex bind leaves the bound program alone");
                assert_eq!(l.ctx_record, gxmctx::off::VERTEX_UNIFORM);
            }
        }
    }
}

#[cfg(test)]
mod texture_inline_tests {
    //! The `SceGxmTexture` control-word getters and setters, each against its handler.
    //!
    //! These had no equivalence test at all before the setters were inlined, which was a gap
    //! rather than a decision: a getter's inline form and its handler are two readings of one
    //! bitfield, and the pair that matters most here is a getter and its SETTER, because they
    //! are given the same `(shift, mask)` constants and a wrong pair would read back exactly
    //! what it wrote while placing the field somewhere the hardware does not keep it.
    //!
    //! What makes a setter's failure invisible is that it writes a word it shares with seven
    //! other settings. A form that stored the word instead of the field would leave a texture
    //! whose address mode, mip count and LOD bias had all silently gone to zero - a picture
    //! that reads as a decode bug, on a call the host no longer sees.

    use super::inline_op_tests::{handler_result_over, PARAM};
    use super::*;
    use crate::nid::gxm as g;
    use crate::{DeterministicWorld, SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::REG_COUNT;
    use vitaslop_transpiler::InlineOp;

    /// A synthetic texture's four control words. Every field a form under test reads holds a
    /// DIFFERENT value, so a form that shifts by the wrong amount or masks the wrong width
    /// cannot agree by coincidence - word 1 in particular carries a width and a height that
    /// are neither equal nor zero.
    ///
    /// Content-free: these are field encodings, not game data.
    fn texture_record() -> [u32; 4] {
        [
            // word 0: gamma 1, lod bias 0x15, mip count 0xb, mag 0x2, min 0x1, mip filter 1,
            // uaddr 0x4, vaddr 0x3, and the low three bits set so a mask that is too wide shows.
            (1 << 27) | (0x15 << 21) | (0xb << 17) | (0x2 << 12) | (0x1 << 10) | (1 << 9)
                | (0x4 << 6) | (0x3 << 3) | 0x7,
            // word 1: type 0x5 at bit 29, base format 0x0c at bit 24, width-1 = 1023 at bit 12,
            // height-1 = 511 at bit 0. Width and height differ, so a form reading one shift for
            // the other fails.
            (0x5 << 29) | (0x0c << 24) | (1023 << 12) | 511,
            0xDEAD_0004,
            0xDEAD_0008,
        ]
    }

    /// The word an inline form reads out of [`texture_record`], at its own offset.
    fn word_at(offset: u32) -> u32 {
        texture_record()[(offset / 4) as usize]
    }

    /// Every inlined texture GETTER computes what its handler returns.
    #[test]
    fn texture_getters_match_their_handlers() {
        for func_nid in COVERED_GETTERS {
            let op = inline_op(func_nid).expect("listed NID has an inline form");
            let offset = op.offset().expect("a texture getter reads through its pointer argument");
            assert_eq!(
                op.eval(word_at(offset)),
                handler_result_over(func_nid, texture_record()),
                "inline form of {} disagrees with its handler",
                crate::nid::name(func_nid)
            );
        }
    }

    /// `sceGxmTextureGetWidth`/`GetHeight` return SIZE, and the control word stores SIZE
    /// MINUS ONE. The bias is the whole reason `LoadShiftMask` carries a `plus`, and a form
    /// that dropped it would be off by one on every texture - which resizes a render target
    /// by a pixel and is far easier to blame on a viewport than on a getter.
    #[test]
    fn the_dimension_getters_carry_the_size_minus_one_bias() {
        let width = inline_op(g::TEXTURE_GET_WIDTH).expect("has an inline form");
        let height = inline_op(g::TEXTURE_GET_HEIGHT).expect("has an inline form");
        assert_eq!(width.eval(word_at(4)), 1024, "the fixture stores 1023 and the API says 1024");
        assert_eq!(height.eval(word_at(4)), 512, "the fixture stores 511 and the API says 512");
        // ...and a field of all ones still reads as the largest size, not as zero.
        assert_eq!(width.eval(0xffff_ffff), 4096);
    }

    /// Run a texture setter through the real dispatch over [`texture_record`] and hand back
    /// the four control words afterwards.
    fn setter_words(func_nid: u32, value: u32) -> [u32; 4] {
        let mut regs = [0u32; REG_COUNT];
        regs[0] = PARAM;
        regs[1] = value;
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 4096];
        for (i, w) in texture_record().iter().enumerate() {
            let off = PARAM as usize + i * 4;
            bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }
        let mut st = VitaState::new(0, 4096, Box::new(DeterministicWorld::default()));
        let mut mem = SliceMemory(&mut bytes);
        {
            let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            super::super::dispatch(crate::nid::lib::SCE_GXM, func_nid, &mut ctx, &mut st);
        }
        let mut out = [0u32; 4];
        for (i, w) in out.iter_mut().enumerate() {
            let off = PARAM as usize + i * 4;
            *w = u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4 bytes"));
        }
        out
    }

    /// Every inlined texture SETTER writes exactly the field its inline form claims, and
    /// leaves every other bit of the control words exactly as it found them.
    ///
    /// The value is deliberately WIDER than any field under test, so a form that failed to
    /// mask it would spill into the neighbouring setting and be caught by the "nothing else
    /// changed" half rather than by the field itself.
    #[test]
    fn texture_setters_write_the_field_their_inline_forms_claim() {
        const WIDE: u32 = 0xFFFF_FFFF;
        for &(func_nid, name) in COVERED_SETTERS {
            let op = inline_op(func_nid).expect("listed NID has an inline form");
            let InlineOp::StoreArgField { offset, shift, mask } = op else {
                panic!("{name} must lower to a field store, got {op:?}");
            };
            assert_eq!(offset, 0, "{name}: every one of these is control word 0");
            for value in [0u32, 1, 2, WIDE] {
                let got = setter_words(func_nid, value);
                let before = texture_record();
                let want =
                    (before[0] & !(mask << shift)) | ((value & mask) << shift);
                assert_eq!(
                    got[0], want,
                    "{name}: the handler must write the field the inline form writes (value {value:#x})"
                );
                for i in 1..4 {
                    assert_eq!(got[i], before[i], "{name} changed control word {i}");
                }
            }
            assert_eq!(op.eval(0), 0, "{name}: a void setter returns the success code");
        }
    }

    /// A setter and the getter of the SAME field must round-trip: what one writes, the other
    /// reads. This is what catches a `(shift, mask)` pair that is self-consistent between the
    /// two implementations of one call and still names the wrong bits.
    #[test]
    fn a_setter_and_its_getter_name_the_same_field() {
        for &(set_nid, get_nid, name) in &[
            (g::TEXTURE_SET_MAG_FILTER, g::TEXTURE_GET_MAG_FILTER, "magFilter"),
            (g::TEXTURE_SET_MIN_FILTER, g::TEXTURE_GET_MIN_FILTER, "minFilter"),
            (g::TEXTURE_SET_U_ADDR_MODE, g::TEXTURE_GET_U_ADDR_MODE_SAFE, "uAddrMode"),
            (g::TEXTURE_SET_V_ADDR_MODE, g::TEXTURE_GET_V_ADDR_MODE_SAFE, "vAddrMode"),
            (g::TEXTURE_SET_LOD_BIAS, g::TEXTURE_GET_LOD_BIAS, "lodBias"),
        ] {
            let get = inline_op(get_nid).expect("the getter has an inline form");
            // A value that fits every field under test, and is not the fixture's own.
            let written = setter_words(set_nid, 2);
            assert_eq!(get.eval(written[0]), 2, "{name} does not read back what it wrote");
        }
    }

    /// The texture getters this module checks. Written out rather than derived from
    /// `inline_op`, so a NID added there without a line here is not silently uncovered.
    const COVERED_GETTERS: [u32; 11] = [
        g::TEXTURE_GET_LOD_BIAS,
        g::TEXTURE_GET_U_ADDR_MODE_SAFE,
        g::TEXTURE_GET_V_ADDR_MODE_SAFE,
        g::TEXTURE_GET_MIN_FILTER,
        g::TEXTURE_GET_MAG_FILTER,
        g::TEXTURE_GET_MIPMAP_COUNT,
        g::TEXTURE_GET_MIPMAP_COUNT_UNSAFE,
        g::TEXTURE_GET_GAMMA_MODE,
        g::TEXTURE_GET_TYPE,
        g::TEXTURE_GET_WIDTH,
        g::TEXTURE_GET_HEIGHT,
    ];

    /// The texture setters this module checks, with the safe spellings included: they share a
    /// handler and must therefore share an inline form, and "they obviously do" is exactly
    /// the assumption a NID table typo defeats.
    const COVERED_SETTERS: &[(u32, &str)] = &[
        (g::TEXTURE_SET_MAG_FILTER, "sceGxmTextureSetMagFilter"),
        (g::TEXTURE_SET_MIN_FILTER, "sceGxmTextureSetMinFilter"),
        (g::TEXTURE_SET_U_ADDR_MODE, "sceGxmTextureSetUAddrMode"),
        (g::TEXTURE_SET_V_ADDR_MODE, "sceGxmTextureSetVAddrMode"),
        (g::TEXTURE_SET_U_ADDR_MODE_SAFE, "sceGxmTextureSetUAddrModeSafe"),
        (g::TEXTURE_SET_V_ADDR_MODE_SAFE, "sceGxmTextureSetVAddrModeSafe"),
        (g::TEXTURE_SET_LOD_BIAS, "sceGxmTextureSetLodBias"),
    ];

    /// The setters whose enum is ALREADY in control-word position, so their handler masks the
    /// argument in place instead of shifting it up. Held to the same obligation as the list
    /// above through a different arithmetic - which is the whole reason they are a separate
    /// inline form.
    const COVERED_SETTERS_IN_PLACE: &[(u32, &str)] =
        &[(g::TEXTURE_SET_MIP_FILTER, "sceGxmTextureSetMipFilter")];

    /// `sceGxmTextureSetData` against its handler: the aligned pointer lands in control
    /// word 2 with the two low LOD bits PRESERVED, nothing else moves, and the inline form
    /// is the in-place field store over exactly that mask.
    ///
    /// Its own fixture rather than [`texture_record`], because the shared record's word 2
    /// has zero low bits - a form that failed to preserve the LOD bits would pass over it
    /// by coincidence, and preserving them is the whole reason this is a FIELD store.
    #[test]
    fn texture_set_data_keeps_the_lod_bits_and_stores_the_aligned_pointer() {
        let op = inline_op(g::TEXTURE_SET_DATA).expect("has an inline form");
        let InlineOp::StoreArgFieldInPlace { offset, mask } = op else {
            panic!("sceGxmTextureSetData must lower to an in-place field store, got {op:?}");
        };
        assert_eq!(offset, 8, "the data address is control word 2");
        assert_eq!(mask, 0xffff_fffc, "the field is every bit but the two low LOD bits");
        // Word 2 carries LOD bits 0b11 and a data-address bit beside them; the argument
        // carries dirty low bits, so a form that failed to mask either side would show.
        let before: [u32; 4] = [0x1111_1111, 0x2222_2222, 0xDEAD_0007, 0x4444_4444];
        let data = 0x1234_5676u32;
        let mut regs = [0u32; REG_COUNT];
        regs[0] = PARAM;
        regs[1] = data;
        let mut vfp = [0u32; VFP_ARG_COUNT];
        let mut bytes = vec![0u8; 4096];
        for (i, w) in before.iter().enumerate() {
            let off = PARAM as usize + i * 4;
            bytes[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }
        let mut st = VitaState::new(0, 4096, Box::new(DeterministicWorld::default()));
        let mut mem = SliceMemory(&mut bytes);
        {
            let mut ctx = crate::host::GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            super::super::dispatch(crate::nid::lib::SCE_GXM, g::TEXTURE_SET_DATA, &mut ctx, &mut st);
        }
        let mut after = [0u32; 4];
        for (i, w) in after.iter_mut().enumerate() {
            let off = PARAM as usize + i * 4;
            *w = u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4 bytes"));
        }
        let want = (before[2] & !mask) | (data & mask);
        assert_eq!(after[2], want, "the handler must write what the inline form writes");
        assert_eq!(after[2], 0x1234_5677, "aligned pointer over preserved LOD bits 0b11");
        for i in [0usize, 1, 3] {
            assert_eq!(after[i], before[i], "control word {i} must not move");
        }
        assert_eq!(regs[0], 0, "the handler returns success");
        assert_eq!(op.eval(0), 0, "the inline form returns the same success code");
    }

    /// The in-place twin of [`texture_setters_write_the_field_their_inline_forms_claim`].
    ///
    /// >>> AND IT ASSERTS THE THING THAT WOULD GO WRONG. The failure this form exists to avoid
    /// is not "the field moves"; it is a pre-shifted enum masked as if it were numbered from
    /// zero, which turns `SCE_GXM_TEXTURE_MIP_FILTER_ENABLED` (`0x200`) into a stored ZERO -
    /// "disabled", silently, visible only as absent mip filtering. So the values swept below
    /// include the real enum constant, and the expectation is written as the mask IN PLACE
    /// rather than as a shifted copy of the other test's arithmetic.
    #[test]
    fn in_place_texture_setters_store_the_argument_where_it_already_is() {
        for &(func_nid, name) in COVERED_SETTERS_IN_PLACE {
            let op = inline_op(func_nid).expect("listed NID has an inline form");
            let InlineOp::StoreArgFieldInPlace { offset, mask } = op else {
                panic!("{name} must lower to an IN-PLACE field store, got {op:?}");
            };
            assert_eq!(offset, 0, "{name}: every one of these is control word 0");
            let (shift, field) = super::texword0::MIP_FILTER;
            assert_eq!(mask, field << shift, "{name}: the mask must be the field in place");
            // 0x200 is `SCE_GXM_TEXTURE_MIP_FILTER_ENABLED` - the value a shifting form loses.
            for value in [0u32, 0x200, 0xFFFF_FFFF] {
                let got = setter_words(func_nid, value);
                let before = texture_record();
                let want = (before[0] & !mask) | (value & mask);
                assert_eq!(
                    got[0], want,
                    "{name}: the handler must write the field the inline form writes (value {value:#x})"
                );
                for i in 1..4 {
                    assert_eq!(got[i], before[i], "{name} changed control word {i}");
                }
            }
            assert_eq!(op.eval(0), 0, "{name}: a void setter returns the success code");
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

#[cfg(test)]
mod texture_control_word_field_tests {
    use super::texword3;

    /// `swizzle_format` is control word 3 bits 28:30, and `normalize_mode` is bit 31.
    ///
    /// From vitasdk `gxm.h struct SceGxmTexture`: word 3 is `palette_addr:26`, `lod_min1:2`,
    /// `swizzle_format:3`, `normalize_mode:1`, so the swizzle's low bit is 26 + 2 = 28.
    ///
    /// This is a LAYOUT test, not a round-trip test, and that distinction is the whole point:
    /// the shift was 29 in both the writer and the control-word reader, so every round-trip
    /// agreed with itself while every GUEST-authored texture was read one bit off. A test that
    /// wrote a swizzle and read it back would have passed against the bug.
    #[test]
    fn control_word_3_places_the_swizzle_at_bit_28_with_normalize_mode_above_it() {
        assert_eq!(texword3::SWIZZLE_SHIFT, 26 + 2, "palette_addr:26 then lod_min1:2");
        assert_eq!(texword3::SWIZZLE_MASK, 0x7, "swizzle_format is 3 bits");
        // The field ends exactly below normalize_mode (bit 31); it does not overlap it.
        assert_eq!(texword3::SWIZZLE_SHIFT + 3, 31);

        let read = |w: u32| (w >> texword3::SWIZZLE_SHIFT) & texword3::SWIZZLE_MASK;
        // Every selector survives a word that also carries a palette address, both lod_min1
        // bits and normalize_mode - none of which may leak into the answer.
        let noise = 0x03ff_ffff | (0b11 << 26) | (1 << 31);
        for s in 0..8u32 {
            let w = (noise & !(0x7 << texword3::SWIZZLE_SHIFT)) | (s << texword3::SWIZZLE_SHIFT);
            assert_eq!(read(w), s, "selector {s} round-trips through a fully populated word");
        }
        // ...and the old shift is genuinely different, so this test fails against the bug:
        // at shift 29 a word carrying swizzle 2 reads back as 1 (plus normalize_mode's bit).
        let w = 2 << texword3::SWIZZLE_SHIFT;
        assert_eq!(read(w), 2);
        assert_eq!((w >> 29) & 0x7, 1, "the old reader halved every selector");
    }
}
