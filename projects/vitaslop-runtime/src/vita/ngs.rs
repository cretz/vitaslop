//! SceNgsUser: the NGS software synthesizer, modeled at the API level (HLE). A
//! title builds a system, racks of voices, and patch routings, then plays AT9/PCM
//! sources through them and pumps the mix each frame. The real synthesizer runs on
//! a firmware audio DSP; here the NID surface is satisfied so a title's audio
//! subsystem initializes, allocates, and runs its per-frame update without stalling
//! or dereferencing a null handle. Actual sample synthesis is deferred - voices
//! report an idle/finished state, so a title that waits on a sound to end proceeds
//! rather than hanging. When synthesis lands, mixed output crosses the
//! [`AudioSink`](crate::audio::AudioSink) seam via the title's `sceAudioOut` port.
//!
//! Handle model: NGS handles are opaque to the guest. We hand back real, non-null
//! guest pointers - the caller's own system/rack buffers where NGS uses them, and
//! small allocated blocks for voices, definitions, and params buffers - so any
//! incidental read lands on valid, zeroed memory.

use crate::hostcall;
use crate::host::{GuestCtx, VitaState};

/// Diagnostic (`RUST_LOG=vitaslop::ngs=trace`): dump an NGS param/output buffer to
/// understand AT9 routing.
fn dump_mem(ctx: &GuestCtx, label: &str, addr: u32, len: usize) {
    if addr == 0 {
        tracing::trace!(target: "vitaslop::ngs", label, "<null>");
        return;
    }
    let bytes = ctx.read_bytes(addr, len);
    let mut s = String::new();
    for (i, b) in bytes.iter().enumerate() {
        if i % 16 == 0 {
            s.push_str(&format!("\n    {:#010x}: ", addr + i as u32));
        }
        s.push_str(&format!("{b:02x} "));
    }
    tracing::trace!(target: "vitaslop::ngs", label, addr = format_args!("{addr:#010x}"), "dump:{s}");
}

/// SceInt32 sceNgsVoiceUnlockParams(SceNgsHVoice voice, SceUInt32 moduleId)
/// The title has just written its params into the buffer we handed back from
/// LockParams. For the AT9 player module (0), capture the AT9 source so the mixer
/// can decode it at output time.
pub(super) fn voice_unlock_params(ctx: &mut GuestCtx, st: &mut VitaState) {
    let voice = ctx.arg(0);
    let module = ctx.arg(1);
    let addr = st
        .audio_state
        .ngs_param_bufs
        .iter()
        .find(|((v, m, _), _)| *v == voice && *m == module)
        .map(|(_, a)| *a);
    if let Some(addr) = addr {
        if tracing::enabled!(target: "vitaslop::ngs", tracing::Level::TRACE) {
            tracing::trace!(
                target: "vitaslop::ngs",
                voice = format_args!("{voice:#x}"), module = format_args!("{module:#x}"),
                "UnlockParams"
            );
            dump_mem(ctx, "params", addr, 96);
            let data_ptr = ctx.read_u32(addr + 0x08);
            if data_ptr != 0 {
                dump_mem(ctx, "at9data", data_ptr, 32);
            }
        }
        // Module 0 is the source player; its params carry the AT9 buffer + config.
        if module == 0 {
            st.audio_state.at9.set_player_params(ctx, voice, addr);
        } else if !st.audio_state.at9.set_module_params(ctx, voice, addr) {
            // An unrecognised non-source module. Dumped rather than dropped: one of
            // these turned out to carry the master level, and it was identified from
            // exactly this report.
            let id = ctx.read_u32(addr);
            let bytes = ctx.read_bytes(addr, 48);
            tracing::debug!(
                target: "vitaslop::at9",
                voice = format_args!("{voice:#x}"),
                module,
                id = format_args!("{id:#010x}"),
                "non-source module params, not interpreted: {bytes:02x?}"
            );
            // ...and COUNTED, split by whether this voice is a buss. The debug line above is
            // one per write (millions in a race) and off by default, so the shape of what is
            // missing was only ever visible to whoever thought to turn it on.
            let is_buss = st.audio_state.at9.is_buss(voice);
            crate::vita::at9::note_unknown_module(id, module, is_buss, bytes);
        }
    }
    ctx.ret(0);
}

/// SceInt32 sceNgsVoicePlay(SceNgsHVoice voice) - begin decoding this voice's AT9.
pub(super) fn voice_play(ctx: &mut GuestCtx, st: &mut VitaState) {
    let voice = ctx.arg(0);
    st.audio_state.at9.play(voice);
    ctx.ret(0);
}

/// SceInt32 sceNgsVoiceInit(SceNgsHVoice voice, const SceNgsVoicePreset *preset,
///                           SceUInt32 initFlags)
/// Reset a voice to its initial state, optionally from a preset. Whatever the flags
/// select, a reset voice is not playing - so the one part of this the engine DOES model,
/// the source player, is stopped. The preset itself configures module parameters that
/// only a real synthesizer would consume; there is nothing here to apply it to, and
/// pretending otherwise would put state in the mixer that no sample ever passes through.
pub(super) fn voice_init(ctx: &mut GuestCtx, st: &mut VitaState) {
    let voice = ctx.arg(0);
    // >>> IS A PRESET HOW THE SILENT VOICES GET THEIR SOURCE?
    //
    // A race plays 11 voices (77 on a device) that never locked module-0 params through
    // EITHER of the two paths that carry them, so they decode nothing and are silent. A
    // preset is the remaining candidate: it configures module parameters, this handler
    // ignores it, and if a source can arrive that way then ignoring it IS the missing sound.
    // Counted rather than assumed - if no title ever passes one, the candidate is dead and
    // the next session should not re-walk it.
    super::at9::note_voice_init(ctx.arg(1) != 0);
    st.audio_state.at9.stop(voice);
    ctx.ret(0);
}

/// `SceNgsVoiceInfo.uVoiceState` at offset 0: the voice's lifecycle state, a BITFIELD
/// (available 0, active 1, finalizing 2, unloading 4). Only offset 0 is written - see
/// [`voice_get_info`].
const NGS_VOICE_INFO_STATE_OFF: u32 = 0;

/// `SCE_NGS_VOICE_STATE_AVAILABLE`: the voice is idle and holds nothing.
const NGS_VOICE_STATE_AVAILABLE: u32 = 0;

/// SceInt32 sceNgsVoiceGetInfo(SceNgsHVoice voice, SceNgsVoiceInfo *info)
///
/// Reports the voice as AVAILABLE, which is the truth for this engine: no synthesis runs,
/// so no voice is ever active. That is also what unblocks a title waiting for a sound to
/// finish - the observed caller polls this and treats `state & (ACTIVE | UNLOADING)` as
/// "still playing", releasing its voice slot only once that clears.
///
/// Writes ONLY the state word. The rest of `SceNgsVoiceInfo` has no layout in any source
/// available here, and zeroing a struct whose length is a guess would scribble over
/// whatever the caller put after it - a stack frame, in the observed call. A field this
/// function does not know is a field it leaves alone.
#[hostcall]
pub(super) fn voice_get_info(ctx: &mut GuestCtx, _st: &mut VitaState, _voice: u32, info: Ptr) -> i32 {
    if info.addr() != 0 {
        ctx.write_u32(info.addr() + NGS_VOICE_INFO_STATE_OFF, NGS_VOICE_STATE_AVAILABLE);
    }
    0
}

/// Voice key-off / kill / pause - stop producing audio from this voice.
pub(super) fn voice_stop(ctx: &mut GuestCtx, st: &mut VitaState) {
    let voice = ctx.arg(0);
    st.audio_state.at9.stop(voice);
    ctx.ret(0);
}

/// SceInt32 sceNgsSystemUpdate(SceNgsHSynSystem system) - dump the surrounding
/// output/work region so the master-buss mix destination becomes visible.
pub(super) fn system_update(ctx: &mut GuestCtx, _st: &mut VitaState) {
    if tracing::enabled!(target: "vitaslop::ngs", tracing::Level::TRACE) {
        let a3 = ctx.arg(3);
        dump_mem(ctx, "sysupdate-a3", a3, 64);
    }
    ctx.ret(0);
}

/// Bytes reported for a system's / rack's working memory. Any sane non-zero size
/// works: the title allocates this much and hands the buffer back as the handle,
/// which we treat opaquely. Kept modest.
const NGS_WORK_SIZE: u32 = 0x4000;

/// Size of the small blocks we allocate for voice handles, voice definitions, and
/// locked-params buffers. Generous enough to absorb any structured read the title
/// makes against a handle it believes NGS populated.
const NGS_BLOCK_SIZE: u32 = 256;

/// SceInt32 sceNgsSystemGetRequiredMemorySize(const SceNgsSystemInitParams *params,
///                                            SceUInt32 *size)
#[hostcall]
pub(super) fn system_get_required_memory_size(ctx: &mut GuestCtx, _st: &mut VitaState, _params: Ptr, size: Ptr) -> i32 {
    if size.addr() != 0 {
        ctx.write_u32(size.addr(), NGS_WORK_SIZE);
    }
    0
}

/// SceInt32 sceNgsSystemInit(void *memory, SceUInt32 memSize,
///                           const SceNgsSystemInitParams *params, SceNgsHSynSystem *handle)
/// The system handle is the title's own working memory; hand that back (or a fresh
/// block if it passed none) through `handle`.
#[hostcall]
pub(super) fn system_init(ctx: &mut GuestCtx, st: &mut VitaState, memory: Ptr, _mem_size: u32, _params: Ptr, handle: Ptr) -> i32 {
    let sys = if memory.addr() != 0 { memory.addr() } else { st.galloc(NGS_WORK_SIZE, 16) };
    if handle.addr() != 0 {
        ctx.write_u32(handle.addr(), sys);
    }
    0
}

/// SceInt32 sceNgsRackGetRequiredMemorySize(SceNgsHSynSystem system,
///                                          const SceNgsRackDescription *desc, SceUInt32 *size)
#[hostcall]
pub(super) fn rack_get_required_memory_size(ctx: &mut GuestCtx, _st: &mut VitaState, _system: u32, _desc: Ptr, size: Ptr) -> i32 {
    if size.addr() != 0 {
        ctx.write_u32(size.addr(), NGS_WORK_SIZE);
    }
    0
}

/// SceInt32 sceNgsRackInit(SceNgsHSynSystem system, SceNgsBufferInfo *rackBuffer,
///                         const SceNgsRackDescription *desc, SceNgsHRack *handle)
/// `SceNgsBufferInfo` is `{ void *data; SceUInt32 size; }`; the rack handle is that
/// data pointer (or a fresh block if the title supplied none).
#[hostcall]
pub(super) fn rack_init(ctx: &mut GuestCtx, st: &mut VitaState, _system: u32, rack_buffer: Ptr, desc: Ptr, handle: Ptr) -> i32 {
    let data = if rack_buffer.addr() != 0 { ctx.read_u32(rack_buffer.addr()) } else { 0 };
    let rack = if data != 0 { data } else { st.galloc(NGS_WORK_SIZE, 16) };
    if handle.addr() != 0 {
        ctx.write_u32(handle.addr(), rack);
    }
    // >>> WHAT THIS RACK IS MADE OF, reported once per rack.
    //
    // The description was ignored entirely, so nothing in a run said whether a title's mix
    // passes through a MASTER or COMPRESSOR buss - and an unexplained attenuation the device
    // applies and we do not is precisely a question about which busses exist. Reporting it
    // is not implementing it: a buss named here is still a buss this engine does not run, and
    // the line says so rather than letting the name imply otherwise.
    //
    // `SceNgsRackDescription`: definition pointer, voice count, channels per voice, max
    // patches per input, patches per output, user release data.
    if !desc.is_null() {
        let d = desc.addr();
        let (defn, voices, channels, per_in, per_out) = (
            ctx.read_u32(d),
            ctx.read_u32(d + 4),
            ctx.read_u32(d + 8),
            ctx.read_u32(d + 12),
            ctx.read_u32(d + 16),
        );
        let name = match voice_def_name(st, defn) {
            Some(n) => n.to_string(),
            // A definition pointer this run never handed out means the title got it from
            // somewhere we do not model. Say the address rather than a name we do not have.
            None => format!("an unrecognised definition at {defn:#010x}"),
        };
        tracing::warn!(
            target: "vitaslop::at9",
            rack = format_args!("{rack:#x}"),
            "sceNgsRackInit: a rack of {voices} voice(s) x {channels} channel(s) of {name}              (max {per_in} patches per input, {per_out} per output). The MIX this engine              performs is a sum of source voices scaled by their routing volumes; any              processing a buss definition implies - compression, EQ, delay - is NOT run."
        );
    }
    0
}

/// SceInt32 sceNgsRackGetVoiceHandle(SceNgsHRack rack, SceUInt32 index, SceNgsHVoice *handle)
///
/// >>> THE SAME (RACK, INDEX) IS THE SAME VOICE. THIS IS A LOOKUP, NOT AN ALLOCATION, AND
/// >>> GETTING THAT WRONG IS WHAT MADE ONE TITLE'S RACE 44% AUDIO.
///
/// A rack is a fixed array of voices; this hands back the handle of one of them, and a
/// title asks for it whenever it wants to play something. The first version allocated a
/// fresh block on every call, so every query for the same voice produced a DIFFERENT
/// handle - and the consequences are not subtle:
///
/// - the mixer's voice bank grew without bound, one entry per query;
/// - a voice the title later stopped was stopped by handle, and the handle it used was a
///   NEW one, so the voice actually playing was never stopped by anything;
/// - so every sound ever started kept playing, kept being decoded in full, and kept being
///   mixed, for the rest of the run.
///
/// MEASURED on a retail racer's race, 11,321 grains: **8,124 voices started, 217 stopped by
/// the title, 8,138 in the bank, 318 playing every grain with a peak of 843** - against a
/// console whose rack holds tens. With the handle memoised the same title's race is a
/// handful of voices, which is what a browser profile blaming ATRAC9 decode for a third of
/// its thread was really measuring.
#[hostcall]
pub(super) fn rack_get_voice_handle(ctx: &mut GuestCtx, st: &mut VitaState, rack: u32, index: u32, handle: Ptr) -> i32 {
    // The A/B arm: `VITASLOP_NGS_VOICE_HANDLE_MEMO=0` restores the fresh-handle-per-call
    // behaviour, which is the one way to put a number on what it costs on any title.
    let memo = !matches!(crate::knobs::var("VITASLOP_NGS_VOICE_HANDLE_MEMO").as_deref(), Ok("0"));
    let found = if memo {
        st.audio_state.ngs_voice_handles.iter().find(|(k, _)| *k == (rack, index)).copied()
    } else {
        None
    };
    let voice = match found.as_ref() {
        Some((_, v)) => *v,
        None => {
            let v = st.galloc(NGS_BLOCK_SIZE, 16);
            st.audio_state.ngs_voice_handles.push(((rack, index), v));
            v
        }
    };
    if handle.addr() != 0 {
        ctx.write_u32(handle.addr(), voice);
    }
    0
}

/// SceInt32 sceNgsVoiceGetStateData(SceNgsHVoice voice, SceUInt32 moduleId,
///                                  void *data, SceUInt32 dataSize)
/// Report an all-zero state: voice available / not playing. A title polling for a
/// sound to finish sees "done" and proceeds rather than waiting forever.
#[hostcall]
pub(super) fn voice_get_state_data(ctx: &mut GuestCtx, _st: &mut VitaState, _voice: u32, _module: u32, data: Ptr, size: u32) -> i32 {
    if data.addr() != 0 && size != 0 {
        ctx.write_bytes(data.addr(), &vec![0u8; size as usize]);
    }
    0
}

/// The declared size of a params interface, by id - what `uSize` must say for a reader
/// to recognise the struct. Sizes are the ones the layouts in [`crate::vita::at9`] were
/// established from; an id we do not know keeps the block size, which is the only
/// honest claim available for it and which every reader will (correctly) refuse.
fn params_interface_size(param_id: u32) -> u32 {
    match param_id {
        0x0101_5caa => 96, // the ATRAC9 player
        0x0101_5ce6 => 84, // the PCM player
        0x0101_5ce1 => 40, // the buss
        _ => NGS_BLOCK_SIZE,
    }
}

/// SceInt32 sceNgsVoiceLockParams(SceNgsHVoice voice, SceUInt32 moduleId,
///                                SceUInt32 paramInterfaceId, SceNgsBufferInfo *buffer)
/// Return a stable, writable params buffer as `{ data, size }` so the title's
/// per-frame lock/edit/unlock cycle reuses one block instead of leaking each frame.
///
/// >>> THE BUFFER COMES BACK CARRYING ITS `SceNgsParamsDescriptor` HEADER
/// >>> (`{ uId, uSize }`), AND THAT IS WHAT MAKES A TITLE'S MUSIC AUDIBLE.
///
/// Lock does not hand out blank memory on the device: it hands back the voice's
/// CURRENT parameters, which already begin with the descriptor naming which params
/// interface they are. A title therefore writes only the fields it wants to change
/// and never writes `uId` itself.
///
/// Handing back a zeroed block instead left `uId == 0`, so
/// [`At9Voice::load_params`](crate::vita::at9) rejected every source that arrived this
/// way at its first check - while the buffer's other fields (source pointer, byte
/// count, channel count, and a valid `0xFE` ATRAC9 config word) were all correct and
/// sitting right there. MEASURED on a title's music voice: every field parsed, only
/// the id was zero. The whole engine below this - decoder, mixer, sink, ring, worklet
/// - was correct and produced exactly nothing, because a stream of digital silence is
/// what a rejected source sounds like.
///
/// The id written is the one the CALLER asked for (`param`), which is the same
/// interface it is about to write params for; a buffer already handed out keeps its
/// contents, because it IS the voice's persistent parameter state across the cycle.
#[hostcall]
pub(super) fn voice_lock_params(ctx: &mut GuestCtx, st: &mut VitaState, voice: u32, module: u32, param: u32, buffer: Ptr) -> i32 {
    let key = (voice, module, param);
    let buf = match st.audio_state.ngs_param_buf(key) {
        Some(a) => a,
        None => {
            let a = st.galloc(NGS_BLOCK_SIZE, 16);
            st.audio_state.ngs_param_bufs.push((key, a));
            // `uSize` is the size of the PARAMS INTERFACE being locked, not of the
            // block we happen to allocate for it.
            //
            // >>> WRITING THE BLOCK SIZE HERE BROKE EVERY VOICE THAT ARRIVES THIS WAY.
            // The readers check `uSize` to confirm they are looking at the struct they
            // were REd from, which is exactly the check that keeps a wrong layout from
            // being applied to audio - so a descriptor claiming 256 was refused by name,
            // and a whole title's sound effects went silent behind a correct-looking
            // report. Measured on a retail title: every PCM voice refused with
            // "uSize 256, not the 84 this layout was REd from".
            ctx.write_u32(a, param);
            ctx.write_u32(a + NGS_PARAMS_DESC_SIZE_OFF, params_interface_size(param));
            a
        }
    };
    if buffer.addr() != 0 {
        ctx.write_u32(buffer.addr(), buf); // SceNgsBufferInfo.data
        ctx.write_u32(buffer.addr() + 4, NGS_BLOCK_SIZE); // .size
    }
    0
}

/// The voice-definition getters (`sceNgsVoiceDefGet*`) each return a
/// `const SceNgsVoiceDefinition *` the title embeds in a rack description. It is an opaque
/// token to the guest, so a zeroed blob serves - but a DISTINCT one per getter, because the
/// pointer is the only thing that says what kind of voice a rack is made of. See
/// [`crate::vita::audio::AudioState::ngs_defs`]; `rack_init` is the reader.
///
/// Not `#[hostcall]`: the handler needs the NID it was reached by, which the macro does not
/// pass. The dispatch arm calls it directly.
pub(super) fn voice_def_get_for(st: &mut VitaState, func_nid: u32) -> u32 {
    if let Some((_, addr)) = st.audio_state.ngs_defs.iter().find(|(n, _)| *n == func_nid) {
        return *addr;
    }
    let addr = st.galloc(NGS_BLOCK_SIZE, 16);
    st.audio_state.ngs_defs.push((func_nid, addr));
    addr
}

/// Which voice definition a rack description points at, by name, or `None` if the pointer is
/// not one this run handed out.
fn voice_def_name(st: &VitaState, addr: u32) -> Option<&'static str> {
    st.audio_state
        .ngs_defs
        .iter()
        .find(|(_, a)| *a == addr)
        .map(|(nid, _)| crate::nid::name(*nid))
}

/// SceInt32 sceNgsPatchCreateRouting(const SceNgsPatchSetupInfo *info, SceNgsHPatch *handle)
///
/// `SceNgsPatchSetupInfo` opens with the SOURCE voice handle, which is what makes a patch
/// addressable back to the voice it carries - and that is what
/// [`voice_patch_set_volume`] needs to turn a routing volume into a voice gain. The
/// mapping is recorded here rather than rediscovered later, because the patch handle is
/// otherwise an opaque block with nothing linking it to anything.
#[hostcall]
pub(super) fn patch_create_routing(ctx: &mut GuestCtx, st: &mut VitaState, info: Ptr, handle: Ptr) -> i32 {
    let patch = st.galloc(NGS_BLOCK_SIZE, 16);
    if handle.addr() != 0 {
        ctx.write_u32(handle.addr(), patch);
    }
    if !info.is_null() {
        // `SceNgsPatchSetupInfo`: source voice at +0x00, destination voice at +0x0c.
        // Confirmed over 276 routings in one run - the destinations form a small tree
        // (many sources -> two sub-busses -> one -> one) and every destination is a
        // voice that never carries a source of its own.
        let source_voice = ctx.read_u32(info.addr());
        let destination_voice = ctx.read_u32(info.addr() + 0x0c);
        tracing::debug!(
            target: "vitaslop::at9",
            patch = format_args!("{patch:#x}"),
            source_voice = format_args!("{source_voice:#x}"),
            destination_voice = format_args!("{destination_voice:#x}"),
            "patch routing created"
        );
        st.audio_state.ngs_patch_voice.push((patch, source_voice));
        st.audio_state.at9.set_route(source_voice, destination_voice);
    }
    0
}

/// SceInt32 sceNgsVoicePatchSetVolume(SceNgsHPatch patch, SceInt32 outputChannel,
///                                    SceInt32 inputChannel, SceFloat32 volume)
///
/// >>> WITHOUT THIS EVERY VOICE MIXES AT FULL SCALE, WHICH CLIPS.
///
/// The routing volume is how NGS actually balances a mix: a title sets many voices
/// playing and turns most of them well down. Stubbed to `ret(0)`, one title's front end
/// summed ~100 simultaneous voices at unity and CLAMPED 14.7% of its nonzero samples -
/// gross distortion that reads as a broken decoder. Applying the volume is not a
/// refinement, it is the difference between a mix and a clipped sum.
#[hostcall]
pub(super) fn voice_patch_set_volume(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    patch: u32,
    _output_channel: i32,
    _input_channel: i32,
    volume: f32,
) -> i32 {
    st.audio_state.set_patch_volume(patch, volume);
    0
}

/// SceInt32 sceNgsVoicePatchSetVolumesMatrix(SceNgsHPatch patch,
///                                           const SceNgsVolumeMatrix *matrix)
///
/// The matrix is a 2x2 of `SceFloat32` (source channels x destination channels). It is
/// reduced to ONE per-voice gain, because the mixer downstream is per voice: the loudest
/// entry is taken, which is the only reduction that cannot make a voice quieter than the
/// title asked for on any channel.
#[hostcall]
pub(super) fn voice_patch_set_volumes_matrix(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    patch: u32,
    matrix: Ptr,
) -> i32 {
    // No early `return` here: `#[hostcall]` rewrites the body, so one would not mean
    // what it reads as.
    if !matrix.is_null() {
        let loudest = (0..4u32)
            .map(|i| f32::from_bits(ctx.read_u32(matrix.addr() + i * 4)))
            .filter(|v| v.is_finite())
            .fold(0.0f32, f32::max);
        st.audio_state.set_patch_volume(patch, loudest);
    }
    0
}

/// Byte size of the `SceNgsModuleParamHeader` that prefixes each entry in a params
/// BLOCK: `{ SceInt32 moduleId; SceInt32 chan; }`.
///
/// EVIDENCE, from the block this title passes: its first two words are `0` and
/// `0xffffffff` - module 0 (the source player) and "all channels" - and the AT9 params
/// descriptor id [`crate::vita::at9`] looks for (`0x01015caa`) sits at `+8`, with the
/// sample-buffer pointer and byte count following it exactly where
/// `At9Voice::load_params` reads them. So the module params start 8 bytes in, and the
/// same reader that serves `sceNgsVoiceUnlockParams` serves this untouched.
const NGS_MODULE_PARAM_HEADER_BYTES: u32 = 8;

/// Offset of `uSize` within the `SceNgsParamsDescriptor` that begins each module's
/// params (`{ SceUInt32 uId; SceUInt32 uSize; }`), used to step to the next entry.
const NGS_PARAMS_DESC_SIZE_OFF: u32 = 4;

/// SceInt32 sceNgsVoiceSetParamsBlock(SceNgsHVoice voice, const SceNgsModuleParamHeader
///     *pParamData, SceUInt32 uSize, SceInt32 *pnErrorCount)
///
/// Apply a whole block of module parameters in one call - the batch form of
/// lock / write / [`voice_unlock_params`], and the form this title's music path uses.
///
/// EVIDENCE for the shape, off the calling code: the caller initialises a stack local,
/// takes its address with `ADD r3, sp, #0`, loads the voice handle and the block pointer
/// from its own state, puts a byte count in `r2`, and then TESTS the return value. So the
/// fourth argument is a real out-parameter (not a leftover register) and the return code
/// is acted on.
///
/// The block holds one or more `(header, params)` entries back to back. Each is applied
/// through the SAME path a single unlocked params buffer takes, so an AT9 source set this
/// way plays exactly as one set the other way. The walk is self-checking: if the entries
/// do not tile `uSize` exactly then this reading of the layout is wrong for that block,
/// and it says so instead of applying whatever it happened to land on.
#[hostcall]
pub(super) fn voice_set_params_block(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    voice: u32,
    block: Ptr,
    size: u32,
    error_count: Ptr,
) -> i32 {
    let mut applied = 0u32;
    if !block.is_null() {
        let mut off = 0u32;
        while off + NGS_MODULE_PARAM_HEADER_BYTES + 8 <= size {
            let base = block.addr() + off;
            let module = ctx.read_u32(base);
            let params = base + NGS_MODULE_PARAM_HEADER_BYTES;
            let entry_bytes = ctx.read_u32(params + NGS_PARAMS_DESC_SIZE_OFF);
            // Module 0 is the source player; its params carry the AT9 buffer + config.
            // Of the rest, only the buss module is understood - see `set_module_params`.
            if module == 0 {
                st.audio_state.at9.set_player_params(ctx, voice, params);
            } else {
                st.audio_state.at9.set_module_params(ctx, voice, params);
            }
            applied += 1;
            // A zero or absurd size cannot be stepped over; stop rather than spin.
            if entry_bytes == 0 || entry_bytes > size {
                break;
            }
            off += NGS_MODULE_PARAM_HEADER_BYTES + entry_bytes;
        }
        if off != size {
            report_block_layout_mismatch(size, off, applied);
        }
    }
    // The caller reads this back; leaving its initialised local alone would let it act on
    // whatever it happened to put there.
    if !error_count.is_null() {
        ctx.write_u32(error_count.addr(), 0);
    }
    0
}

/// Say so, once, when a params block does not tile exactly - the entry layout above is
/// REd from one title's music path, and a block that does not fit it is evidence that the
/// reading is incomplete, not something to apply silently.
fn report_block_layout_mismatch(size: u32, walked: u32, entries: u32) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DONE: AtomicBool = AtomicBool::new(false);
    if !DONE.swap(true, Ordering::Relaxed) {
        eprintln!(
            "sceNgsVoiceSetParamsBlock: block of {size} bytes did not tile into \
             (8-byte header + descriptor size) entries - walked {walked} bytes in \
             {entries} entr(ies). The entry layout is REd from one title; the remainder \
             was NOT applied."
        );
    }
}

#[cfg(test)]
mod lock_params_tests {
    //! The lock/write/unlock cycle a title's music takes, end to end against real
    //! `VitaState` and guest memory.
    //!
    //! These pin the contract whose absence made EVERY title silent: a params buffer
    //! handed back by `sceNgsVoiceLockParams` must already carry its
    //! `SceNgsParamsDescriptor` id, because the title never writes one and the AT9
    //! reader rejects the source without it. The failure had no symptom other than
    //! correctly-paced digital silence, which is why it survived a working decoder, a
    //! working mixer and a working browser sink.
    use super::*;
    use crate::host::{GuestCtx, SliceMemory, VFP_ARG_COUNT};
    use crate::world::DeterministicWorld;
    use vitaslop_transpiler::abi::REG_COUNT;

    /// The AT9 params descriptor id, as `vita::at9` matches it.
    const AT9_PLAYER_ID: u32 = 0x0101_5caa;
    /// Offsets inside the player params, mirrored from `vita::at9`.
    const OFF_BUFFER_PTR: u32 = 0x08;
    const OFF_BUFFER_BYTES: u32 = 0x0c;
    const OFF_CHANNELS: u32 = 0x58;
    const OFF_CONFIG: u32 = 0x5c;

    /// A real ATRAC9 config word (48 kHz stereo), taken from a title's music voice. It
    /// is a four-byte format descriptor, and the decoder must accept it for the voice
    /// to start - which is what makes `plays` below a genuine end-to-end assertion
    /// rather than a check that we set a flag.
    const AT9_CONFIG: [u8; 4] = [0xfe, 0x74, 0x09, 0xf0];

    /// Guest memory large enough for `galloc` (which starts a megabyte above the base)
    /// plus the 5 MB main-stack reserve it refuses to encroach on.
    const MEM_BYTES: u32 = 16 * 1024 * 1024;

    struct Harness {
        st: VitaState,
        mem: Vec<u8>,
        regs: [u32; REG_COUNT],
        vfp: [u32; VFP_ARG_COUNT],
    }

    impl Harness {
        fn new() -> Harness {
            Harness {
                st: VitaState::new(0, MEM_BYTES, Box::new(DeterministicWorld::default())),
                mem: vec![0u8; MEM_BYTES as usize],
                regs: [0u32; REG_COUNT],
                vfp: [0u32; VFP_ARG_COUNT],
            }
        }

        /// Call a host handler with the given guest arguments in r0..r3.
        fn call(&mut self, f: fn(&mut GuestCtx, &mut VitaState), args: [u32; 4]) {
            self.regs[..4].copy_from_slice(&args);
            let mut mem = SliceMemory(&mut self.mem);
            let mut ctx = GuestCtx::new(&mut self.regs, &mut self.vfp, &mut mem, 0);
            f(&mut ctx, &mut self.st);
        }

        fn read_u32(&self, addr: u32) -> u32 {
            let a = addr as usize;
            u32::from_le_bytes([self.mem[a], self.mem[a + 1], self.mem[a + 2], self.mem[a + 3]])
        }

        fn write_u32(&mut self, addr: u32, v: u32) {
            let a = addr as usize;
            self.mem[a..a + 4].copy_from_slice(&v.to_le_bytes());
        }

        /// Lock module 0's AT9 player params and return the buffer the title gets.
        fn lock(&mut self, voice: u32) -> u32 {
            let info = 0x2000u32; // somewhere harmless for the SceNgsBufferInfo out-param
            self.call(voice_lock_params, [voice, 0, AT9_PLAYER_ID, info]);
            self.read_u32(info)
        }

        /// Fill in a params buffer the way a title does: everything but the id.
        fn write_source(&mut self, buf: u32, data_ptr: u32, data_bytes: u32) {
            self.write_u32(buf + OFF_BUFFER_PTR, data_ptr);
            self.write_u32(buf + OFF_BUFFER_BYTES, data_bytes);
            self.write_u32(buf + OFF_CHANNELS, 2);
            let a = (buf + OFF_CONFIG) as usize;
            self.mem[a..a + 4].copy_from_slice(&AT9_CONFIG);
        }
    }

    /// THE REGRESSION. Lock must hand back a buffer that already names which params
    /// interface it is - the title asked for one by id and writes only the fields it
    /// wants to change.
    #[test]
    fn locked_params_buffer_carries_its_descriptor_id() {
        let mut h = Harness::new();
        let buf = h.lock(0x1234);
        assert_ne!(buf, 0, "lock must hand back a buffer");
        assert_eq!(
            h.read_u32(buf),
            AT9_PLAYER_ID,
            "the params buffer must carry the descriptor id the caller locked; a zero here \
             is the defect that made every AT9 source be rejected and every title silent"
        );
        assert_eq!(
            h.read_u32(buf + NGS_PARAMS_DESC_SIZE_OFF),
            96,
            "uSize must be the size of the PARAMS INTERFACE locked (96 for the AT9 player), \
             not the size of the block allocated for it - the readers check it to confirm they \
             are looking at the struct they were REd from, so a block size here makes every \
             voice that arrives this way refuse itself"
        );
    }

    /// The same buffer, filled in as a title fills it, is accepted as an AT9 source and
    /// the voice actually starts - which requires the real decoder to accept the config.
    #[test]
    fn a_locked_and_unlocked_source_plays() {
        let mut h = Harness::new();
        let voice = 0x1234;
        let buf = h.lock(voice);
        // A source buffer somewhere in guest memory; the bytes need not decode for the
        // voice to START, which is the step this test is about.
        h.write_source(buf, 0x0020_0000, 0x3fc0);
        h.call(voice_unlock_params, [voice, 0, 0, 0]);
        h.call(voice_play, [voice, 0, 0, 0]);
        assert!(
            h.st.audio_state.at9.any_playing(),
            "after lock -> write -> unlock -> play the voice must be playing; if it is not, \
             the source was rejected and the title will be silent"
        );
    }

    /// The negative control, which is what the engine did before: with the descriptor id
    /// cleared, the identical params are refused. Without this a future change that
    /// stops writing the id would leave both tests above passing for the wrong reason
    /// only if it also broke lock - this pins the id itself as the load-bearing field.
    #[test]
    fn a_source_without_its_descriptor_id_is_refused() {
        let mut h = Harness::new();
        let voice = 0x1234;
        let buf = h.lock(voice);
        h.write_source(buf, 0x0020_0000, 0x3fc0);
        h.write_u32(buf, 0); // undo the descriptor id
        h.call(voice_unlock_params, [voice, 0, 0, 0]);
        h.call(voice_play, [voice, 0, 0, 0]);
        assert!(
            !h.st.audio_state.at9.any_playing(),
            "a params buffer with no descriptor id is not an AT9 source and must not play"
        );
    }
}
