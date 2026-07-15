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
use std::sync::LazyLock;

/// Diagnostic: dump NGS param/output buffers to understand AT9 routing.
static TRACE: LazyLock<bool> = LazyLock::new(|| std::env::var("VITASLOP_TRACE_NGS").is_ok());

fn dump_mem(ctx: &GuestCtx, label: &str, addr: u32, len: usize) {
    if addr == 0 {
        eprintln!("  {label}: <null>");
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
    eprintln!("  {label} @ {addr:#010x}:{s}");
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
        if *TRACE {
            eprintln!("UnlockParams voice={voice:#x} module={module:#x}");
            dump_mem(ctx, "params", addr, 96);
            let data_ptr = ctx.read_u32(addr + 0x08);
            if data_ptr != 0 {
                dump_mem(ctx, "at9data", data_ptr, 32);
            }
        }
        // Module 0 is the source player; its params carry the AT9 buffer + config.
        if module == 0 {
            st.audio_state.at9.set_player_params(ctx, voice, addr);
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

/// Voice key-off / kill / pause - stop producing audio from this voice.
pub(super) fn voice_stop(ctx: &mut GuestCtx, st: &mut VitaState) {
    let voice = ctx.arg(0);
    st.audio_state.at9.stop(voice);
    ctx.ret(0);
}

/// SceInt32 sceNgsSystemUpdate(SceNgsHSynSystem system) - dump the surrounding
/// output/work region so the master-buss mix destination becomes visible.
pub(super) fn system_update(ctx: &mut GuestCtx, _st: &mut VitaState) {
    if *TRACE {
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
pub(super) fn rack_init(ctx: &mut GuestCtx, st: &mut VitaState, _system: u32, rack_buffer: Ptr, _desc: Ptr, handle: Ptr) -> i32 {
    let data = if rack_buffer.addr() != 0 { ctx.read_u32(rack_buffer.addr()) } else { 0 };
    let rack = if data != 0 { data } else { st.galloc(NGS_WORK_SIZE, 16) };
    if handle.addr() != 0 {
        ctx.write_u32(handle.addr(), rack);
    }
    0
}

/// SceInt32 sceNgsRackGetVoiceHandle(SceNgsHRack rack, SceUInt32 index, SceNgsHVoice *handle)
/// Hand back a distinct, valid block per voice so the title can hold and pass it.
#[hostcall]
pub(super) fn rack_get_voice_handle(ctx: &mut GuestCtx, st: &mut VitaState, _rack: u32, _index: u32, handle: Ptr) -> i32 {
    let voice = st.galloc(NGS_BLOCK_SIZE, 16);
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

/// SceInt32 sceNgsVoiceLockParams(SceNgsHVoice voice, SceUInt32 moduleId,
///                                SceUInt32 paramInterfaceId, SceNgsBufferInfo *buffer)
/// Return a stable, writable params buffer as `{ data, size }` so the title's
/// per-frame lock/edit/unlock cycle reuses one block instead of leaking each frame.
#[hostcall]
pub(super) fn voice_lock_params(ctx: &mut GuestCtx, st: &mut VitaState, voice: u32, module: u32, param: u32, buffer: Ptr) -> i32 {
    let key = (voice, module, param);
    let buf = match st.audio_state.ngs_param_buf(key) {
        Some(a) => a,
        None => {
            let a = st.galloc(NGS_BLOCK_SIZE, 16);
            st.audio_state.ngs_param_bufs.push((key, a));
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
/// `const SceNgsVoiceDefinition *` the title embeds in a rack description. It is an
/// opaque token to the guest, so one shared zeroed blob serves them all.
#[hostcall]
pub(super) fn voice_def_get(_ctx: &mut GuestCtx, st: &mut VitaState) -> u32 {
    if st.audio_state.ngs_def_blob == 0 {
        st.audio_state.ngs_def_blob = st.galloc(NGS_BLOCK_SIZE, 16);
    }
    st.audio_state.ngs_def_blob
}

/// SceInt32 sceNgsPatchCreateRouting(const SceNgsPatchSetupInfo *info, SceNgsHPatch *handle)
#[hostcall]
pub(super) fn patch_create_routing(ctx: &mut GuestCtx, st: &mut VitaState, _info: Ptr, handle: Ptr) -> i32 {
    let patch = st.galloc(NGS_BLOCK_SIZE, 16);
    if handle.addr() != 0 {
        ctx.write_u32(handle.addr(), patch);
    }
    0
}
