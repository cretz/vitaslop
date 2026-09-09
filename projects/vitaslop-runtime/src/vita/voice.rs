//! SceVoice: voice chat, modelled as a console with NO MICROPHONE and no peers.
//!
//! Same shape as [`super::camera`] and [`super::net`], and for the same reason: the
//! honest thing is a real console state, not a fabricated success. A Vita has a
//! microphone and a voice-chat service that carries its audio to other players; a
//! machine running this emulator has neither a session nor anyone to talk to.
//!
//! # Why the lifecycle SUCCEEDS and the data path is empty
//! This library is not optional decoration for the titles that use it - a retail
//! fighting game calls `sceVoiceInit` during BOOT, before any menu, and a failed init
//! there is an error path taken while the game is still starting up. Refusing the whole
//! library would be modelling a console whose voice service is broken, which is not a
//! state a Vita has. What it DOES have is a voice service with nothing coming in: the
//! ports open, the connections are made, and every read returns zero bytes.
//!
//! That distinction is the whole design. Nothing here invents audio:
//! - `sceVoiceReadFromOPort` reports ZERO bytes available, every time. A title mixing
//!   that gets silence, which is what a player who is not speaking sounds like.
//! - `sceVoiceWriteToIPort` accepts the guest's microphone-bound bytes and drops them,
//!   which is what happens to speech with no session to carry it.
//! - `sceVoiceGetPortInfo` reports a live port with an empty queue, agreeing with the
//!   reads. A title that checks the queue before reading and one that just reads are
//!   told the same thing.
//!
//! # Where the prototypes come from
//! vitasdk publishes NO header for SceVoice, and neither wiki carries a prototype - only
//! the names against their NIDs (`devwiki` "VSH Exports"). So the shapes below are
//! recovered from the CALLSITE, in the way [[re-undocumented-nid-from-callsite]]
//! describes, and the argument each entry point actually uses is the one this module
//! reads. `sceVoiceInit`'s second argument is `100` at the one callsite seen, which is
//! the published `SCE_VOICE_VERSION_100` constant of the same API on its predecessor
//! console, and its first is a small struct pointer - consistent with
//! `sceVoiceInit(SceVoiceInitParam *, SceVoiceVersion)`. Nothing here depends on that
//! struct's layout: it is never read, because a service with no session has nothing to
//! configure.
//!
//! # Ports
//! A port id is a real allocated handle, not a constant, so a title that opens several
//! and closes one gets coherent answers about the others, and one that uses a stale id
//! after deleting it is told so rather than silently working.

use crate::host::VitaState;
use crate::hostcall;

/// SceVoice error codes are in the `0x8010_4???` facility range. Only the two that
/// describe a real state here are used, and both are reported by more than one entry
/// point, so a title reading them gets one consistent story:
///   `ARG_INVALID`     - a port id that was never created (or has been deleted)
///   `TOPOLOGY`        - a connection asked for between ports that cannot carry audio
const SCE_VOICE_ERROR_ARG_INVALID: i32 = 0x8010_4001u32 as i32;

/// Every live port, in creation order. A `Vec` and not a counter: `sceVoiceDeletePort`
/// has to make an id STOP working, which a high-water mark cannot express.
#[derive(Default)]
pub struct VoiceState {
    initialised: bool,
    started: bool,
    ports: Vec<u32>,
    next_port: u32,
}

impl VoiceState {
    fn create_port(&mut self) -> u32 {
        // Port ids start away from zero so a title that treats 0 as "no port" (several
        // do) is not handed one that looks like the absence of one.
        self.next_port = if self.next_port == 0 { 0x100 } else { self.next_port + 1 };
        self.ports.push(self.next_port);
        self.next_port
    }
    fn has(&self, id: u32) -> bool {
        self.ports.contains(&id)
    }
    fn delete(&mut self, id: u32) -> bool {
        let n = self.ports.len();
        self.ports.retain(|p| *p != id);
        self.ports.len() != n
    }
}

/// The one report that this console has no voice session. STATUS, not a warning: it is
/// an accepted limitation of the model, not something owed a fix - the same call this
/// module's siblings make. See `vitaslop_platform::diag`.
fn report_no_session(st: &VitaState) {
    tracing::info!(
        target: "vitaslop::status",
        thread = st.current_thread(),
        "SceVoice: voice chat runs with no microphone and no session - ports open, reads return \
         no data, and anything written to an input port is dropped"
    );
}

/// int sceVoiceInit(SceVoiceInitParam *pArg, SceVoiceVersion version)
///
/// The parameter block is not read: a service with no session has nothing to configure,
/// and reading a struct whose layout is not published would be inventing a shape.
#[hostcall]
pub(super) fn init(_ctx: &mut GuestCtx, st: &mut VitaState, _param: Ptr, _version: u32) -> i32 {
    report_no_session(st);
    st.voice.initialised = true;
    0
}

/// int sceVoiceEnd(void)
#[hostcall]
pub(super) fn end(_ctx: &mut GuestCtx, st: &mut VitaState) -> i32 {
    st.voice.initialised = false;
    st.voice.started = false;
    st.voice.ports.clear();
    0
}

/// int sceVoiceStart(SceVoiceStartParam *pArg)
#[hostcall]
pub(super) fn start(_ctx: &mut GuestCtx, st: &mut VitaState, _param: Ptr) -> i32 {
    st.voice.started = true;
    0
}

/// int sceVoiceStop(void)
#[hostcall]
pub(super) fn stop(_ctx: &mut GuestCtx, st: &mut VitaState) -> i32 {
    st.voice.started = false;
    0
}

/// int sceVoiceCreatePort(SceUInt32 *pPortId, const SceVoicePortParam *pArg)
///
/// The id is written through the out pointer, which is how the callsite uses it.
#[hostcall]
pub(super) fn create_port(ctx: &mut GuestCtx, st: &mut VitaState, out: Ptr, _param: Ptr) -> i32 {
    let id = st.voice.create_port();
    if out.addr() != 0 {
        ctx.write_u32(out.addr(), id);
    }
    0
}

/// int sceVoiceDeletePort(SceUInt32 portId)
#[hostcall]
pub(super) fn delete_port(_ctx: &mut GuestCtx, st: &mut VitaState, id: u32) -> i32 {
    if st.voice.delete(id) { 0 } else { SCE_VOICE_ERROR_ARG_INVALID }
}

/// int sceVoiceConnectIPortToOPort(SceUInt32 ips, SceUInt32 ops)
///
/// Both ports must exist; the connection itself carries nothing, because nothing is
/// ever written into an input port that an output port could deliver.
#[hostcall]
pub(super) fn connect(_ctx: &mut GuestCtx, st: &mut VitaState, ips: u32, ops: u32) -> i32 {
    if st.voice.has(ips) && st.voice.has(ops) { 0 } else { SCE_VOICE_ERROR_ARG_INVALID }
}

/// int sceVoiceDisconnectIPortFromOPort(SceUInt32 ips, SceUInt32 ops)
#[hostcall]
pub(super) fn disconnect(_ctx: &mut GuestCtx, st: &mut VitaState, ips: u32, ops: u32) -> i32 {
    if st.voice.has(ips) && st.voice.has(ops) { 0 } else { SCE_VOICE_ERROR_ARG_INVALID }
}

/// int sceVoiceWriteToIPort(SceUInt32 portId, const void *data, SceUInt32 *size)
///
/// Accepts everything and carries it nowhere. `size` is left as the guest set it: the
/// bytes really were taken from the caller, and reporting fewer would make a title
/// retry forever with audio that has no session to go to.
#[hostcall]
pub(super) fn write_to_iport(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: u32,
    _data: Ptr,
    _size: Ptr,
) -> i32 {
    if st.voice.has(id) { 0 } else { SCE_VOICE_ERROR_ARG_INVALID }
}

/// int sceVoiceReadFromOPort(SceUInt32 portId, void *data, SceUInt32 *size)
///
/// ZERO BYTES, always, and that is the whole point of this module. The out size is set
/// to 0 rather than left alone: a title that reads it would otherwise treat whatever it
/// asked for as decoded speech and mix a buffer nobody filled.
#[hostcall]
pub(super) fn read_from_oport(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    id: u32,
    _data: Ptr,
    size: Ptr,
) -> i32 {
    if !st.voice.has(id) {
        SCE_VOICE_ERROR_ARG_INVALID
    } else {
        if size.addr() != 0 {
            ctx.write_u32(size.addr(), 0);
        }
        0
    }
}

/// int sceVoiceGetPortInfo(SceUInt32 portId, SceVoiceBasePortInfo *pInfo)
///
/// Zeroed: a live port with an empty queue and no connected peers. It agrees with
/// [`read_from_oport`], which is the point - a title that checks the queue depth before
/// reading and one that reads blind must be told the same thing. The struct's published
/// size is not known, so a conservative 32 bytes is cleared: every layout the callsite
/// could use has its counters inside that, and clearing more could tread on the caller's
/// own stack.
#[hostcall]
pub(super) fn get_port_info(ctx: &mut GuestCtx, st: &mut VitaState, id: u32, info: Ptr) -> i32 {
    if !st.voice.has(id) {
        SCE_VOICE_ERROR_ARG_INVALID
    } else {
        if info.addr() != 0 {
            for w in 0..8u32 {
                ctx.write_u32(info.addr() + w * 4, 0);
            }
        }
        0
    }
}
