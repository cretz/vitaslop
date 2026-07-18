//! System and online services touched during boot: SceSysmodule, SceNet /
//! SceNetCtl, SceHttp, SceSsl, SceNpManager / SceNpBasic, SceRtc, the SceFios2
//! overlay list, and the libult object manager.
//!
//! None of these have a backing service off-console, and the engine is offline by
//! design. The faithful model is "initialized but offline": every init succeeds so
//! the title proceeds past its startup checks, connection state reads as
//! disconnected, and callback registration succeeds but the callback never fires
//! (there is no network event to deliver). This keeps a title on its offline path -
//! menus and gameplay - without waiting on a connection that will never come.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;

/// SceNetCtl connection state: disconnected (no link).
const SCE_NETCTL_STATE_DISCONNECTED: u32 = 0;
/// SceSysmodule "module is loaded" status.
const SCE_SYSMODULE_LOADED: i32 = 0;
/// A generic "no online account / not signed in" error, returned by the online
/// identity calls so the title takes its offline path instead of dereferencing an
/// account it will never get.
pub(super) const SCE_NP_ERROR_SIGNED_OUT: i32 = 0x8055_0605u32 as i32;

/// SceAppUtil system-parameter IDs (from the MIT `psp2/system_param.h`).
const SCE_SYSTEM_PARAM_ID_LANG: u32 = 1;
const SCE_SYSTEM_PARAM_ID_ENTER_BUTTON: u32 = 2;
/// The username system-parameter id (the sole string parameter).
const SCE_SYSTEM_PARAM_ID_USERNAME: u32 = 3;
/// Language: English (US) - a neutral Western default. The title id is a North
/// American release; the old blanket-0 reported `LANG = Japanese` (enum value 0).
const SCE_SYSTEM_PARAM_LANG_ENGLISH_US: u32 = 1;
/// Enter-button assignment: Cross. The Western "enter" is Cross (enum: Circle = 0,
/// Cross = 1); blanket-0 reported Circle, the Japanese assignment, which can swap a
/// title's confirm/cancel in any button-navigable menu.
const SCE_SYSTEM_PARAM_ENTER_BUTTON_CROSS: u32 = 1;

/// int sceAppUtilSystemParamGetInt(SceAppUtilSystemParamId id, SceInt32 *value)
/// Report a faithful default for each queried system parameter, written through
/// `value`. Language and the enter-button assignment are the two that actually
/// affect behaviour (locale selection, confirm/cancel mapping); everything else gets
/// a safe, in-range 0 rather than an uninitialized read. NOTE: this is a faithfulness
/// fix, not a fix for a touch-driven front-end (that is `sceTouchRead`'s job); on such
/// a title no button, Cross or Circle, drives a game-drawn touch dialog.
#[hostcall]
pub(super) fn apputil_system_param_get_int(ctx: &mut GuestCtx, _st: &mut VitaState, id: u32, value: Ptr) -> i32 {
    if !value.is_null() {
        let v = match id {
            SCE_SYSTEM_PARAM_ID_LANG => SCE_SYSTEM_PARAM_LANG_ENGLISH_US,
            SCE_SYSTEM_PARAM_ID_ENTER_BUTTON => SCE_SYSTEM_PARAM_ENTER_BUTTON_CROSS,
            _ => 0,
        };
        ctx.write_u32(value.addr(), v);
    }
    0
}

/// A neutral account username, reported by `sceAppUtilSystemParamGetString`. Off
/// console there is no signed-in account; a short ASCII name keeps any title that
/// stamps the player's name into a save or HUD on its normal path (an empty string
/// can trip a "no name set" branch on some titles).
const DEFAULT_USERNAME: &[u8] = b"Player";

/// int sceAppUtilSystemParamGetString(SceUInt32 id, SceChar8 *buf, SceSize bufSize)
/// Write a NUL-terminated system-parameter string into the caller's buffer. The only
/// string parameter is the username; anything else yields an empty string. Truncates
/// to `bufSize` (always leaving room for the terminator), like the real call.
#[hostcall]
pub(super) fn apputil_system_param_get_string(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    id: u32,
    buf: Ptr,
    buf_size: u32,
) -> i32 {
    if !buf.is_null() && buf_size != 0 {
        let src: &[u8] = match id {
            SCE_SYSTEM_PARAM_ID_USERNAME => DEFAULT_USERNAME,
            _ => b"",
        };
        // Reserve one byte for the NUL terminator.
        let max = (buf_size as usize).saturating_sub(1).min(src.len());
        let mut out = src[..max].to_vec();
        out.push(0);
        ctx.write_bytes(buf.addr(), &out);
    }
    0
}

/// int sceSysmoduleIsLoaded(SceSysmoduleModuleId id)
/// Report every queried module as already loaded, so the title does not block on a
/// load we cannot perform (the modules it needs are linked in already).
#[hostcall]
pub(super) fn sysmodule_is_loaded(_st: &mut VitaState, _id: u32) -> i32 {
    SCE_SYSMODULE_LOADED
}

/// int sceNetCtlInetGetState(int *state)
/// Always disconnected: there is no network link.
#[hostcall]
pub(super) fn netctl_inet_get_state(ctx: &mut GuestCtx, _st: &mut VitaState, state: Ptr) -> i32 {
    if !state.is_null() {
        ctx.write_u32(state.addr(), SCE_NETCTL_STATE_DISCONNECTED);
    }
    0
}

/// int sceNetCtlInetRegisterCallback(SceNetCtlCallback func, void *arg, int *cid)
/// Records the callback so [`net_check_callback`] can deliver the one-time offline
/// state, and hands back a callback id.
#[hostcall]
pub(super) fn netctl_register_callback(ctx: &mut GuestCtx, st: &mut VitaState, func: Ptr, arg: Ptr, cid: Ptr) -> i32 {
    st.set_net_inet_callback(func.addr(), arg.addr());
    tracing::debug!(target: "vitaslop::cb", func = format_args!("{:#x}", func.addr()), arg = format_args!("{:#x}", arg.addr()), "netCtlRegisterCallback");
    if !cid.is_null() {
        ctx.write_u32(cid.addr(), 0);
    }
    0
}

/// SceNetCtlState the callback reports: no link off-console.
const SCE_NETCTL_STATE_DISCONNECTED_EVT: u32 = 0;

/// int sceNetCtlCheckCallback(void)
/// The per-frame pump. Delivers the registered inet callback once with the
/// disconnected state, so a title waiting for its network-state notification
/// before leaving its boot screen proceeds. The callback runs on its own fiber
/// under the preemptive scheduler (see [`VitaState::pump_net_callback`]).
#[hostcall]
pub(super) fn net_check_callback(_ctx: &mut GuestCtx, st: &mut VitaState) -> i32 {
    st.pump_net_callback(SCE_NETCTL_STATE_DISCONNECTED_EVT);
    0
}

/// int sceNpRegisterServiceStateCallback(SceNpServiceStateCallback cb, void *userdata)
/// Records the callback so [`np_check_callback`] can deliver the one-time offline
/// (signed-out) service state a title's boot state machine waits for.
#[hostcall]
pub(super) fn np_register_service_state_callback(_ctx: &mut GuestCtx, st: &mut VitaState, cb: Ptr, userdata: Ptr) -> i32 {
    st.set_np_service_callback(cb.addr(), userdata.addr());
    tracing::debug!(target: "vitaslop::cb", cb = format_args!("{:#x}", cb.addr()), userdata = format_args!("{:#x}", userdata.addr()), "npRegisterServiceStateCallback");
    0
}

/// SceNpServiceState: the account/service sign-in state. Off-console there is no
/// PSN account, so the faithful state is signed-out - a title then takes its
/// offline path (local play) instead of waiting on an online session. The enum
/// (0=UNKNOWN, 1=SIGNED_OUT, 2=SIGNED_IN, 3=ONLINE) is passed as the callback's
/// first argument; verified against the game's own handler, which does
/// `switch(state)` on it.
const SCE_NP_SERVICE_STATE_SIGNED_OUT: u32 = 1;

/// int sceNpCheckCallback(void)
/// The per-frame pump for the NP service-state callback. Delivers the callback
/// once with the signed-out state, so a title gating its boot on the PSN sign-in
/// notification proceeds to its offline menu. The callback receives the state enum
/// in its first argument and the registered userdata as its `this` (see the ABI
/// note on [`VitaState::pump_np_callback`]).
pub(super) fn np_check_callback(ctx: &mut GuestCtx, st: &mut VitaState) {
    st.pump_np_callback(SCE_NP_SERVICE_STATE_SIGNED_OUT);
    ctx.ret(0);
}

// --- SceCommonDialog families -------------------------------------------------
//
// System-drawn dialogs (trophy setup, message boxes, the network check, savedata
// UI, ...). Off-console there is no system UI to draw, so the faithful offline
// model is a dialog that completes INSTANTLY: `Init` opens it, the very next
// `GetStatus` reports FINISHED, and `Term` closes it. A title that opens one at
// boot (e.g. the trophy-setup dialog) then busy-waits on its status proceeds
// immediately instead of spinning forever on a dialog no one can dismiss.
// `GetStatus` on a family that was never opened reports NONE, so a state machine
// polling before `Init` is not tricked into seeing a phantom dialog close.

/// SceCommonDialogStatus: no dialog open / dialog completed. (RUNNING is never
/// reported - our dialogs finish instantly.)
const DIALOG_STATUS_NONE: i32 = 0;
const DIALOG_STATUS_FINISHED: i32 = 2;

/// One bit per dialog family in [`VitaState::open_dialogs`].
#[derive(Clone, Copy)]
pub(super) enum DialogFamily {
    Msg = 0,
    NetCheck = 1,
    SaveData = 2,
    NpMessage = 3,
    NpTrophySetup = 4,
    StoreCheckout = 5,
    NpSnsFacebook = 6,
}

/// `*DialogInit`: mark the family open; the dialog will report finished on the
/// next status poll. Every family's init succeeds offline.
pub(super) fn dialog_init(ctx: &mut GuestCtx, st: &mut VitaState, family: DialogFamily) {
    st.open_dialogs |= 1 << family as u32;
    ctx.ret(0);
}

/// `*DialogGetStatus`: FINISHED once opened, NONE before. The return value IS the
/// status (these calls return `SceCommonDialogStatus`, not an errno).
pub(super) fn dialog_get_status(ctx: &mut GuestCtx, st: &mut VitaState, family: DialogFamily) {
    let open = st.open_dialogs & (1 << family as u32) != 0;
    ctx.ret(if open { DIALOG_STATUS_FINISHED } else { DIALOG_STATUS_NONE } as u32);
}

/// `*DialogTerm`: close the family. Also the landing spot for the lifecycle
/// helpers that need nothing from us (`Continue`/`Finish`/`SubClose`/`Abort`
/// route to [`dialog_ok`]).
pub(super) fn dialog_term(ctx: &mut GuestCtx, st: &mut VitaState, family: DialogFamily) {
    st.open_dialogs &= !(1 << family as u32);
    ctx.ret(0);
}

/// `*DialogGetResult` and the other lifecycle calls that just succeed: return 0
/// and leave the caller's result struct as the caller prepared it (a zeroed
/// result reads as "OK / no selection" in every family).
pub(super) fn dialog_ok(ctx: &mut GuestCtx, _st: &mut VitaState) {
    ctx.ret(0);
}

/// Microseconds from the SceRtc epoch (0001-01-01) to the Unix epoch (1970-01-01):
/// 719162 days. RTC ticks count from the former; the world clock from the latter.
const RTC_UNIX_EPOCH_TICKS: u64 = 719_162 * 86_400 * 1_000_000;

/// int sceRtcGetCurrentTick(SceRtcTick *tick)
/// The current time as a 64-bit microsecond tick since 0001-01-01, from the world
/// wall clock. A title polls this every frame for wall-time deltas; an unwritten
/// out-param would leave its delta computation on stack garbage.
#[hostcall]
pub(super) fn rtc_get_current_tick(ctx: &mut GuestCtx, st: &mut VitaState, tick: Ptr) -> i32 {
    if !tick.is_null() {
        let t = RTC_UNIX_EPOCH_TICKS + st.world.wall_us();
        ctx.write_u32(tick.addr(), t as u32);
        ctx.write_u32(tick.addr() + 4, (t >> 32) as u32);
    }
    0
}

/// Size of SceMotionState (vitasdk asserts 0xF8).
const MOTION_STATE_SIZE: usize = 0xF8;

/// int sceMotionGetState(SceMotionState *motionState)
/// A device at rest, flat: zero acceleration/velocity, identity orientation
/// (quaternion and both matrices), timestamps from the virtual clock. The whole
/// struct is written so a title reading any field sees a defined, neutral pose
/// rather than stack garbage steering its camera.
#[hostcall]
pub(super) fn motion_get_state(ctx: &mut GuestCtx, st: &mut VitaState, state: Ptr) -> i32 {
    if !state.is_null() {
        let now = st.world.monotonic_us();
        let mut buf = [0u8; MOTION_STATE_SIZE];
        buf[0..4].copy_from_slice(&(now as u32).to_le_bytes()); // timestamp
        let one = 1.0f32.to_le_bytes();
        buf[52..56].copy_from_slice(&one); // deviceQuat.w (x,y,z stay 0)
        for d in 0..4 {
            // rotationMatrix @56 and nedMatrix @120: identity diagonals.
            let i = 56 + d * 20;
            buf[i..i + 4].copy_from_slice(&one);
            let j = 120 + d * 20;
            buf[j..j + 4].copy_from_slice(&one);
        }
        buf[200..208].copy_from_slice(&now.to_le_bytes()); // hostTimestamp
        ctx.write_bytes(state.addr(), &buf);
    }
    0
}

/// SCE_APPUTIL_ERROR_DRM_NO_ENTITLEMENT: the queried additional content has no
/// license on this device.
const SCE_APPUTIL_ERROR_DRM_NO_ENTITLEMENT: i32 = 0x8010_0660u32 as i32;

/// int sceAppUtilDrmOpen(const SceAppUtilDrmAddcontId *dirName, SceAppUtilMountPoint *mountPoint)
/// No additional content is installed offline, so every entitlement query fails
/// with NO_ENTITLEMENT. A blanket-success stub here is a trap: the title then
/// tries to mount/read the (absent) addcont data, fails, and retries the open
/// every frame - wedging its boot flow on a DLC probe loop.
#[hostcall]
pub(super) fn apputil_drm_open(ctx: &mut GuestCtx, _st: &mut VitaState, dir: Ptr, _mount: Ptr) -> i32 {
    let name = if dir.is_null() {
        String::new()
    } else {
        let raw = ctx.read_bytes(dir.addr(), 16);
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        String::from_utf8_lossy(&raw[..end]).into_owned()
    };
    tracing::debug!(target: "vitaslop::cb", dir = %name, "sceAppUtilDrmOpen");
    SCE_APPUTIL_ERROR_DRM_NO_ENTITLEMENT
}

/// int sceAppUtilDrmClose(const SceAppUtilDrmAddcontId *dirName, SceAppUtilMountPoint *mountPoint)
#[hostcall]
pub(super) fn apputil_drm_close(_st: &mut VitaState, _dir: Ptr, _mount: Ptr) -> i32 {
    0
}

/// int sceNetCtlAdhocRegisterCallback(SceNetCtlCallback func, void *arg, int *cid)
/// Ad-hoc (device-to-device) networking has no peers off-console. Registration
/// succeeds and hands back a callback id through `cid`; the callback never fires
/// (there is no ad-hoc state change to deliver), so the title stays on its offline
/// path. An unwritten `cid` would leave the title's later unregister on stack garbage.
#[hostcall]
pub(super) fn netctl_adhoc_register_callback(ctx: &mut GuestCtx, _st: &mut VitaState, _func: Ptr, _arg: Ptr, cid: Ptr) -> i32 {
    if !cid.is_null() {
        ctx.write_u32(cid.addr(), 0);
    }
    0
}

/// int sceNpTrophyCreateContext(SceNpTrophyContext *context, const SceNpCommunicationId
///     *commId, const SceNpCommunicationSignature *commSig, SceUInt64 options)
/// Trophies work fully offline (they persist locally and unlock through the egress
/// ledger). Hand back a fresh non-zero context id so the title can create a handle
/// and query/unlock against it.
// `options` (SceUInt64) is the trailing argument and is not needed, so it is left
// unread rather than declared (the macro has no u64 value-arg class).
#[hostcall]
pub(super) fn np_trophy_create_context(ctx: &mut GuestCtx, st: &mut VitaState, context: Ptr, _comm_id: Ptr, _comm_sig: Ptr) -> i32 {
    if !context.is_null() {
        ctx.write_u32(context.addr(), st.new_handle());
    }
    0
}

/// int sceNpTrophyCreateHandle(SceNpTrophyHandle *handle)
/// Hand back a fresh non-zero trophy handle (used to scope async trophy operations).
#[hostcall]
pub(super) fn np_trophy_create_handle(ctx: &mut GuestCtx, st: &mut VitaState, handle: Ptr) -> i32 {
    if !handle.is_null() {
        ctx.write_u32(handle.addr(), st.new_handle());
    }
    0
}

/// int sceRtcGetCurrentClockLocalTime(SceDateTime *time)
/// Fill a fixed, deterministic local date/time. `SceDateTime` is
/// {u16 year, month, day, hour, minute, second; u32 microsecond}.
#[hostcall]
pub(super) fn rtc_get_current_clock_local_time(ctx: &mut GuestCtx, _st: &mut VitaState, time: Ptr) -> i32 {
    if !time.is_null() {
        let mut buf = [0u8; 16];
        buf[0..2].copy_from_slice(&2016u16.to_le_bytes()); // year
        buf[2..4].copy_from_slice(&1u16.to_le_bytes()); // month
        buf[4..6].copy_from_slice(&1u16.to_le_bytes()); // day
        // hour/minute/second/microsecond stay zero.
        ctx.write_bytes(time.addr(), &buf);
    }
    0
}
