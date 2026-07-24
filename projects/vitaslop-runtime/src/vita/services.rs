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

use crate::host::{GuestCtx, Ptr, VitaState};
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

/// int sceNpBasicGetFriendListEntryCount(SceUInt32 *count)
/// The number of PSN friends in the local NpBasic cache. Off-console there is no PSN
/// session and no friend list, so the faithful count is zero: a title enumerating
/// friends then retrieves none rather than reading stack garbage as a bogus count and
/// walking a friend array that was never filled. The kernel writes the count out-param;
/// an unwritten one would leave the title's loop bound undefined.
#[hostcall]
pub(super) fn np_basic_get_friend_list_entry_count(ctx: &mut GuestCtx, _st: &mut VitaState, count: Ptr) -> i32 {
    if !count.is_null() {
        ctx.write_u32(count.addr(), 0);
    }
    0
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

/// `SceMsgDialogButtonId`: which button the user pressed. INVALID (0) reads as "no
/// choice made yet"; a title that instantly finishes its prompt and sees INVALID
/// re-opens the same dialog every frame forever. YES/OK (1) is the affirmative
/// default - a MsgDialog offline off-console has no user, so reporting the affirmative
/// press is the faithful auto-confirm that lets the title's dialog-completion path run
/// its "button pressed" action and proceed.
const SCE_MSG_DIALOG_BUTTON_ID_YESOK: u32 = 1;
/// `SceMsgDialogResult.buttonId` field offset (after `mode`:i32 and `result`:i32).
const MSG_DIALOG_RESULT_BUTTON_ID_OFFSET: u32 = 8;

/// int sceMsgDialogGetResult(SceMsgDialogResult *result)
/// Unlike the other dialog GetResult calls (a zeroed result reads fine), a MsgDialog
/// carries the user's button choice, and a zeroed `buttonId` (INVALID) makes a title
/// re-prompt indefinitely. Off-console we auto-confirm the affirmative button so the
/// "could not connect - play offline?" prompt behind a boot online-check resolves and
/// boot proceeds. `mode`/`result` stay 0 (SCE_COMMON_DIALOG_RESULT_OK), as the caller
/// zeroed them.
pub(super) fn msg_dialog_get_result(ctx: &mut GuestCtx, _st: &mut VitaState) {
    let result = ctx.arg(0);
    if result != 0 {
        ctx.write_u32(result + MSG_DIALOG_RESULT_BUTTON_ID_OFFSET, SCE_MSG_DIALOG_BUTTON_ID_YESOK);
    }
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

/// SceRtc error codes (psp2common/kernel/rtc.h). sceRtcGetTick validates each field
/// of the broken-down time and reports the first one out of range, so a title that
/// probes validity by feeding a bad component sees the same rejection the kernel gives.
const SCE_RTC_ERROR_INVALID_POINTER: i32 = 0x8025_1001u32 as i32;
const SCE_RTC_ERROR_INVALID_YEAR: i32 = 0x8025_1081u32 as i32;
const SCE_RTC_ERROR_INVALID_MONTH: i32 = 0x8025_1082u32 as i32;
const SCE_RTC_ERROR_INVALID_DAY: i32 = 0x8025_1083u32 as i32;
const SCE_RTC_ERROR_INVALID_HOUR: i32 = 0x8025_1084u32 as i32;
const SCE_RTC_ERROR_INVALID_MINUTE: i32 = 0x8025_1085u32 as i32;
const SCE_RTC_ERROR_INVALID_SECOND: i32 = 0x8025_1086u32 as i32;
const SCE_RTC_ERROR_INVALID_MICROSECOND: i32 = 0x8025_1087u32 as i32;

/// True for a proleptic-Gregorian leap year (the rule sceRtcIsLeapYear applies).
fn is_leap_year(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Days in `month` (1..=12) of `year`, honoring February in a leap year.
fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Days from the civil date `y-m-d` (proleptic Gregorian) to 1970-01-01: the count of
/// days since the Unix epoch, negative for earlier dates. Howard Hinnant's algorithm,
/// exact across the whole SceRtcTime year range. Added to the RTC-to-Unix epoch offset
/// this yields the same 0001-01-01 timeline sceRtcGetCurrentTick reports.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719_468
}

/// int sceRtcGetTick(const SceRtcTime *pTime, SceRtcTick *pTick)
/// Convert a broken-down SceRtcTime {u16 year, month, day, hour, minute, second;
/// u32 microsecond} into a 64-bit microsecond tick since the RTC epoch 0001-01-01.
/// A title uses this to fold a date it built (a savedata timestamp, a scheduled event)
/// into one comparable scalar; a wrong or unwritten tick would corrupt every ordering
/// or delta it later derives. Each field is range-checked exactly as the kernel does so
/// an invalid time is rejected rather than silently producing a bogus tick, and the
/// epoch matches sceRtcGetCurrentTick so ticks from both share one timeline.
#[hostcall]
pub(super) fn rtc_get_tick(ctx: &mut GuestCtx, _st: &mut VitaState, time: Ptr, tick: Ptr) -> i32 {
    // The #[hostcall]-generated body cannot early-return, so the work (which needs
    // early exits for each validation failure) lives in a plain helper.
    rtc_get_tick_impl(ctx, time, tick)
}

fn rtc_get_tick_impl(ctx: &mut GuestCtx, time: Ptr, tick: Ptr) -> i32 {
    if time.is_null() || tick.is_null() {
        return SCE_RTC_ERROR_INVALID_POINTER;
    }
    let raw = ctx.read_bytes(time.addr(), 16);
    let u16_at = |off: usize| u16::from_le_bytes([raw[off], raw[off + 1]]) as i64;
    let year = u16_at(0);
    let month = u16_at(2);
    let day = u16_at(4);
    let hour = u16_at(6);
    let minute = u16_at(8);
    let second = u16_at(10);
    let micro = u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]) as i64;

    if year < 1 || year > 9999 {
        return SCE_RTC_ERROR_INVALID_YEAR;
    }
    if month < 1 || month > 12 {
        return SCE_RTC_ERROR_INVALID_MONTH;
    }
    if day < 1 || day > days_in_month(year, month) {
        return SCE_RTC_ERROR_INVALID_DAY;
    }
    if hour > 23 {
        return SCE_RTC_ERROR_INVALID_HOUR;
    }
    if minute > 59 {
        return SCE_RTC_ERROR_INVALID_MINUTE;
    }
    if second > 59 {
        return SCE_RTC_ERROR_INVALID_SECOND;
    }
    if micro > 999_999 {
        return SCE_RTC_ERROR_INVALID_MICROSECOND;
    }

    let days = days_from_civil(year, month, day);
    let secs = days * 86_400 + hour * 3_600 + minute * 60 + second;
    let t = RTC_UNIX_EPOCH_TICKS as i64 + secs * 1_000_000 + micro;
    ctx.write_u32(tick.addr(), t as u32);
    ctx.write_u32(tick.addr() + 4, (t >> 32) as u32);
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

/// SCE_APPUTIL_ERROR_SAVEDATA_SLOT_NOT_FOUND: the queried save-data slot does not exist.
const SCE_APPUTIL_ERROR_SAVEDATA_SLOT_NOT_FOUND: i32 = 0x8010_0641u32 as i32;
/// SCE_APPUTIL_ERROR_SAVEDATA_SLOT_EXISTS: create was asked to make a slot that already exists.
const SCE_APPUTIL_ERROR_SAVEDATA_SLOT_EXISTS: i32 = 0x8010_0640u32 as i32;
/// Size of a guest SceAppUtilSaveDataSlotParam (vitasdk asserts 0x34C): the localized
/// title/subtitle/detail plus icon path, user param, size, modified time, reserved.
const SAVEDATA_SLOT_PARAM_SIZE: usize = 0x34C;

/// Read the mount-point name (SceAppUtilSaveDataMountPoint, 16 opaque bytes) from a
/// guest pointer as a NUL-trimmed string, used as the savedata-store namespace so a
/// title with more than one mount keeps its slots distinct. A null pointer maps to the
/// empty mount (a title that always passes the same - possibly null - handle stays
/// self-consistent, which is all read-after-write needs).
fn read_mount_name(ctx: &GuestCtx, mount: Ptr) -> String {
    if mount.is_null() {
        return String::new();
    }
    let raw = ctx.read_bytes(mount.addr(), 16);
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

/// int sceAppUtilSaveDataSlotGetParam(unsigned int slotId,
///     SceAppUtilSaveDataSlotParam *param, SceAppUtilSaveDataMountPoint *mountPoint)
/// Read back a slot the title created earlier this session (in-memory savedata store).
/// A slot that was never created truthfully reports SAVEDATA_SLOT_NOT_FOUND - the title
/// then follows its "no save present" path (new game / defaults). On success the stored
/// param is copied back verbatim so create-then-get round-trips faithfully; a blanket-
/// success stub would instead leave `param` as stack garbage the title reads as a real
/// save. The kernel leaves `param` untouched on error, so we do too.
#[hostcall]
pub(super) fn apputil_savedata_slot_get_param(ctx: &mut GuestCtx, st: &mut VitaState, slot_id: u32, param: Ptr, mount: Ptr) -> i32 {
    let mount_name = read_mount_name(ctx, mount);
    match st.savedata_slot_get(&mount_name, slot_id) {
        Some(bytes) => {
            if !param.is_null() {
                ctx.write_bytes(param.addr(), &bytes);
            }
            tracing::debug!(target: "vitaslop::cb", slot = slot_id, "sceAppUtilSaveDataSlotGetParam -> OK");
            0
        }
        None => {
            tracing::debug!(target: "vitaslop::cb", slot = slot_id, "sceAppUtilSaveDataSlotGetParam -> SLOT_NOT_FOUND");
            SCE_APPUTIL_ERROR_SAVEDATA_SLOT_NOT_FOUND
        }
    }
}

/// int sceAppUtilSaveDataSlotCreate(SceUInt32 slotId,
///     const SceAppUtilSaveDataSlotParam *param, const SceAppUtilSaveDataMountPoint *mountPoint)
/// The title reached the fresh-save path (SaveDataSlotGetParam returned SLOT_NOT_FOUND,
/// it built a default param) and now creates the slot. We record the param blob in the
/// in-memory savedata store keyed by (mount, slot), so a later GetParam reads it back -
/// faithful read-after-write. Creating a slot that already exists reports SLOT_EXISTS,
/// as the kernel does. A null `param` stores a zeroed blob (the slot exists but carries
/// no metadata). Nothing is persisted to a real disk - the store is per-run and offline.
#[hostcall]
pub(super) fn apputil_savedata_slot_create(ctx: &mut GuestCtx, st: &mut VitaState, slot_id: u32, param: Ptr, mount: Ptr) -> i32 {
    let mount_name = read_mount_name(ctx, mount);
    if st.savedata_slot_exists(&mount_name, slot_id) {
        tracing::debug!(target: "vitaslop::cb", slot = slot_id, "sceAppUtilSaveDataSlotCreate -> SLOT_EXISTS");
        SCE_APPUTIL_ERROR_SAVEDATA_SLOT_EXISTS
    } else {
        let bytes = if param.is_null() {
            vec![0u8; SAVEDATA_SLOT_PARAM_SIZE]
        } else {
            ctx.read_bytes(param.addr(), SAVEDATA_SLOT_PARAM_SIZE)
        };
        st.savedata_slot_put(&mount_name, slot_id, bytes);
        tracing::debug!(target: "vitaslop::cb", slot = slot_id, "sceAppUtilSaveDataSlotCreate -> OK");
        0
    }
}

/// `SceAppUtilSaveDataFile` layout (psp2/apputil.h). `offset` is a 64-bit `SceOff`
/// 8-byte-aligned after the three leading words, so the struct is 64 bytes:
///   filePath*(0) buf*(4) bufSize(8) pad(12) offset:u64(16) mode(24) progDelta(28)
///   reserved[32](32).
const SAVEDATA_FILE_STRIDE: u32 = 64;
const SAVEDATA_FILE_PATH_OFF: u32 = 0;
const SAVEDATA_FILE_BUF_OFF: u32 = 4;
const SAVEDATA_FILE_BUFSIZE_OFF: u32 = 8;
const SAVEDATA_FILE_OFFSET_OFF: u32 = 16;
const SAVEDATA_FILE_MODE_OFF: u32 = 24;

/// int sceAppUtilSaveDataDataSave(SceAppUtilSaveDataFileSlot *slot,
///     SceAppUtilSaveDataFile *files, unsigned int fileNum,
///     SceAppUtilSaveDataMountPoint *mountPoint, SceSize *requiredSizeKiB)
/// Persist `fileNum` file writes into the title's savedata mount. Each entry names a
/// path relative to the mount, a source buffer, a size, and a byte offset. We apply
/// each write to the in-memory guest filesystem (the same store `sceIoOpen`/`Read`
/// serve), so a later read of the same path round-trips - a title that writes its
/// "online terms accepted" / progress record here and re-reads it next boot sees its
/// own data instead of a phantom. A directory-mode entry creates no file. Faithful
/// enough for bring-up: writes land in the guest FS; a future disk/OPFS backend can
/// persist them across sessions. `requiredSizeKiB` (an out estimate) is reported as 0.
#[hostcall]
pub(super) fn apputil_savedata_data_save(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    _slot: Ptr,
    files: Ptr,
    file_num: u32,
    mount: Ptr,
    required_kib: Ptr,
) -> i32 {
    let mount_name = read_mount_name(ctx, mount);
    for i in 0..file_num {
        let entry = files.addr() + i * SAVEDATA_FILE_STRIDE;
        let path_ptr = ctx.read_u32(entry + SAVEDATA_FILE_PATH_OFF);
        let buf_ptr = ctx.read_u32(entry + SAVEDATA_FILE_BUF_OFF);
        let buf_size = ctx.read_u32(entry + SAVEDATA_FILE_BUFSIZE_OFF);
        let offset = ctx.read_u32(entry + SAVEDATA_FILE_OFFSET_OFF);
        let mode = ctx.read_u32(entry + SAVEDATA_FILE_MODE_OFF);
        let rel = if path_ptr != 0 { ctx.read_cstr(path_ptr, 512) } else { String::new() };
        // Resolve to a mount-qualified path (e.g. "savedata0:/foo"); vfs_key normalizes.
        let full = if rel.contains(':') {
            rel.clone()
        } else {
            format!("{}/{}", mount_name.trim_end_matches('/'), rel.trim_start_matches('/'))
        };
        tracing::debug!(
            target: "vitaslop::io",
            path = %full, size = buf_size, offset, mode,
            "sceAppUtilSaveDataDataSave file",
        );
        // A directory-create entry carries no source buffer; every file mode (truncate
        // at 0, write-at-offset) has one. Skip the entries with nothing to write.
        if buf_ptr == 0 || buf_size == 0 {
            continue;
        }
        let data = ctx.read_bytes(buf_ptr, buf_size as usize);
        // Apply the write at `offset` over any existing content, growing as needed.
        let mut cur = st.read_file(&full).unwrap_or_default();
        let end = offset as usize + data.len();
        if cur.len() < end {
            cur.resize(end, 0);
        }
        cur[offset as usize..end].copy_from_slice(&data);
        st.add_file(&full, cur);
    }
    if !required_kib.is_null() {
        ctx.write_u32(required_kib.addr(), 0);
    }
    0
}

/// Size of SceAppMgrAppState (vitasdk asserts 0x80): {u32 systemEventNum, u32
/// appEventNum, SceBool isSystemUiOverlaid, u8 reserved[116]}.
const APP_MGR_APP_STATE_SIZE: usize = 0x80;

/// int _sceAppMgrGetAppState(SceAppMgrAppState *appState, SceSize len, uint32_t version)
/// Poll the app-lifecycle state. The title's main thread calls this every frame to learn
/// whether any system event (a notification, a resume, a system-UI overlay) is pending
/// before it drains them. Off-console the app runs foregrounded with no system shell
/// around it, so the honest state is zero pending system/app events and no overlay - the
/// title then skips its receive-event path (there is nothing to receive) rather than
/// reading stack garbage that could invent a phantom event count and loop draining events
/// that never arrive. Only `len` bytes are written (the caller-declared buffer size),
/// capped at the real struct size.
#[hostcall]
pub(super) fn app_mgr_get_app_state(ctx: &mut GuestCtx, _st: &mut VitaState, state: Ptr, len: u32, _version: u32) -> i32 {
    if !state.is_null() {
        let n = (len as usize).min(APP_MGR_APP_STATE_SIZE);
        ctx.write_bytes(state.addr(), &vec![0u8; n]);
    }
    0
}

/// SceAppUtilAppParamId SCE_APPUTIL_APPPARAM_ID_SKU_FLAG: the sole documented launch
/// app-param, read to tell a trial SKU from a full one. (The vitasdk headers only
/// typedef the id as `unsigned int`; the id and its values are from the SCE SDK.)
const SCE_APPUTIL_APPPARAM_ID_SKU_FLAG: u32 = 0;
/// SCE_APPUTIL_APPPARAM_SKU_FLAG_FULL: a full (non-trial) game. We boot a full retail
/// dump, so this is the faithful value - a blanket 0 is an undefined SKU flag that can
/// drop a title into trial mode (limited levels, nag screens).
const SCE_APPUTIL_APPPARAM_SKU_FLAG_FULL: u32 = 3;
/// SCE_APPUTIL_ERROR_PARAMETER: returned for an app-param id we carry no value for,
/// rather than inventing one.
const SCE_APPUTIL_ERROR_PARAMETER: i32 = 0x8010_0600u32 as i32;

/// int sceAppUtilAppParamGetInt(SceAppUtilAppParamId paramId, int *value)
/// Read an integer launch parameter. The only one titles query at boot is the SKU flag,
/// which we report as FULL (this is a full retail launch). Any other id has no backing
/// value off-console, so it truthfully errors rather than returning a fabricated int the
/// title would trust. The kernel leaves `value` untouched on error, so we do too.
#[hostcall]
pub(super) fn apputil_app_param_get_int(ctx: &mut GuestCtx, _st: &mut VitaState, param_id: u32, value: Ptr) -> i32 {
    if param_id == SCE_APPUTIL_APPPARAM_ID_SKU_FLAG {
        if !value.is_null() {
            ctx.write_u32(value.addr(), SCE_APPUTIL_APPPARAM_SKU_FLAG_FULL);
        }
        0
    } else {
        SCE_APPUTIL_ERROR_PARAMETER
    }
}

/// int sceLiveAreaGetStatus(...): query the app's LiveArea (its home-screen tile)
/// state. LiveArea lives on the system home screen, which does not exist off-console,
/// so the query simply succeeds and the title proceeds to its (no-op offline) LiveArea
/// update. The out-param layout (`SceLiveAreaStatus`) is undocumented in the permissive
/// headers; the caller passed a heap pointer in r0 but the observed boot advances with
/// it left untouched (the title reads the return code, not the struct), so we return 0
/// rather than fabricate a status word of unknown meaning. Revisit if a title is found
/// to branch on the struct contents.
#[hostcall]
pub(super) fn live_area_get_status(_st: &mut VitaState) -> i32 {
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

/// Zero every field of an OUT struct except its leading caller-set `.size` word, which
/// the trophy APIs take as an input version tag. Guarantees defined content regardless
/// of whether the caller pre-zeroed the buffer.
fn zero_out_struct_keep_size(ctx: &mut GuestCtx, p: Ptr, size: usize) {
    if p.is_null() {
        return;
    }
    let mut buf = vec![0u8; size];
    // Preserve the caller-written size field at offset 0.
    let sz = ctx.read_u32(p.addr());
    buf[0..4].copy_from_slice(&sz.to_le_bytes());
    ctx.write_bytes(p.addr(), &buf);
}

/// sizeof(SceNpTrophyGameDetails): {u32 size, numGroups, numTrophies, numPlatinum,
/// numGold, numSilver, numBronze; char title[128]; char description[1024]} = 0x49C.
/// Confirmed by the caller-set `.size` (=1180) this title writes before the call.
const NP_TROPHY_GAME_DETAILS_SIZE: usize = 0x49C;
/// sizeof(SceNpTrophyGameData): {u32 size, unlockedTrophies, unlockedPlatinum,
/// unlockedGold, unlockedSilver, unlockedBronze, progressPercentage} = 0x1C. Confirmed
/// by the caller-set `.size` (=28) this title writes before the call.
const NP_TROPHY_GAME_DATA_SIZE: usize = 0x1C;

/// int sceNpTrophyGetGameInfo(SceNpTrophyContext context, SceNpTrophyHandle handle,
///     SceNpTrophyGameDetails *details, SceNpTrophyGameData *data)
/// Report a defined EMPTY trophy set for this game: zero counts, empty title/description,
/// zero unlocks. Trophies have no backing service off-console, so an empty-but-defined
/// result is the honest state - the title reads "0 of 0 trophies" and takes its no-progress
/// path rather than looping over an invented count or reading stack garbage. Either OUT
/// pointer may be null (the caller can request only one of the two structs), so each is
/// filled independently. The caller-set `.size` version tag at offset 0 is preserved.
#[hostcall]
pub(super) fn np_trophy_get_game_info(ctx: &mut GuestCtx, _st: &mut VitaState, _context: u32, _handle: u32, details: Ptr, data: Ptr) -> i32 {
    zero_out_struct_keep_size(ctx, details, NP_TROPHY_GAME_DETAILS_SIZE);
    zero_out_struct_keep_size(ctx, data, NP_TROPHY_GAME_DATA_SIZE);
    0
}

/// sizeof(SceNpTrophyFlagArray): a fixed 128-bit unlock bitmap, {u32 flag_bits[4]} = 16
/// bytes (SCE_NP_TROPHY_FLAG_BITS_LENGTH = 128, one bit per trophy id). Fixed by the API,
/// not caller-set. With `count` reported as 0 the caller inspects no bits, so the exact
/// span barely matters, but we clear the full array to leave defined content.
const NP_TROPHY_FLAG_ARRAY_SIZE: usize = 16;

/// int sceNpTrophyGetTrophyUnlockState(SceNpTrophyContext context, SceNpTrophyHandle
///     handle, SceNpTrophyFlagArray *flags, SceUInt32 *count)
/// Report no unlocked trophies: an all-zero unlock bitmap and a total `count` of 0,
/// consistent with the empty trophy set sceNpTrophyGetGameInfo reports. The title reads
/// count 0 and skips iterating per-trophy info, so this is the tail of the offline trophy
/// query path. Both OUT pointers are written so neither is left as stack garbage.
#[hostcall]
pub(super) fn np_trophy_get_trophy_unlock_state(ctx: &mut GuestCtx, _st: &mut VitaState, _context: u32, _handle: u32, flags: Ptr, count: Ptr) -> i32 {
    if !flags.is_null() {
        ctx.write_bytes(flags.addr(), &[0u8; NP_TROPHY_FLAG_ARRAY_SIZE]);
    }
    if !count.is_null() {
        ctx.write_u32(count.addr(), 0);
    }
    0
}

/// Write the fixed, deterministic wall-clock date into a `SceDateTime`, which is
/// {u16 year, month, day, hour, minute, second; u32 microsecond}. Both RTC clock
/// getters serve the same constant so a title reading the struct sees a defined
/// value rather than stack garbage.
fn write_fixed_date_time(ctx: &mut GuestCtx, time: Ptr) {
    if !time.is_null() {
        let mut buf = [0u8; 16];
        buf[0..2].copy_from_slice(&2016u16.to_le_bytes()); // year
        buf[2..4].copy_from_slice(&1u16.to_le_bytes()); // month
        buf[4..6].copy_from_slice(&1u16.to_le_bytes()); // day
        // hour/minute/second/microsecond stay zero.
        ctx.write_bytes(time.addr(), &buf);
    }
}

/// int sceRtcGetCurrentClock(SceDateTime *time, int tzMinutes)
/// The UTC wall clock adjusted by a caller-supplied timezone offset in minutes.
/// We serve a fixed deterministic date regardless of `tz` (the emulated clock has
/// no real timezone), matching the local-time variant.
#[hostcall]
pub(super) fn rtc_get_current_clock(ctx: &mut GuestCtx, _st: &mut VitaState, time: Ptr, _tz: i32) -> i32 {
    write_fixed_date_time(ctx, time);
    0
}

/// int sceRtcGetCurrentClockLocalTime(SceDateTime *time)
/// Fill a fixed, deterministic local date/time.
#[hostcall]
pub(super) fn rtc_get_current_clock_local_time(ctx: &mut GuestCtx, _st: &mut VitaState, time: Ptr) -> i32 {
    write_fixed_date_time(ctx, time);
    0
}
