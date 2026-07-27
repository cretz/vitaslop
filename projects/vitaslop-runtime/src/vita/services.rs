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

/// `SCE_NET_CTL_ERROR_NOT_CONNECTED` - no network link to describe (psdevwiki error codes).
const SCE_NET_CTL_ERROR_NOT_CONNECTED: i32 = 0x8041_2108u32 as i32;

/// int sceNetCtlInetGetInfo(SceNetCtlInfoType code, SceNetCtlInfo *info)
/// Every query is about a connection that does not exist - [`netctl_inet_get_state`]
/// reports DISCONNECTED - so this reports NOT_CONNECTED, which is what a console with its
/// interface down answers.
///
/// The output union is deliberately left UNTOUCHED. Writing a zeroed IP address or an
/// empty SSID alongside an error is a lie a caller that ignores the return code cannot
/// detect, and "0.0.0.0" reads as a real answer; leaving the buffer alone means such a
/// caller sees whatever it initialized. The requested `code` is logged so a title that
/// needs a specific field answered shows up as a diagnostic rather than as a mystery.
#[hostcall]
pub(super) fn netctl_inet_get_info(
    _ctx: &mut GuestCtx,
    _st: &mut VitaState,
    code: u32,
    _info: Ptr,
) -> i32 {
    tracing::debug!(
        target: "vitaslop::err",
        code,
        "sceNetCtlInetGetInfo: no network link, reporting NOT_CONNECTED"
    );
    SCE_NET_CTL_ERROR_NOT_CONNECTED
}

/// int sceNetCtlAdhocGetInAddr(SceNetInAddr *inaddr)
/// The ad-hoc interface's IP address. There is no ad-hoc link off-console, so this reports
/// NOT_CONNECTED and leaves the output untouched, for the same reason
/// [`netctl_inet_get_info`] does: writing 0.0.0.0 beside an error is a lie a caller that
/// ignores the return code cannot detect.
#[hostcall]
pub(super) fn netctl_adhoc_get_in_addr(_ctx: &mut GuestCtx, _st: &mut VitaState, _inaddr: Ptr) -> i32 {
    SCE_NET_CTL_ERROR_NOT_CONNECTED
}

/// int sceNetCtlAdhocGetState(int *state)
/// The ad-hoc link state, reported DISCONNECTED for the same reason
/// [`netctl_inet_get_state`] is: there is no radio here.
#[hostcall]
pub(super) fn netctl_adhoc_get_state(ctx: &mut GuestCtx, _st: &mut VitaState, state: Ptr) -> i32 {
    if !state.is_null() {
        ctx.write_u32(state.addr(), SCE_NETCTL_STATE_DISCONNECTED);
    }
    0
}

/// int sceNetCtlAdhocDisconnect(void)
/// Leave the ad-hoc network. Reported as success rather than an error, because success is
/// what the caller is asking for: its postcondition - not connected - already holds. This
/// is a teardown path, and a title that gets an error tearing down a link it never had
/// tends to treat it as a failure to clean up rather than as "there was nothing to do".
#[hostcall]
pub(super) fn netctl_adhoc_disconnect(_ctx: &mut GuestCtx, _st: &mut VitaState) -> i32 {
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

/// int sceRtcGetCurrentNetworkTick(SceRtcTick *tick)
/// The network-synchronised clock, reported as the same tick [`rtc_get_current_tick`]
/// gives.
///
/// There is no time server off-console, but a console that has been through setup HAS a
/// synced clock, and its network tick then equals its RTC - so this is what the ordinary
/// case looks like. The alternative (reporting a failure) would push titles down an
/// error path they almost never exercise, to no benefit: the point of the call is to get a
/// trustworthy timestamp, and ours is already the one deterministic clock in the system.
#[hostcall]
pub(super) fn rtc_get_current_network_tick(ctx: &mut GuestCtx, st: &mut VitaState, tick: Ptr) -> i32 {
    if !tick.is_null() {
        let t = RTC_UNIX_EPOCH_TICKS + st.world.wall_us();
        ctx.write_u32(tick.addr(), t as u32);
        ctx.write_u32(tick.addr() + 4, (t >> 32) as u32);
    }
    0
}

/// int sceRtcConvertUtcToLocalTime(const SceRtcTick *utc, SceRtcTick *local_time)
/// int sceRtcConvertLocalTimeToUtc(const SceRtcTick *local_time, SceRtcTick *utc)
/// Shift a tick by the system's time-zone offset. The engine's clock has no time zone,
/// and inventing one would make two clocks that disagree - the wall clock a title stamps
/// a save with and the local time it displays - so the offset is zero and the tick is
/// copied. That is exactly what a console configured to UTC reports.
///
/// A NULL destination is an error, not a silent success: the caller reads the destination
/// straight after, and a call that "succeeded" without writing hands it stack garbage.
#[hostcall]
pub(super) fn rtc_convert_time_zone(ctx: &mut GuestCtx, _st: &mut VitaState, src: Ptr, dst: Ptr) -> i32 {
    if src.is_null() || dst.is_null() {
        SCE_RTC_ERROR_INVALID_POINTER
    } else {
        let lo = ctx.read_u32(src.addr());
        let hi = ctx.read_u32(src.addr() + 4);
        ctx.write_u32(dst.addr(), lo);
        ctx.write_u32(dst.addr() + 4, hi);
        0
    }
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

/// Decode and RANGE-CHECK a broken-down `SceRtcTime`/`SceDateTime` (they share a layout:
/// u16 year, month, day, hour, minute, second; u32 microsecond), returning UNIX seconds and
/// the microsecond field, or the kernel's error code for the first field out of range.
///
/// Shared by every conversion out of broken-down time so they cannot disagree about which
/// dates are valid - two copies of this validation would eventually accept a date in one
/// call and reject it in the other, and a title comparing the results would see time move
/// backwards.
fn rtc_decode_broken_down(ctx: &mut GuestCtx, time: Ptr) -> Result<(i64, i64), i32> {
    let raw = ctx.read_bytes(time.addr(), 16);
    let u16_at = |off: usize| u16::from_le_bytes([raw[off], raw[off + 1]]) as i64;
    let (year, month, day) = (u16_at(0), u16_at(2), u16_at(4));
    let (hour, minute, second) = (u16_at(6), u16_at(8), u16_at(10));
    let micro = u32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]) as i64;
    if year < 1 || year > 9999 {
        return Err(SCE_RTC_ERROR_INVALID_YEAR);
    }
    if month < 1 || month > 12 {
        return Err(SCE_RTC_ERROR_INVALID_MONTH);
    }
    if day < 1 || day > days_in_month(year, month) {
        return Err(SCE_RTC_ERROR_INVALID_DAY);
    }
    if hour > 23 {
        return Err(SCE_RTC_ERROR_INVALID_HOUR);
    }
    if minute > 59 {
        return Err(SCE_RTC_ERROR_INVALID_MINUTE);
    }
    if second > 59 {
        return Err(SCE_RTC_ERROR_INVALID_SECOND);
    }
    if micro > 999_999 {
        return Err(SCE_RTC_ERROR_INVALID_MICROSECOND);
    }
    // `days_from_civil` counts days from 1970-01-01, so this is UNIX seconds already.
    Ok((days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second, micro))
}

/// Civil date `(year, month, day)` from a day count since 1970-01-01 - the exact inverse
/// of [`days_from_civil`], by the same proleptic-Gregorian construction, so a tick
/// converted out and back is unchanged across the whole SceRtcTime year range.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11], March-based
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// int sceRtcSetTick(SceRtcTime *pTime, const SceRtcTick *pTick)
/// Expand a 64-bit microsecond tick back into broken-down time - the inverse of
/// [`rtc_get_tick`]. Titles round-trip through this pair to do date arithmetic (add a day
/// to a savedata stamp, then display it), so the two must agree exactly; they share one
/// calendar construction for that reason.
#[hostcall]
pub(super) fn rtc_set_tick(ctx: &mut GuestCtx, _st: &mut VitaState, time: Ptr, tick: Ptr) -> i32 {
    rtc_set_tick_impl(ctx, time, tick)
}

fn rtc_set_tick_impl(ctx: &mut GuestCtx, time: Ptr, tick: Ptr) -> i32 {
    if time.is_null() || tick.is_null() {
        return SCE_RTC_ERROR_INVALID_POINTER;
    }
    let lo = ctx.read_u32(tick.addr()) as u64;
    let hi = ctx.read_u32(tick.addr() + 4) as u64;
    let t = ((hi << 32) | lo) as i64;
    // To microseconds since the Unix epoch, then to whole seconds plus a remainder.
    // FLOORED division on both, so a tick before 1970 (which the RTC epoch allows)
    // expands to the right second rather than being truncated toward zero.
    let us = t - RTC_UNIX_EPOCH_TICKS as i64;
    let secs = us.div_euclid(1_000_000);
    let micro = us.rem_euclid(1_000_000);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    if y < 1 || y > 9999 {
        return SCE_RTC_ERROR_INVALID_YEAR;
    }
    let fields: [u16; 6] =
        [y as u16, m as u16, d as u16, (sod / 3600) as u16, (sod / 60 % 60) as u16, (sod % 60) as u16];
    let mut bytes = [0u8; 16];
    for (i, f) in fields.iter().enumerate() {
        bytes[i * 2..i * 2 + 2].copy_from_slice(&f.to_le_bytes());
    }
    bytes[12..16].copy_from_slice(&(micro as u32).to_le_bytes());
    ctx.write_bytes(time.addr(), &bytes);
    0
}

/// Read a 64-bit tick out of a `SceRtcTick*`.
fn read_tick(ctx: &mut GuestCtx, p: u32) -> i64 {
    let lo = ctx.read_u32(p) as u64;
    let hi = ctx.read_u32(p + 4) as u64;
    ((hi << 32) | lo) as i64
}

fn write_tick(ctx: &mut GuestCtx, p: u32, t: i64) {
    ctx.write_u32(p, t as u32);
    ctx.write_u32(p + 4, (t >> 32) as u32);
}

/// The `sceRtcTickAdd*` family that adds a FIXED-length unit: `int sceRtcTickAddX(
/// SceRtcTick *pTick0, const SceRtcTick *pTick1, <count>)`, where the count is scaled by
/// `us_per_unit` microseconds.
///
/// `wide` selects how the count is read, and it is not cosmetic: the SDK declares the
/// count as `SceLong64` for ticks/microseconds/seconds/minutes and as `int` for
/// hours/days/weeks, so under AAPCS the 64-bit forms occupy an even-aligned register PAIR
/// (r2:r3 after two pointer arguments) while the 32-bit forms are a single register. Read
/// the wrong one and the amount added is unrelated to what the caller asked for.
///
/// Written against the registers directly rather than through `#[hostcall]` for exactly
/// that reason: the 64-bit pair is the whole subtlety here and it should be visible.
pub(super) fn rtc_tick_add_fixed(ctx: &mut GuestCtx, us_per_unit: i64, wide: bool) {
    let (out, src) = (ctx.arg(0), ctx.arg(1));
    if out == 0 || src == 0 {
        ctx.ret(SCE_RTC_ERROR_INVALID_POINTER as u32);
        return;
    }
    let n = if wide {
        (((ctx.arg(3) as u64) << 32) | ctx.arg(2) as u64) as i64
    } else {
        ctx.arg(2) as i32 as i64
    };
    let base = read_tick(ctx, src);
    // Saturating: a title probing the representable range must not get a wrapped tick,
    // which would read as a date thousands of years off rather than as a clamp.
    let t = base.saturating_add(n.saturating_mul(us_per_unit));
    write_tick(ctx, out, t);
    ctx.ret(0);
}

/// The CALENDAR-length members of the same family: `sceRtcTickAddMonths` and
/// `sceRtcTickAddYears`. A month is not a fixed number of microseconds, so these convert
/// to broken-down time, add, clamp the day into the target month (31 January plus one
/// month is 28 or 29 February, not 31 February) and convert back.
pub(super) fn rtc_tick_add_calendar(ctx: &mut GuestCtx, months_per_unit: i64) {
    let (out, src) = (ctx.arg(0), ctx.arg(1));
    if out == 0 || src == 0 {
        ctx.ret(SCE_RTC_ERROR_INVALID_POINTER as u32);
        return;
    }
    let n = ctx.arg(2) as i32 as i64;
    let base = read_tick(ctx, src);
    let us = base - RTC_UNIX_EPOCH_TICKS as i64;
    let secs = us.div_euclid(1_000_000);
    let micro = us.rem_euclid(1_000_000);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    // Months as an absolute count so a negative delta borrows across the year boundary.
    let total = (y * 12 + (m - 1)) + n * months_per_unit;
    let (ny, nm) = (total.div_euclid(12), total.rem_euclid(12) + 1);
    if ny < 1 || ny > 9999 {
        ctx.ret(SCE_RTC_ERROR_INVALID_YEAR as u32);
        return;
    }
    let nd = d.min(days_in_month(ny, nm));
    let t = RTC_UNIX_EPOCH_TICKS as i64
        + (days_from_civil(ny, nm, nd) * 86_400 + sod) * 1_000_000
        + micro;
    write_tick(ctx, out, t);
    ctx.ret(0);
}

/// int sceRtcGetTime64_t(const SceDateTime *time, SceUInt64 *pullTime)
/// Convert broken-down time to a 64-bit UNIX timestamp (seconds since 1970-01-01), the
/// POSIX-compatible counterpart of [`rtc_get_tick`]'s RTC-epoch microsecond tick. Titles
/// use it to stamp and compare savedata; a wrong or unwritten value corrupts every
/// ordering derived from it, so the same exact range checks apply and an invalid date is
/// rejected rather than folded into a plausible number.
#[hostcall]
pub(super) fn rtc_get_time64_t(ctx: &mut GuestCtx, _st: &mut VitaState, time: Ptr, out: Ptr) -> i32 {
    rtc_get_time64_t_impl(ctx, time, out)
}

fn rtc_get_time64_t_impl(ctx: &mut GuestCtx, time: Ptr, out: Ptr) -> i32 {
    if time.is_null() || out.is_null() {
        return SCE_RTC_ERROR_INVALID_POINTER;
    }
    match rtc_decode_broken_down(ctx, time) {
        Err(e) => e,
        Ok((secs, _micro)) => {
            ctx.write_u32(out.addr(), secs as u32);
            ctx.write_u32(out.addr() + 4, (secs >> 32) as u32);
            0
        }
    }
}

fn rtc_get_tick_impl(ctx: &mut GuestCtx, time: Ptr, tick: Ptr) -> i32 {
    if time.is_null() || tick.is_null() {
        return SCE_RTC_ERROR_INVALID_POINTER;
    }
    match rtc_decode_broken_down(ctx, time) {
        Err(e) => e,
        Ok((secs, micro)) => {
            let t = RTC_UNIX_EPOCH_TICKS as i64 + secs * 1_000_000 + micro;
            ctx.write_u32(tick.addr(), t as u32);
            ctx.write_u32(tick.addr() + 4, (t >> 32) as u32);
            0
        }
    }
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

/// int sceAppUtilSaveDataSlotSetParam(unsigned int slotId,
///     SceAppUtilSaveDataSlotParam *param, SceAppUtilSaveDataMountPoint *mountPoint)
/// Update the metadata of a slot that already exists - the title's "save my progress" path
/// once the slot has been created (a first-run setup writing back the profile it just built).
/// Distinct from `Create` in exactly one way that matters to the caller: a slot that was never
/// created reports SAVEDATA_SLOT_NOT_FOUND rather than being conjured into existence, so a
/// title that sets before creating learns it, as it would on hardware. On success the param
/// blob replaces the stored one verbatim, so a later `GetParam` reads back what was last set.
#[hostcall]
pub(super) fn apputil_savedata_slot_set_param(ctx: &mut GuestCtx, st: &mut VitaState, slot_id: u32, param: Ptr, mount: Ptr) -> i32 {
    let mount_name = read_mount_name(ctx, mount);
    if st.savedata_slot_exists(&mount_name, slot_id) {
        let bytes = if param.is_null() {
            vec![0u8; SAVEDATA_SLOT_PARAM_SIZE]
        } else {
            ctx.read_bytes(param.addr(), SAVEDATA_SLOT_PARAM_SIZE)
        };
        st.savedata_slot_put(&mount_name, slot_id, bytes);
        tracing::debug!(target: "vitaslop::cb", slot = slot_id, "sceAppUtilSaveDataSlotSetParam -> OK");
        0
    } else {
        tracing::debug!(target: "vitaslop::cb", slot = slot_id, "sceAppUtilSaveDataSlotSetParam -> SLOT_NOT_FOUND");
        SCE_APPUTIL_ERROR_SAVEDATA_SLOT_NOT_FOUND
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

/// int sceFiosOverlayGetRecommendedScheduler02(int avail, const char *path, SceUInt64 *out)
///
/// Asks which I/O scheduler suits a path. Advisory by nature: there is one synchronous
/// file backend here and no overlays are mounted, so there is nothing to recommend and the
/// out-param is set to a DEFINED zero rather than left as stack garbage (which a caller
/// would read as a real scheduler id).
///
/// Note what is deliberately NOT implemented alongside it: the overlay MOUNT calls
/// (`sceFiosOverlayAddForProcess02` and friends). Accepting a mount and then ignoring it
/// would silently resolve paths against the wrong layer - a title overlaying its patch
/// directory over its app directory would read pre-patch files with no sign anything was
/// wrong. Those stay unimplemented so they hard-fail if a title actually uses them.
#[hostcall]
pub(super) fn fios_overlay_get_recommended_scheduler(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    _avail: u32,
    _path: Ptr,
    out: Ptr,
) -> i32 {
    if !out.is_null() {
        ctx.write_u32(out.addr(), 0);
        ctx.write_u32(out.addr() + 4, 0);
    }
    0
}

/// int sceAppMgrIsGameProgram(void)
/// Whether the running program is a GAME rather than a system application. We boot a
/// retail game dump, so this is a fact about the emulator and the answer is yes. Titles
/// gate game-only behaviour on it (savedata paths, trophy setup, the pause menu), so
/// answering 0 would quietly put a game into system-app mode.
#[hostcall]
pub(super) fn app_mgr_is_game_program(_st: &mut VitaState) -> i32 {
    1
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

// ---------------------------------------------------------------------------
// SceNpTrophy
//
// A title's trophy set is not online state: every count, grade, name, description and
// icon a game can ask for is shipped inside the title itself, at
// `sce_sys/trophy/<NPCOMMID>/TROPHY.TRP` (see [`crate::trophy`]). So these handlers report
// the title's OWN data, not an invented set. The only part a console adds is the
// per-account unlock ledger, which for a fresh offline profile is empty and grows during
// the run as the title unlocks trophies - so unlock counts start at zero and a title's
// unlock-then-read-back path is faithful.
//
// An earlier version of this file reported a defined but EMPTY set on the reasoning that
// trophies have no backing service off-console. That was wrong twice over: the data was
// there all along, and a zero-trophy set is not something every title accepts - one read
// it as a failed cache load and respawned its trophy thread about 74,000 times until the
// guest heap was exhausted.
// ---------------------------------------------------------------------------

/// `SCE_NP_TROPHY_ERROR_*` (psdevwiki error-code table).
const SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT: i32 = 0x8055_1604u32 as i32;
const SCE_NP_TROPHY_ERROR_INSUFFICIENT_BUFFER: i32 = 0x8055_1605u32 as i32;
const SCE_NP_TROPHY_ERROR_INVALID_CONTEXT: i32 = 0x8055_1609u32 as i32;
const SCE_NP_TROPHY_ERROR_INVALID_NPCOMMID: i32 = 0x8055_160Au32 as i32;
const SCE_NP_TROPHY_ERROR_INVALID_GROUP_ID: i32 = 0x8055_160Du32 as i32;
const SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID: i32 = 0x8055_160Eu32 as i32;
const SCE_NP_TROPHY_ERROR_TROPHY_ALREADY_UNLOCKED: i32 = 0x8055_160Fu32 as i32;
const SCE_NP_TROPHY_ERROR_PLATINUM_CANNOT_UNLOCK: i32 = 0x8055_1610u32 as i32;
const SCE_NP_TROPHY_ERROR_BROKEN_DATA: i32 = 0x8055_1614u32 as i32;
const SCE_NP_TROPHY_ERROR_ICON_FILE_NOT_FOUND: i32 = 0x8055_1618u32 as i32;
const SCE_NP_TROPHY_ERROR_TRP_FILE_NOT_FOUND: i32 = 0x8055_1619u32 as i32;
const SCE_NP_TROPHY_ERROR_INVALID_TRP_FILE_FORMAT: i32 = 0x8055_161Au32 as i32;
const SCE_NP_TROPHY_ERROR_UNSUPPORTED_TRP_FILE: i32 = 0x8055_161Bu32 as i32;
const SCE_NP_TROPHY_ERROR_INVALID_TROPHY_CONF_FORMAT: i32 = 0x8055_161Cu32 as i32;

/// `SCE_NP_TROPHY_INVALID_TROPHY_ID`: the "no such trophy" id an unlock reports through
/// `platinumId` when the unlock did not complete the set.
const SCE_NP_TROPHY_INVALID_TROPHY_ID: i32 = -1;

/// Struct sizes from the MIT `psp2/np/trophy.h` layouts, each also confirmed against the
/// `.size` version tag the caller writes before the call:
/// `SceNpTrophyGameDetails` {u32 size, numGroups, numTrophies, numPlatinum, numGold,
/// numSilver, numBronze; char title[128]; char description[1024]}.
const NP_TROPHY_GAME_DETAILS_SIZE: usize = 1180;
/// `SceNpTrophyGameData` {u32 size, unlockedTrophies, unlockedPlatinum, unlockedGold,
/// unlockedSilver, unlockedBronze, progressPercentage}.
const NP_TROPHY_GAME_DATA_SIZE: usize = 28;
/// `SceNpTrophyGroupDetails` {u32 size; i32 groupId; u32 numTrophies, numPlatinum,
/// numGold, numSilver, numBronze; char title[128]; char description[1024]}.
const NP_TROPHY_GROUP_DETAILS_SIZE: usize = 1180;
/// `SceNpTrophyGroupData` {u32 size; i32 groupId; u32 unlockedTrophies, unlockedPlatinum,
/// unlockedGold, unlockedSilver, unlockedBronze, progressPercentage}.
const NP_TROPHY_GROUP_DATA_SIZE: usize = 32;
/// `SceNpTrophyDetails` {u32 size; i32 trophyId, trophyGrade, groupId, hidden;
/// char name[128]; char description[1024]}.
const NP_TROPHY_DETAILS_SIZE: usize = 1172;
/// `SceNpTrophyData` {u32 size; i32 trophyId, unlocked, unk0; SceRtcTick timestamp}. The
/// 8-byte tick is 8-aligned, which the preceding `unk0` pads out to.
const NP_TROPHY_DATA_SIZE: usize = 24;
/// `SceNpTrophyFlagArray` {u32 flag_bits[SCE_NP_TROPHY_FLAG_SETSIZE >> 5]} - a fixed
/// 128-bit unlock bitmap, one bit per trophy id.
const NP_TROPHY_FLAG_ARRAY_SIZE: usize = 16;
/// `SCE_NP_TROPHY_NAME_MAX_SIZE` / `SCE_NP_TROPHY_GAME_TITLE_MAX_SIZE`.
const NP_TROPHY_TITLE_MAX: usize = 128;
/// `SCE_NP_TROPHY_DESCR_MAX_SIZE` / `SCE_NP_TROPHY_GAME_DESCR_MAX_SIZE`.
const NP_TROPHY_DESCR_MAX: usize = 1024;

/// Build one fixed-layout OUT struct. Fields are appended in declaration order and the
/// total is checked against the declared size at [`Self::finish`], so a layout that does
/// not add up is a build-time-visible failure rather than a struct the guest reads past.
struct OutStruct {
    buf: Vec<u8>,
}

impl OutStruct {
    /// Start a struct whose leading `SceSize size` field is the declared size.
    fn new(size: usize) -> OutStruct {
        let mut s = OutStruct { buf: Vec::with_capacity(size) };
        s.u32(size as u32);
        s
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    /// A fixed-width NUL-padded char array. An over-long string is truncated on a
    /// character boundary with room kept for the terminator, so the guest always reads a
    /// valid C string.
    fn text(&mut self, s: &str, width: usize) {
        // Keep whole characters only, and always leave the last byte for the terminator.
        let mut cut = 0;
        for (i, c) in s.char_indices() {
            let next = i + c.len_utf8();
            if next > width - 1 {
                break;
            }
            cut = next;
        }
        let mut field = vec![0u8; width];
        field[..cut].copy_from_slice(&s.as_bytes()[..cut]);
        self.buf.extend_from_slice(&field);
    }
    fn finish(self, size: usize) -> Vec<u8> {
        debug_assert_eq!(self.buf.len(), size, "trophy OUT struct layout does not add up");
        self.buf
    }
}

/// Check the caller-written `.size` version tag at offset 0 and write the struct when it
/// matches. These APIs take `.size` as an input version tag; a size the system does not
/// implement is `SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT` on hardware, and refusing here also
/// means we never write a layout the caller did not allocate for.
///
/// A null pointer is not an error: a caller can request only one of the two OUT structs.
fn write_out_struct(ctx: &mut GuestCtx, p: Ptr, expect: usize, build: impl FnOnce() -> Vec<u8>) -> i32 {
    if p.is_null() {
        return 0;
    }
    let declared = ctx.read_u32(p.addr()) as usize;
    if declared != expect {
        tracing::warn!(
            target: "vitaslop::err",
            declared,
            expect,
            "SceNpTrophy OUT struct size tag is not one this engine implements"
        );
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
    }
    ctx.write_bytes(p.addr(), &build());
    0
}

/// Serve an icon through the API's shared `(void *buffer, SceSize *size)` protocol:
/// `*size` is the buffer's capacity going in and the icon's real size coming out, and a
/// null `buffer` is a size query. A too-small buffer still learns the size it needs.
fn write_icon(ctx: &mut GuestCtx, buffer: Ptr, size: Ptr, icon: Option<&[u8]>) -> i32 {
    let Some(icon) = icon else {
        return SCE_NP_TROPHY_ERROR_ICON_FILE_NOT_FOUND;
    };
    if size.is_null() {
        return SCE_NP_TROPHY_ERROR_INVALID_ARGUMENT;
    }
    let capacity = ctx.read_u32(size.addr()) as usize;
    ctx.write_u32(size.addr(), icon.len() as u32);
    if buffer.is_null() {
        return 0;
    }
    if capacity < icon.len() {
        return SCE_NP_TROPHY_ERROR_INSUFFICIENT_BUFFER;
    }
    ctx.write_bytes(buffer.addr(), icon);
    0
}

/// The percentage of a set a player has completed, as the trophy APIs report it in
/// `progressPercentage`: unlocked over total, with the platinum excluded from both. The
/// platinum is awarded FOR completing the set, so counting it would make 100% unreachable
/// until it lands and then jump past every other trophy's contribution.
fn progress_percentage(unlocked: &crate::trophy::GradeCounts, total: &crate::trophy::GradeCounts) -> u32 {
    let denom = total.total - total.platinum;
    if denom == 0 {
        return 0;
    }
    ((unlocked.total - unlocked.platinum) * 100) / denom
}

/// The directory a trophy set lives in, from a `SceNpCommunicationId`
/// {char data[9]; char term; uint8_t num; char dummy}: the 9-character id, an
/// underscore, and the two-digit sub-id, e.g. `NPWR12345_00`. `None` if the id is not
/// the printable ASCII the format requires.
fn read_comm_id(ctx: &GuestCtx, p: Ptr) -> Option<String> {
    if p.is_null() {
        return None;
    }
    let raw = ctx.read_bytes(p.addr(), 12);
    let end = raw[..9].iter().position(|&b| b == 0).unwrap_or(9);
    if end == 0 || !raw[..end].iter().all(|b| b.is_ascii_alphanumeric() || *b == b'_') {
        return None;
    }
    let id = std::str::from_utf8(&raw[..end]).ok()?;
    Some(format!("{id}_{:02}", raw[10]))
}

/// The NP communication id of the trophy set the RUNNING TITLE ships, recovered from the
/// one directory under `app0:/sce_sys/trophy/` (its name IS the id, e.g. `NPWR12345_00`).
/// This is what a console answers a NULL `commId` with. `None` when the title ships no
/// trophy set, or more than one and so nothing to pick unambiguously - both of which are
/// reported so a mis-resolution never looks like a missing file later.
fn title_comm_id(st: &mut VitaState) -> Option<String> {
    let dirs: Vec<String> = st
        .list_dir("app0:/sce_sys/trophy")
        .into_iter()
        .filter(|e| e.is_dir)
        .map(|e| e.name)
        .collect();
    match dirs.len() {
        1 => Some(dirs.into_iter().next().unwrap()),
        n => {
            tracing::warn!(
                target: "vitaslop::err",
                candidates = n,
                "sceNpTrophyCreateContext(commId=NULL): cannot resolve the title's own trophy set"
            );
            None
        }
    }
}

/// Read and parse the trophy set a context is being created for, out of the title's own
/// data. Returns the SCE error a real system reports for the same broken file when the
/// set cannot be read - the errors are distinct on purpose, so "the title ships no trophy
/// set" and "this engine cannot read the one it ships" never look alike in a log.
fn load_trophy_set(st: &mut VitaState, comm_id: &str) -> Result<(), i32> {
    if st.trophies.has_set(comm_id) {
        return Ok(());
    }
    let path = format!("app0:/sce_sys/trophy/{comm_id}/TROPHY.TRP");
    let Some(bytes) = st.read_file(&path) else {
        tracing::warn!(target: "vitaslop::err", %path, "no trophy set shipped at this path");
        return Err(SCE_NP_TROPHY_ERROR_TRP_FILE_NOT_FOUND);
    };
    match crate::trophy::TrophySet::parse(bytes, SCE_SYSTEM_PARAM_LANG_ENGLISH_US) {
        Ok(set) => {
            let counts = set.counts(|_| true);
            tracing::info!(
                target: "vitaslop::cb",
                comm_id,
                trophies = counts.total,
                platinum = counts.platinum,
                gold = counts.gold,
                silver = counts.silver,
                bronze = counts.bronze,
                groups = set.groups.len(),
                "read the title's own trophy set"
            );
            st.trophies.insert_set(set);
            Ok(())
        }
        Err(e) => {
            tracing::error!(target: "vitaslop::err", %path, error = %e, "cannot read the title's trophy set");
            Err(match e {
                crate::trophy::TrophyError::NotTrp => SCE_NP_TROPHY_ERROR_INVALID_TRP_FILE_FORMAT,
                crate::trophy::TrophyError::UnsupportedVersion(_) => SCE_NP_TROPHY_ERROR_UNSUPPORTED_TRP_FILE,
                crate::trophy::TrophyError::Truncated(_) => SCE_NP_TROPHY_ERROR_BROKEN_DATA,
                crate::trophy::TrophyError::MissingConf | crate::trophy::TrophyError::BadConf(_) => {
                    SCE_NP_TROPHY_ERROR_INVALID_TROPHY_CONF_FORMAT
                }
            })
        }
    }
}

/// int sceNpTrophyCreateContext(SceNpTrophyContext *context, const SceNpCommunicationId
///     *commId, const SceNpCommunicationSignature *commSig, SceUInt64 options)
/// Open a context on the trophy set the title ships for `commId`, reading and parsing it
/// on first use. A context is what every later query is scoped to, so this is where the
/// set is bound; a title whose TRP is missing or unreadable learns so here, at the call
/// that a real system would also fail, rather than through empty counts later.
///
/// `commSig` is the set's signature, which a console verifies against Sony's key. There is
/// no key here and nothing to verify against, so the set is read on its own terms - the
/// same position the engine takes on every other signature it cannot check.
// `options` (SceUInt64) is the trailing argument and is not needed, so it is left
// unread rather than declared (the macro has no u64 value-arg class).
// Each of these delegates to a plain function: a `#[hostcall]` body is spliced into a
// wrapper and cannot use an early `return`, and the SceNpTrophy handlers are all
// guard-clause shaped.
#[hostcall]
pub(super) fn np_trophy_create_context(ctx: &mut GuestCtx, st: &mut VitaState, context: Ptr, comm_id: Ptr, _comm_sig: Ptr) -> i32 {
    create_context(ctx, st, context, comm_id)
}

fn create_context(ctx: &mut GuestCtx, st: &mut VitaState, context: Ptr, comm_id: Ptr) -> i32 {
    // A NULL `commId` means "the running title's own trophy set" - the console already
    // knows the id from the installed package, so a title that only ever touches its own
    // trophies does not repeat it. Rejecting NULL is not a harmless strictness: a title
    // whose trophy worker fails here TERMINATES that worker at boot, and every later
    // trophy request then never completes, which reads as a hang on a screen thousands of
    // frames away with no trophy call anywhere near it.
    let comm_id = match read_comm_id(ctx, comm_id) {
        Some(id) => id,
        None if comm_id.is_null() => match title_comm_id(st) {
            Some(id) => id,
            None => return SCE_NP_TROPHY_ERROR_INVALID_NPCOMMID,
        },
        None => return SCE_NP_TROPHY_ERROR_INVALID_NPCOMMID,
    };
    if let Err(e) = load_trophy_set(st, &comm_id) {
        return e;
    }
    let id = st.new_handle();
    st.trophies.open_context(id, &comm_id);
    if !context.is_null() {
        ctx.write_u32(context.addr(), id);
    }
    0
}

/// int sceNpTrophyDestroyContext(SceNpTrophyContext context)
/// Close a context. The set and this run's unlock ledger outlive it, exactly as they do on
/// hardware: a title that destroys and recreates a context sees the same unlocks.
#[hostcall]
pub(super) fn np_trophy_destroy_context(_ctx: &mut GuestCtx, st: &mut VitaState, context: u32) -> i32 {
    if st.trophies.close_context(context) { 0 } else { SCE_NP_TROPHY_ERROR_INVALID_CONTEXT }
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

/// int sceNpTrophyGetGameInfo(SceNpTrophyContext context, SceNpTrophyHandle handle,
///     SceNpTrophyGameDetails *details, SceNpTrophyGameData *data)
/// Report the title's own trophy set: the group and grade counts, and the localized title
/// and description, exactly as its `TROPCONF.SFM`/`TROP.SFM` declare them. `data` reports
/// this run's unlock ledger, which starts empty on a fresh offline profile.
///
/// Either OUT pointer may be null (a caller can request only one of the two structs), so
/// each is filled independently.
#[hostcall]
pub(super) fn np_trophy_get_game_info(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, _handle: u32, details: Ptr, data: Ptr) -> i32 {
    game_info(ctx, st, context, details, data)
}

fn game_info(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, details: Ptr, data: Ptr) -> i32 {
    let Some(set) = st.trophies.set_for(context) else {
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    };
    let total = set.counts(|_| true);
    let rc = write_out_struct(ctx, details, NP_TROPHY_GAME_DETAILS_SIZE, || {
        let mut s = OutStruct::new(NP_TROPHY_GAME_DETAILS_SIZE);
        s.u32(set.groups.len() as u32);
        s.u32(total.total);
        s.u32(total.platinum);
        s.u32(total.gold);
        s.u32(total.silver);
        s.u32(total.bronze);
        s.text(&set.title, NP_TROPHY_TITLE_MAX);
        s.text(&set.detail, NP_TROPHY_DESCR_MAX);
        s.finish(NP_TROPHY_GAME_DETAILS_SIZE)
    });
    if rc != 0 {
        return rc;
    }
    let unlocked = st.trophies.unlocked_counts(set, |_| true);
    write_out_struct(ctx, data, NP_TROPHY_GAME_DATA_SIZE, || {
        let mut s = OutStruct::new(NP_TROPHY_GAME_DATA_SIZE);
        s.u32(unlocked.total);
        s.u32(unlocked.platinum);
        s.u32(unlocked.gold);
        s.u32(unlocked.silver);
        s.u32(unlocked.bronze);
        s.u32(progress_percentage(&unlocked, &total));
        s.finish(NP_TROPHY_GAME_DATA_SIZE)
    })
}

/// int sceNpTrophyGetGroupInfo(SceNpTrophyContext context, SceNpTrophyHandle handle,
///     SceInt32 groupId, SceNpTrophyGroupDetails *details, SceNpTrophyGroupData *data)
/// The same report, narrowed to one group. A group id the set does not declare is
/// `INVALID_GROUP_ID` rather than an empty group, so a title enumerating groups learns
/// where the list ends.
#[hostcall]
pub(super) fn np_trophy_get_group_info(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, _handle: u32, group_id: i32, details: Ptr, data: Ptr) -> i32 {
    group_info(ctx, st, context, group_id, details, data)
}

fn group_info(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, group_id: i32, details: Ptr, data: Ptr) -> i32 {
    let Some(set) = st.trophies.set_for(context) else {
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    };
    let Some(group) = set.groups.iter().find(|g| g.id == group_id) else {
        return SCE_NP_TROPHY_ERROR_INVALID_GROUP_ID;
    };
    let total = set.counts(|t| t.group_id == group_id);
    let rc = write_out_struct(ctx, details, NP_TROPHY_GROUP_DETAILS_SIZE, || {
        let mut s = OutStruct::new(NP_TROPHY_GROUP_DETAILS_SIZE);
        s.i32(group.id);
        s.u32(total.total);
        s.u32(total.platinum);
        s.u32(total.gold);
        s.u32(total.silver);
        s.u32(total.bronze);
        s.text(&group.name, NP_TROPHY_TITLE_MAX);
        s.text(&group.detail, NP_TROPHY_DESCR_MAX);
        s.finish(NP_TROPHY_GROUP_DETAILS_SIZE)
    });
    if rc != 0 {
        return rc;
    }
    let unlocked = st.trophies.unlocked_counts(set, |t| t.group_id == group_id);
    write_out_struct(ctx, data, NP_TROPHY_GROUP_DATA_SIZE, || {
        let mut s = OutStruct::new(NP_TROPHY_GROUP_DATA_SIZE);
        s.i32(group_id);
        s.u32(unlocked.total);
        s.u32(unlocked.platinum);
        s.u32(unlocked.gold);
        s.u32(unlocked.silver);
        s.u32(unlocked.bronze);
        s.u32(progress_percentage(&unlocked, &total));
        s.finish(NP_TROPHY_GROUP_DATA_SIZE)
    })
}

/// int sceNpTrophyGetTrophyInfo(SceNpTrophyContext context, SceNpTrophyHandle handle,
///     SceNpTrophyId trophyId, SceNpTrophyDetails *details, SceNpTrophyData *data)
/// One trophy's declared grade, group, hidden flag, name and description, plus whether
/// this run has unlocked it.
///
/// The name and description of a HIDDEN trophy are reported as the set declares them. The
/// hiding is the caller's job - the `hidden` flag is exactly how a title is told to draw a
/// locked secret trophy as `???` - and a system that blanked the text would break the
/// trophy list the moment one was earned.
///
/// `timestamp` is the `SceRtcTick` the trophy was earned at. Nothing here has been earned
/// at a real wall-clock time, so an unlocked trophy reports the same fixed epoch the rest
/// of the engine's clock reports, and a locked one reports 0.
#[hostcall]
pub(super) fn np_trophy_get_trophy_info(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, _handle: u32, trophy_id: u32, details: Ptr, data: Ptr) -> i32 {
    trophy_info(ctx, st, context, trophy_id, details, data)
}

fn trophy_info(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, trophy_id: u32, details: Ptr, data: Ptr) -> i32 {
    let Some(set) = st.trophies.set_for(context) else {
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    };
    let Some(trophy) = set.trophy(trophy_id) else {
        return SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID;
    };
    let rc = write_out_struct(ctx, details, NP_TROPHY_DETAILS_SIZE, || {
        let mut s = OutStruct::new(NP_TROPHY_DETAILS_SIZE);
        s.u32(trophy.id);
        s.i32(trophy.grade as i32);
        s.i32(trophy.group_id);
        s.i32(trophy.hidden as i32);
        s.text(&trophy.name, NP_TROPHY_TITLE_MAX);
        s.text(&trophy.detail, NP_TROPHY_DESCR_MAX);
        s.finish(NP_TROPHY_DETAILS_SIZE)
    });
    if rc != 0 {
        return rc;
    }
    let earned_at = st.trophies.unlocked_at(&set.comm_id, trophy_id);
    write_out_struct(ctx, data, NP_TROPHY_DATA_SIZE, || {
        let mut s = OutStruct::new(NP_TROPHY_DATA_SIZE);
        s.u32(trophy_id);
        s.i32(earned_at.is_some() as i32);
        s.i32(0); // padding ahead of the 8-aligned tick
        s.u64(earned_at.unwrap_or(0));
        s.finish(NP_TROPHY_DATA_SIZE)
    })
}

/// int sceNpTrophyGetGameIcon(SceNpTrophyContext context, SceNpTrophyHandle handle,
///     void *buffer, SceSize *size)
/// The set's own icon (`ICON0.PNG` inside the title's TRP).
///
/// An earlier version returned SUCCESS with `*size = 0` on the reasoning that an empty set
/// has no icon. That is a lie a caller cannot act on: one title's trophy thread read the
/// zero as a transient read failure, tore itself down and respawned ~74,000 times until
/// the guest heap was exhausted, and the crash surfaced as an out-of-bounds access with a
/// nonsense stack pointer nowhere near the cause. If an icon is genuinely absent the
/// answer is a definite `ICON_FILE_NOT_FOUND`, never a hollow success.
#[hostcall]
pub(super) fn np_trophy_get_game_icon(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, _handle: u32, buffer: Ptr, size: Ptr) -> i32 {
    match st.trophies.set_for(context) {
        Some(set) => write_icon(ctx, buffer, size, set.game_icon()),
        None => SCE_NP_TROPHY_ERROR_INVALID_CONTEXT,
    }
}

/// int sceNpTrophyGetGroupIcon(SceNpTrophyContext context, SceNpTrophyHandle handle,
///     SceInt32 groupId, void *buffer, SceSize *size)
/// One group's icon (`GR<nnn>.PNG`).
#[hostcall]
pub(super) fn np_trophy_get_group_icon(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, _handle: u32, group_id: i32, buffer: Ptr, size: Ptr) -> i32 {
    group_icon(ctx, st, context, group_id, buffer, size)
}

fn group_icon(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, group_id: i32, buffer: Ptr, size: Ptr) -> i32 {
    let Some(set) = st.trophies.set_for(context) else {
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    };
    if !set.groups.iter().any(|g| g.id == group_id) {
        return SCE_NP_TROPHY_ERROR_INVALID_GROUP_ID;
    }
    write_icon(ctx, buffer, size, set.group_icon(group_id))
}

/// int sceNpTrophyGetTrophyIcon(SceNpTrophyContext context, SceNpTrophyHandle handle,
///     SceNpTrophyId trophyId, void *buffer, SceSize *size)
/// One trophy's icon (`TROP<nnn>.PNG`).
#[hostcall]
pub(super) fn np_trophy_get_trophy_icon(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, _handle: u32, trophy_id: u32, buffer: Ptr, size: Ptr) -> i32 {
    trophy_icon(ctx, st, context, trophy_id, buffer, size)
}

fn trophy_icon(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, trophy_id: u32, buffer: Ptr, size: Ptr) -> i32 {
    let Some(set) = st.trophies.set_for(context) else {
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    };
    if set.trophy(trophy_id).is_none() {
        return SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID;
    }
    write_icon(ctx, buffer, size, set.trophy_icon(trophy_id))
}

/// int sceNpTrophyGetTrophyUnlockState(SceNpTrophyContext context, SceNpTrophyHandle
///     handle, SceNpTrophyFlagArray *flags, SceUInt32 *count)
/// The unlock bitmap (one bit per trophy id, LSB-first within each word) and the number of
/// trophies in the SET - `count` is the set's size, not the number unlocked, which is what
/// lets a caller know how many bits of the fixed 128-bit array are meaningful.
#[hostcall]
pub(super) fn np_trophy_get_trophy_unlock_state(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, _handle: u32, flags: Ptr, count: Ptr) -> i32 {
    unlock_state(ctx, st, context, flags, count)
}

fn unlock_state(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, flags: Ptr, count: Ptr) -> i32 {
    let Some(set) = st.trophies.set_for(context) else {
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    };
    if !flags.is_null() {
        let mut bits = [0u8; NP_TROPHY_FLAG_ARRAY_SIZE];
        for t in &set.trophies {
            let bit = t.id as usize;
            if bit < NP_TROPHY_FLAG_ARRAY_SIZE * 8 && st.trophies.is_unlocked(&set.comm_id, t.id) {
                bits[bit / 8] |= 1 << (bit % 8);
            }
        }
        ctx.write_bytes(flags.addr(), &bits);
    }
    if !count.is_null() {
        ctx.write_u32(count.addr(), set.trophies.len() as u32);
    }
    0
}

/// int sceNpTrophyUnlockTrophy(SceNpTrophyContext context, SceNpTrophyHandle handle,
///     SceNpTrophyId trophyId, SceNpTrophyId *platinumId)
/// Record an unlock in this run's ledger, so the title reads back what it just earned.
///
/// `platinumId` reports the platinum trophy's id when THIS unlock completed the set (the
/// platinum is awarded by the system, never unlocked directly), and
/// `SCE_NP_TROPHY_INVALID_TROPHY_ID` otherwise.
#[hostcall]
pub(super) fn np_trophy_unlock_trophy(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, _handle: u32, trophy_id: u32, platinum_id: Ptr) -> i32 {
    unlock_trophy(ctx, st, context, trophy_id, platinum_id)
}

fn unlock_trophy(ctx: &mut GuestCtx, st: &mut VitaState, context: u32, trophy_id: u32, platinum_id: Ptr) -> i32 {
    let Some(comm_id) = st.trophies.comm_id_of(context).map(str::to_string) else {
        return SCE_NP_TROPHY_ERROR_INVALID_CONTEXT;
    };
    let set = st.trophies.set_for(context).expect("a bound context always has its set");
    let Some(trophy) = set.trophy(trophy_id) else {
        return SCE_NP_TROPHY_ERROR_INVALID_TROPHY_ID;
    };
    if trophy.grade == crate::trophy::Grade::Platinum {
        return SCE_NP_TROPHY_ERROR_PLATINUM_CANNOT_UNLOCK;
    }
    if st.trophies.is_unlocked(&comm_id, trophy_id) {
        return SCE_NP_TROPHY_ERROR_TROPHY_ALREADY_UNLOCKED;
    }
    // Which trophies must be unlocked for the platinum to be awarded, decided before the
    // ledger write so the set borrow ends here.
    let platinum = set.trophies.iter().find(|t| t.grade == crate::trophy::Grade::Platinum).map(|p| p.id);
    let rest: Vec<u32> = set.trophies.iter().map(|t| t.id).filter(|id| Some(*id) != platinum).collect();

    let tick = RTC_UNIX_EPOCH_TICKS + st.world.wall_us();
    st.trophies.unlock(&comm_id, trophy_id, tick);

    let awarded = match platinum {
        Some(p) if rest.iter().all(|id| st.trophies.is_unlocked(&comm_id, *id)) => {
            st.trophies.unlock(&comm_id, p, tick);
            p as i32
        }
        _ => SCE_NP_TROPHY_INVALID_TROPHY_ID,
    };
    tracing::info!(target: "vitaslop::cb", trophy = trophy_id, platinum = awarded, "sceNpTrophyUnlockTrophy");
    if !platinum_id.is_null() {
        ctx.write_u32(platinum_id.addr(), awarded as u32);
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

#[cfg(test)]
mod calendar_tests {
    //! The proleptic-Gregorian conversions the RTC calls share. Content-free: no game
    //! data, just the arithmetic. These exist because `sceRtcGetTick`/`sceRtcSetTick` are
    //! an INVERSE PAIR that titles round-trip through to do date arithmetic - and a
    //! calendar bug does not crash, it silently shifts a date by a day near a leap year.
    use super::{civil_from_days, days_from_civil};

    #[test]
    fn civil_and_days_are_exact_inverses_over_the_whole_rtc_range() {
        // The range is about 3.65 million days. Walk it with a PRIME stride, so the sample
        // is not aligned to the week, the 4/100/400-year leap rules or the 146097-day
        // Gregorian cycle - an aligned stride can miss the very cases the calendar gets
        // wrong. 997 gives a few thousand dates, which runs instantly.
        let (mut checked, mut leap_days) = (0u32, 0u32);
        let mut d = days_from_civil(1, 1, 1);
        let end = days_from_civil(9999, 12, 31);
        while d <= end {
            let (y, m, day) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, day), d, "round trip at day {d} -> {y}-{m}-{day}");
            assert!((1..=12).contains(&m), "month out of range at day {d}: {m}");
            assert!((1..=31).contains(&day), "day out of range at day {d}: {day}");
            if m == 2 && day == 29 {
                leap_days += 1;
            }
            checked += 1;
            d += 997;
        }
        assert!(checked > 3_000, "the stride should sample thousands of dates, got {checked}");
        assert!(leap_days > 0, "the sample should include at least one 29 February");
    }

    /// The month arithmetic `sceRtcTickAddMonths` does, over the calendar rather than over
    /// microseconds. Exercised here directly (rather than through the guest-facing call,
    /// which needs a GuestCtx) because the clamping rule is the whole subtlety.
    #[test]
    fn adding_months_clamps_the_day_into_the_target_month() {
        // The same absolute-month arithmetic `rtc_tick_add_calendar` performs.
        let add_months = |y: i64, m: i64, d: i64, n: i64| -> (i64, i64, i64) {
            let total = (y * 12 + (m - 1)) + n;
            let (ny, nm) = (total.div_euclid(12), total.rem_euclid(12) + 1);
            (ny, nm, d.min(super::days_in_month(ny, nm)))
        };
        // 31 January plus a month is the end of February, not "31 February".
        assert_eq!(add_months(2025, 1, 31, 1), (2025, 2, 28));
        assert_eq!(add_months(2024, 1, 31, 1), (2024, 2, 29), "2024 is a leap year");
        // A day that fits is untouched.
        assert_eq!(add_months(2025, 1, 15, 1), (2025, 2, 15));
        // Crossing a year boundary, forwards and backwards.
        assert_eq!(add_months(2025, 11, 10, 3), (2026, 2, 10));
        assert_eq!(add_months(2025, 2, 10, -3), (2024, 11, 10));
        // A whole year as twelve months, which is how AddYears is dispatched.
        assert_eq!(add_months(2024, 2, 29, 12), (2025, 2, 28), "29 Feb + 1 year clamps");
    }

    #[test]
    fn the_century_leap_rules_are_honoured() {
        // 2000 is a leap year (divisible by 400), 1900 and 2100 are not (divisible by 100).
        assert_eq!(civil_from_days(days_from_civil(2000, 2, 29)), (2000, 2, 29));
        // 1900-02-29 does not exist, so the day BEFORE 1900-03-01 must be 1900-02-28.
        assert_eq!(civil_from_days(days_from_civil(1900, 3, 1) - 1), (1900, 2, 28));
        assert_eq!(civil_from_days(days_from_civil(2100, 3, 1) - 1), (2100, 2, 28));
        assert_eq!(civil_from_days(days_from_civil(2004, 3, 1) - 1), (2004, 2, 29));
        // The Unix epoch is day zero, which pins the offset rather than just the deltas.
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }
}
