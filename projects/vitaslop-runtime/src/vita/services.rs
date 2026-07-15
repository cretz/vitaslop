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
/// fix, not the fix for OlliOlli's touch-driven front-end (that was `sceTouchRead`);
/// no button, Cross or Circle, drove the game-drawn connect dialog.
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
/// Succeeds and hands back a callback id; the callback is never invoked offline.
#[hostcall]
pub(super) fn netctl_register_callback(ctx: &mut GuestCtx, _st: &mut VitaState, _func: Ptr, _arg: Ptr, cid: Ptr) -> i32 {
    if !cid.is_null() {
        ctx.write_u32(cid.addr(), 0);
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
