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
use crate::nid::services as nid;
use crate::SvcOutcome;

/// SceNetCtl connection state: disconnected (no link).
const SCE_NETCTL_STATE_DISCONNECTED: u32 = 0;
/// SceSysmodule "module is loaded" status.
const SCE_SYSMODULE_LOADED: i32 = 0;
/// A generic "no online account / not signed in" error, returned by the online
/// identity calls so the title takes its offline path instead of dereferencing an
/// account it will never get.
const SCE_NP_ERROR_SIGNED_OUT: i32 = 0x8055_0605u32 as i32;

pub fn try_dispatch(func_nid: u32, ctx: &mut GuestCtx, st: &mut VitaState) -> Option<SvcOutcome> {
    match func_nid {
        nid::SYSMODULE_IS_LOADED => sysmodule_is_loaded(ctx, st),
        nid::NET_CTL_INET_GET_STATE => netctl_inet_get_state(ctx, st),
        nid::NET_CTL_INET_REGISTER_CALLBACK => netctl_register_callback(ctx, st),
        nid::RTC_GET_CURRENT_CLOCK_LOCAL_TIME => rtc_get_current_clock_local_time(ctx, st),
        nid::APPUTIL_SYSTEM_PARAM_GET_INT => apputil_system_param_get_int(ctx, st),
        // No online account off-console: the identity calls report signed-out so the
        // title skips its online features rather than reading an account that is not
        // there (a null identity pointer would fault deep in the title's code).
        nid::NP_MANAGER_GET_NP_ID | nid::NP_SCORE_CREATE_TITLE_CTX => {
            ctx.ret(SCE_NP_ERROR_SIGNED_OUT as u32)
        }
        // Everything else here is an init/register that simply succeeds offline.
        nid::NET_INIT
        | nid::NET_CTL_INIT
        | nid::HTTP_INIT
        | nid::SSL_INIT
        | nid::NP_INIT
        | nid::NP_REGISTER_SERVICE_STATE_CALLBACK
        | nid::NP_BASIC_INIT
        | nid::NP_BASIC_REGISTER_HANDLER
        | nid::FIOS_OVERLAY_GET_LIST
        | nid::ULOBJ_REGISTER_PROTOCOL_REVISION
        | nid::APPUTIL_INIT
        | nid::NP_SCORE_INIT
        | nid::TOUCH_SET_SAMPLING_STATE => ctx.ret(0),
        _ => return None,
    }
    Some(SvcOutcome::Continue)
}

/// int sceAppUtilSystemParamGetInt(SceAppUtilSystemParamId id, SceInt32 *value)
/// Report a neutral default for any system parameter (language, button assign,
/// ...), written through `value`. Zero is a safe, in-range default; the title uses
/// it rather than an uninitialized read (which could index out of bounds).
#[hostcall]
fn apputil_system_param_get_int(ctx: &mut GuestCtx, _st: &mut VitaState, _id: u32, value: Ptr) -> i32 {
    if !value.is_null() {
        ctx.write_u32(value.addr(), 0);
    }
    0
}

/// int sceSysmoduleIsLoaded(SceSysmoduleModuleId id)
/// Report every queried module as already loaded, so the title does not block on a
/// load we cannot perform (the modules it needs are linked in already).
#[hostcall]
fn sysmodule_is_loaded(_st: &mut VitaState, _id: u32) -> i32 {
    SCE_SYSMODULE_LOADED
}

/// int sceNetCtlInetGetState(int *state)
/// Always disconnected: there is no network link.
#[hostcall]
fn netctl_inet_get_state(ctx: &mut GuestCtx, _st: &mut VitaState, state: Ptr) -> i32 {
    if !state.is_null() {
        ctx.write_u32(state.addr(), SCE_NETCTL_STATE_DISCONNECTED);
    }
    0
}

/// int sceNetCtlInetRegisterCallback(SceNetCtlCallback func, void *arg, int *cid)
/// Succeeds and hands back a callback id; the callback is never invoked offline.
#[hostcall]
fn netctl_register_callback(ctx: &mut GuestCtx, _st: &mut VitaState, _func: Ptr, _arg: Ptr, cid: Ptr) -> i32 {
    if !cid.is_null() {
        ctx.write_u32(cid.addr(), 0);
    }
    0
}

/// int sceRtcGetCurrentClockLocalTime(SceDateTime *time)
/// Fill a fixed, deterministic local date/time. `SceDateTime` is
/// {u16 year, month, day, hour, minute, second; u32 microsecond}.
#[hostcall]
fn rtc_get_current_clock_local_time(ctx: &mut GuestCtx, _st: &mut VitaState, time: Ptr) -> i32 {
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
