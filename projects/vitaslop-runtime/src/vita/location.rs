//! SceLibLocation: the positioning service, backed by whatever location provider the
//! HOST has (real GPS/Wi-Fi positioning in the browser, nothing on a bare desktop).
//!
//! # Why this is a real implementation and not a `camera.rs`
//! [`crate::vita::camera`] models absent hardware, because no host this emulator runs on
//! has a Vita camera. Location is different: the browser has a genuine positioning
//! provider (the W3C Geolocation API), and the device this project targets is a PHONE,
//! where that provider is a real GPS. So the honest answer is not "there is none" - it
//! is "ask the host", which is what [`crate::world::World::poll_location`] does. A host
//! that really has no provider still says so, through
//! [`LocationPermission::Unavailable`], and gets the API's own
//! `SCE_LOCATION_ERROR_PROVIDER_UNAVAILABLE`.
//!
//! # The permission dialog maps onto the platform's own prompt
//! The Vita shows a system dialog the first time a title asks for position, driven by
//! `sceLocationConfirm` / `ConfirmGetStatus` / `ConfirmGetResult`. A browser shows its
//! own permission prompt the first time a page calls `watchPosition`. These are the same
//! event, so they are wired to each other rather than one being simulated: `Confirm`
//! asks the host to start acquiring (which raises the prompt), the guest's dialog reads
//! RUNNING while [`LocationPermission::Pending`], and FINISHED with ENABLE or DISABLE
//! when the user answers. Nothing here invents a user's decision.
//!
//! # What is deliberately NOT here
//! `sceLocationStartLocationCallback` / `StartHeadingCallback` deliver fixes by CALLING
//! A GUEST FUNCTION, and this runtime has no mechanism for calling back into guest code.
//! They are left unimplemented so a title that uses them hard-fails and names itself,
//! rather than being told a callback was registered that will never fire - which is
//! indistinguishable, from inside the title, from a device that never gets a fix.
//! `sceLocationSetGpsEmulationFile` is left out for the same reason: it names a file
//! format we have never seen, and accepting it would claim a behaviour we do not have.
//!
//! # Structure
//! Every entry point is a plain `do_*` function holding the logic, with a thin
//! `#[hostcall]` wrapper. That is not decoration: a `#[hostcall]` body is spliced into a
//! generated wrapper, so an early `return` inside one leaves the WRAPPER rather than the
//! handler (`vita::sync` and `vita::services` record the same trap). These handlers are
//! almost all guard clauses, so they need real early returns - and the split has the
//! side benefit that the logic is directly unit-testable without a guest context.
//!
//! Signatures, struct layouts and the error enum are from `psp2/location.h` (vitasdk,
//! MIT) and the NIDs from `db/360/SceLibLocation.yml`, so nothing here is guessed.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;
use crate::world::LocationPermission;

// ---------------------------------------------------------------------------
// Error and enum values, all from `psp2/location.h`.
// ---------------------------------------------------------------------------

/// The position could not be determined right now. A Vita indoors reports this; so does
/// a provider that has been permitted but has not produced a first fix yet.
const SCE_LOCATION_INFO_UNDETERMINED_LOCATION: i32 = 0x8010_1200u32 as i32;
/// The user refused this title access to position.
const SCE_LOCATION_INFO_DENIED_BY_USER: i32 = 0x8010_1203u32 as i32;
/// A pointer argument was null or otherwise unusable.
const SCE_LOCATION_ERROR_INVALID_ADDRESS: i32 = 0x8010_1204u32 as i32;
/// The handle is not one this library handed out.
const SCE_LOCATION_ERROR_INVALID_HANDLE: i32 = 0x8010_1205u32 as i32;
/// No more handles can be allocated.
const SCE_LOCATION_ERROR_TOO_MANY_HANDLES: i32 = 0x8010_1207u32 as i32;
/// `locateMethod` is not a `SceLocationLocationMethod`.
const SCE_LOCATION_ERROR_INVALID_LOCATION_METHOD: i32 = 0x8010_1208u32 as i32;
/// `headingMethod` is not a `SceLocationHeadingMethod`.
const SCE_LOCATION_ERROR_INVALID_HEADING_METHOD: i32 = 0x8010_1209u32 as i32;
/// `ConfirmGetResult` was called before the dialog finished.
const SCE_LOCATION_ERROR_DIALOG_RESULT_NONE: i32 = 0x8010_120Cu32 as i32;
/// A second `Confirm` while one is already running.
const SCE_LOCATION_ERROR_MULTIPLE_CONFIRM: i32 = 0x8010_120Eu32 as i32;
/// The positioning provider itself is not available on this host.
const SCE_LOCATION_ERROR_PROVIDER_UNAVAILABLE: i32 = 0x8010_1281u32 as i32;

/// `SCE_LOCATION_DATA_INVALID` - the value the API reserves for a field that could not
/// be obtained. Note it is defined as a DOUBLE literal (-9999.0) and appears in both the
/// `SceDouble64` and `SceFloat32` fields, so it is written at each field's own width.
const SCE_LOCATION_DATA_INVALID: f64 = -9999.0;

/// `SceLocationDialogStatus`.
const DIALOG_STATUS_IDLE: u32 = 0;
const DIALOG_STATUS_RUNNING: u32 = 1;
const DIALOG_STATUS_FINISHED: u32 = 2;

/// `SceLocationDialogResult`.
const DIALOG_RESULT_NONE: u32 = 0;
const DIALOG_RESULT_DISABLE: u32 = 1;
const DIALOG_RESULT_ENABLE: u32 = 2;

/// `SceLocationPermissionStatus`.
const PERMISSION_DENY: u32 = 0;
const PERMISSION_ALLOW: u32 = 1;

/// `SceLocationPermissionApplicationStatus`.
const PERMISSION_APPLICATION_INIT: u32 = 1;
const PERMISSION_APPLICATION_DENY: u32 = 2;
const PERMISSION_APPLICATION_ALLOW: u32 = 3;

/// The highest valid `SceLocationLocationMethod` (`SCE_LOCATION_LMETHOD_GPS`).
const MAX_LOCATION_METHOD: u32 = 5;
/// The highest valid `SceLocationHeadingMethod` (`SCE_LOCATION_HMETHOD_CAMERA`).
const MAX_HEADING_METHOD: u32 = 4;

/// `sizeof(SceLocationLocationInfo)`, asserted by the vitasdk header.
const LOCATION_INFO_BYTES: usize = 0x30;
/// `sizeof(SceLocationHeadingInfo)`, asserted by the vitasdk header.
const HEADING_INFO_BYTES: usize = 0x20;
/// `sizeof(SceLocationPermissionInfo)`, asserted by the vitasdk header.
const PERMISSION_INFO_BYTES: usize = 0x14;

/// How many handles this table hands out before reporting
/// `SCE_LOCATION_ERROR_TOO_MANY_HANDLES`.
///
/// The console's own limit is UNMEASURED - the error exists in the header, but no
/// published source gives its value and we have no device to count it on. So this is
/// this table's capacity, not a claim about the hardware, and it is set well above any
/// plausible use (a title opens one handle) so a correct title never meets it.
const MAX_HANDLES: usize = 8;

/// One open `sceLocationOpen` handle.
#[derive(Clone, Copy, Debug)]
struct Handle {
    /// The value handed to the guest. Never 0, so a zeroed variable is never a valid
    /// handle by accident.
    id: u32,
    locate_method: u32,
    heading_method: u32,
}

/// Per-run SceLibLocation state: the open handles and the permission-dialog state.
///
/// The dialog is per-LIBRARY rather than per-handle: it is one system dialog about one
/// decision ("may this title use your location"), and the answer does not depend on
/// which handle asked. Keying it by handle would let two handles hold contradictory
/// answers to the same question.
#[derive(Default)]
pub(crate) struct LocationState {
    handles: Vec<Handle>,
    /// Next handle value to hand out. Starts at 1 - see [`Handle::id`].
    next_id: u32,
    /// Whether `sceLocationConfirm` has been called and not yet aborted. Drives the
    /// difference between IDLE (never asked) and RUNNING/FINISHED.
    confirm_started: bool,
    /// Whether a `note_no_provider` line has already been printed this run.
    noted_no_provider: bool,
}

impl LocationState {
    fn find(&self, id: u32) -> Option<&Handle> {
        self.handles.iter().find(|h| h.id == id)
    }
    fn find_mut(&mut self, id: u32) -> Option<&mut Handle> {
        self.handles.iter_mut().find(|h| h.id == id)
    }
}

/// Report, once, that a title asked for its position on a host that cannot supply one.
///
/// Like [`crate::vita::camera`]'s note, this is not an emulator failure but it IS a
/// difference from the console that changes what the title does, so it says so out loud
/// rather than being silently absent from the log.
fn note_no_provider(st: &mut VitaState) {
    if st.location.noted_no_provider {
        return;
    }
    st.location.noted_no_provider = true;
    eprintln!(
        "sceLocation: this title asked for its position and this host has no location \
         provider - reporting SCE_LOCATION_ERROR_PROVIDER_UNAVAILABLE. Any location-driven \
         feature will be unavailable, which is a real console state, not a stub. The \
         browser engine supplies a real provider."
    );
}

/// Validate the two method arguments common to `Open` and `Reopen`.
fn check_methods(locate_method: u32, heading_method: u32) -> Result<(), i32> {
    if locate_method > MAX_LOCATION_METHOD {
        return Err(SCE_LOCATION_ERROR_INVALID_LOCATION_METHOD);
    }
    if heading_method > MAX_HEADING_METHOD {
        return Err(SCE_LOCATION_ERROR_INVALID_HEADING_METHOD);
    }
    Ok(())
}

/// Write `SCE_LOCATION_DATA_INVALID` into every data field of a
/// `SceLocationLocationInfo` at `addr`, leaving the timestamp zero.
///
/// # Why this writes rather than leaving the buffer alone
/// [`crate::vita::camera`] deliberately does not touch its out-parameter, because there
/// is no way to say "no frame" inside a `SceCameraRead` - any bytes it wrote would be a
/// picture. This API is the opposite case: it DEFINES a value for "this field could not
/// be obtained" (`SCE_LOCATION_DATA_INVALID`), so writing it is not inventing data, it
/// is saying "no data" in the API's own vocabulary. Leaving the buffer untouched would
/// hand a caller that ignores the return code whatever was on its stack, which really
/// would be an invented position.
fn write_undetermined_location(ctx: &mut GuestCtx, addr: u32) {
    let mut buf = [0u8; LOCATION_INFO_BYTES];
    let d = SCE_LOCATION_DATA_INVALID.to_le_bytes();
    let f = (SCE_LOCATION_DATA_INVALID as f32).to_le_bytes();
    buf[0x00..0x08].copy_from_slice(&d); // latitude
    buf[0x08..0x10].copy_from_slice(&d); // longitude
    buf[0x10..0x18].copy_from_slice(&d); // altitude
    buf[0x18..0x1C].copy_from_slice(&f); // accuracy
    buf[0x1C..0x20].copy_from_slice(&f); // reserve
    buf[0x20..0x24].copy_from_slice(&f); // direction
    buf[0x24..0x28].copy_from_slice(&f); // speed
    // timestamp stays 0: there is no acquisition to date-stamp.
    ctx.write_bytes(addr, &buf);
}

// ---------------------------------------------------------------------------
// Entry points. Each is `do_<name>` (the logic) plus a `#[hostcall]` wrapper.
// ---------------------------------------------------------------------------

/// See [`open`].
fn do_open(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    handle_addr: u32,
    locate_method: u32,
    heading_method: u32,
) -> i32 {
    if handle_addr == 0 {
        return SCE_LOCATION_ERROR_INVALID_ADDRESS;
    }
    if let Err(e) = check_methods(locate_method, heading_method) {
        return e;
    }
    if st.world.location_permission() == LocationPermission::Unavailable {
        note_no_provider(st);
        return SCE_LOCATION_ERROR_PROVIDER_UNAVAILABLE;
    }
    if st.location.handles.len() >= MAX_HANDLES {
        return SCE_LOCATION_ERROR_TOO_MANY_HANDLES;
    }
    st.location.next_id += 1;
    let id = st.location.next_id;
    st.location.handles.push(Handle { id, locate_method, heading_method });
    ctx.write_u32(handle_addr, id);
    0
}

/// SceInt32 sceLocationOpen(SceLocationHandle *handle, SceLocationLocationMethod locateMethod, SceLocationHeadingMethod headingMethod)
///
/// Opening the library is not the same as obtaining a fix: on a console this succeeds
/// indoors, and the position arrives (or does not) later through `GetLocation`. So the
/// only thing that fails here is a host with no provider at all, which is the condition
/// `SCE_LOCATION_ERROR_PROVIDER_UNAVAILABLE` names.
///
/// `*handle` is written ONLY on success. A failed open leaves the caller's variable
/// exactly as it was - the observed call site passes a word holding `0xffffffff`, and
/// writing a plausible handle into it after failing would be handing out a handle that
/// no later call would accept.
#[hostcall]
pub(super) fn open(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    handle: Ptr,
    locate_method: u32,
    heading_method: u32,
) -> i32 {
    do_open(ctx, st, handle.addr(), locate_method, heading_method)
}

/// See [`close`].
fn do_close(st: &mut VitaState, handle: u32) -> i32 {
    let before = st.location.handles.len();
    st.location.handles.retain(|h| h.id != handle);
    if st.location.handles.len() == before {
        return SCE_LOCATION_ERROR_INVALID_HANDLE;
    }
    if st.location.handles.is_empty() {
        st.location.confirm_started = false;
        st.world.release_location();
    }
    0
}

/// SceInt32 sceLocationClose(SceLocationHandle handle)
///
/// Releasing the host provider is tied to the LAST handle closing, not to any one close:
/// two handles are two views of one device, and stopping acquisition while another
/// handle is still open would silently starve it.
#[hostcall]
pub(super) fn close(_ctx: &mut GuestCtx, st: &mut VitaState, handle: u32) -> i32 {
    do_close(st, handle)
}

/// See [`reopen`].
fn do_reopen(st: &mut VitaState, handle: u32, locate_method: u32, heading_method: u32) -> i32 {
    if let Err(e) = check_methods(locate_method, heading_method) {
        return e;
    }
    match st.location.find_mut(handle) {
        Some(h) => {
            h.locate_method = locate_method;
            h.heading_method = heading_method;
            0
        }
        None => SCE_LOCATION_ERROR_INVALID_HANDLE,
    }
}

/// SceInt32 sceLocationReopen(SceLocationHandle handle, SceLocationLocationMethod locateMethod, SceLocationHeadingMethod headingMethod)
#[hostcall]
pub(super) fn reopen(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    handle: u32,
    locate_method: u32,
    heading_method: u32,
) -> i32 {
    do_reopen(st, handle, locate_method, heading_method)
}

/// See [`get_method`].
fn do_get_method(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    handle: u32,
    locate_addr: u32,
    heading_addr: u32,
) -> i32 {
    let Some(h) = st.location.find(handle).copied() else {
        return SCE_LOCATION_ERROR_INVALID_HANDLE;
    };
    if locate_addr != 0 {
        ctx.write_u32(locate_addr, h.locate_method);
    }
    if heading_addr != 0 {
        ctx.write_u32(heading_addr, h.heading_method);
    }
    0
}

/// SceInt32 sceLocationGetMethod(SceLocationHandle handle, SceLocationLocationMethod *locateMethod, SceLocationHeadingMethod *headingMethod)
///
/// Both out-parameters are optional (a caller may want only one), so a null pointer is
/// skipped rather than being an error - the header gives no indication either is
/// mandatory, and refusing the call would break a caller asking a legal question.
#[hostcall]
pub(super) fn get_method(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    handle: u32,
    locate_method: Ptr,
    heading_method: Ptr,
) -> i32 {
    do_get_method(ctx, st, handle, locate_method.addr(), heading_method.addr())
}

/// See [`confirm`].
fn do_confirm(st: &mut VitaState, handle: u32) -> i32 {
    if st.location.find(handle).is_none() {
        return SCE_LOCATION_ERROR_INVALID_HANDLE;
    }
    // A second Confirm while one is still awaiting an answer is the error the API names.
    // Once the user HAS answered, a repeat Confirm is not "multiple" - the dialog is
    // finished - and re-asking a permitted host is harmless and idempotent.
    if st.location.confirm_started && st.world.location_permission() == LocationPermission::Pending
    {
        return SCE_LOCATION_ERROR_MULTIPLE_CONFIRM;
    }
    st.location.confirm_started = true;
    st.world.request_location();
    0
}

/// SceInt32 sceLocationConfirm(SceLocationHandle handle)
///
/// The guest asking to show the system permission dialog. On the host this starts
/// acquisition, which is what raises the platform's own prompt - see the module docs.
#[hostcall]
pub(super) fn confirm(_ctx: &mut GuestCtx, st: &mut VitaState, handle: u32) -> i32 {
    do_confirm(st, handle)
}

/// See [`confirm_get_status`].
fn do_confirm_get_status(ctx: &mut GuestCtx, st: &mut VitaState, handle: u32, addr: u32) -> i32 {
    if st.location.find(handle).is_none() {
        return SCE_LOCATION_ERROR_INVALID_HANDLE;
    }
    if addr == 0 {
        return SCE_LOCATION_ERROR_INVALID_ADDRESS;
    }
    let value = if !st.location.confirm_started {
        DIALOG_STATUS_IDLE
    } else {
        match st.world.location_permission() {
            LocationPermission::Pending => DIALOG_STATUS_RUNNING,
            LocationPermission::Granted | LocationPermission::Denied => DIALOG_STATUS_FINISHED,
            // A provider that vanished, or one that has somehow not been asked despite
            // Confirm having run: the dialog is not showing and has no answer, which is
            // IDLE. Reporting FINISHED here would send the caller to GetResult for an
            // answer that does not exist.
            LocationPermission::Unavailable | LocationPermission::NotAsked => DIALOG_STATUS_IDLE,
        }
    };
    ctx.write_u32(addr, value);
    0
}

/// SceInt32 sceLocationConfirmGetStatus(SceLocationHandle handle, SceLocationDialogStatus *status)
///
/// The dialog's status is DERIVED from the host's permission state rather than stored,
/// so it cannot drift from what the platform prompt is actually doing.
#[hostcall]
pub(super) fn confirm_get_status(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    handle: u32,
    status: Ptr,
) -> i32 {
    do_confirm_get_status(ctx, st, handle, status.addr())
}

/// See [`confirm_get_result`].
fn do_confirm_get_result(ctx: &mut GuestCtx, st: &mut VitaState, handle: u32, addr: u32) -> i32 {
    if st.location.find(handle).is_none() {
        return SCE_LOCATION_ERROR_INVALID_HANDLE;
    }
    if addr == 0 {
        return SCE_LOCATION_ERROR_INVALID_ADDRESS;
    }
    let started = st.location.confirm_started;
    let value = match st.world.location_permission() {
        LocationPermission::Granted if started => DIALOG_RESULT_ENABLE,
        LocationPermission::Denied if started => DIALOG_RESULT_DISABLE,
        _ => DIALOG_RESULT_NONE,
    };
    // The API distinguishes "no result stored" as an ERROR as well as a value, so a
    // caller that checks only the return code and one that reads only the out-parameter
    // reach the same conclusion.
    if value == DIALOG_RESULT_NONE {
        ctx.write_u32(addr, DIALOG_RESULT_NONE);
        return SCE_LOCATION_ERROR_DIALOG_RESULT_NONE;
    }
    ctx.write_u32(addr, value);
    0
}

/// SceInt32 sceLocationConfirmGetResult(SceLocationHandle handle, SceLocationDialogResult *result)
#[hostcall]
pub(super) fn confirm_get_result(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    handle: u32,
    result: Ptr,
) -> i32 {
    do_confirm_get_result(ctx, st, handle, result.addr())
}

/// See [`confirm_abort`].
fn do_confirm_abort(st: &mut VitaState, handle: u32) -> i32 {
    if st.location.find(handle).is_none() {
        return SCE_LOCATION_ERROR_INVALID_HANDLE;
    }
    st.location.confirm_started = false;
    0
}

/// SceInt32 sceLocationConfirmAbort(SceLocationHandle handle)
///
/// Aborts the guest's dialog. It does NOT dismiss the platform's own prompt - a browser
/// permission prompt cannot be withdrawn by the page - so this returns the guest's
/// dialog to IDLE and leaves the host to finish answering in its own time. That is the
/// faithful outcome: the title stops waiting, and if the user later grants permission
/// the provider is live for the next `Confirm`.
#[hostcall]
pub(super) fn confirm_abort(_ctx: &mut GuestCtx, st: &mut VitaState, handle: u32) -> i32 {
    do_confirm_abort(st, handle)
}

/// The shared body of [`get_location`] and [`get_location_with_timeout`].
fn do_get_location(ctx: &mut GuestCtx, st: &mut VitaState, handle: u32, addr: u32) -> i32 {
    if st.location.find(handle).is_none() {
        return SCE_LOCATION_ERROR_INVALID_HANDLE;
    }
    if addr == 0 {
        return SCE_LOCATION_ERROR_INVALID_ADDRESS;
    }
    match st.world.location_permission() {
        LocationPermission::Unavailable => {
            note_no_provider(st);
            write_undetermined_location(ctx, addr);
            return SCE_LOCATION_ERROR_PROVIDER_UNAVAILABLE;
        }
        LocationPermission::Denied => {
            write_undetermined_location(ctx, addr);
            return SCE_LOCATION_INFO_DENIED_BY_USER;
        }
        // Permitted, or not yet asked: either way the question below is the same one -
        // is there a fix? A provider that has not been asked simply never has one.
        LocationPermission::Granted | LocationPermission::NotAsked | LocationPermission::Pending => {
        }
    }
    let Some(fix) = st.world.poll_location() else {
        write_undetermined_location(ctx, addr);
        return SCE_LOCATION_INFO_UNDETERMINED_LOCATION;
    };

    // Each field is written at its own width, and an absent one gets the API's INVALID
    // sentinel rather than a zero - zero is a real latitude, a real speed and a real
    // heading, so it must never stand for "unknown". `reserve` stays zero, not INVALID:
    // it is not a measurement that failed.
    let mut buf = [0u8; LOCATION_INFO_BYTES];
    let opt_d = |v: Option<f64>| v.unwrap_or(SCE_LOCATION_DATA_INVALID).to_le_bytes();
    let opt_f = |v: Option<f32>| v.unwrap_or(SCE_LOCATION_DATA_INVALID as f32).to_le_bytes();
    buf[0x00..0x08].copy_from_slice(&fix.latitude_deg.to_le_bytes());
    buf[0x08..0x10].copy_from_slice(&fix.longitude_deg.to_le_bytes());
    buf[0x10..0x18].copy_from_slice(&opt_d(fix.altitude_m));
    buf[0x18..0x1C].copy_from_slice(&opt_f(fix.accuracy_m));
    buf[0x20..0x24].copy_from_slice(&opt_f(fix.direction_deg));
    buf[0x24..0x28].copy_from_slice(&opt_f(fix.speed_mps));
    buf[0x28..0x30].copy_from_slice(&fix.timestamp_us.to_le_bytes());
    ctx.write_bytes(addr, &buf);
    0
}

/// SceInt32 sceLocationGetLocation(SceLocationHandle handle, SceLocationLocationInfo *locationInfo)
///
/// # This does not block
/// On a console this call waits for a fix. Parking the calling thread here would be more
/// literal, but the host provider is event-driven (a browser delivers a position through
/// a callback on the page's thread), so a wait would have to be woken by that event
/// anyway - and a title that polls, which is what the observed caller does through its
/// own dialog loop, reaches the same fix a frame or two later either way. What matters
/// for faithfulness is that "no fix yet" is reported as
/// `SCE_LOCATION_INFO_UNDETERMINED_LOCATION`, which is exactly the code a console gives
/// for it, and never as a fabricated position.
#[hostcall]
pub(super) fn get_location(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    handle: u32,
    location_info: Ptr,
) -> i32 {
    do_get_location(ctx, st, handle, location_info.addr())
}

/// SceInt32 sceLocationGetLocationWithTimeout(SceLocationHandle handle, SceLocationLocationInfo *locationInfo, SceUInt32 timeout)
///
/// Since [`get_location`] does not block, the timeout has nothing to elapse against and
/// this is the same call. It is a distinct NID, so it gets a distinct handler rather
/// than sharing one by aliasing - a future blocking implementation must be able to tell
/// them apart.
#[hostcall]
pub(super) fn get_location_with_timeout(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    handle: u32,
    location_info: Ptr,
    _timeout: u32,
) -> i32 {
    do_get_location(ctx, st, handle, location_info.addr())
}

/// See [`cancel_get_location`].
fn do_cancel_get_location(st: &mut VitaState, handle: u32) -> i32 {
    if st.location.find(handle).is_none() {
        return SCE_LOCATION_ERROR_INVALID_HANDLE;
    }
    0
}

/// SceInt32 sceLocationCancelGetLocation(SceLocationHandle handle)
///
/// Cancels an in-flight `GetLocation`. Nothing is ever in flight here (the call returns
/// immediately), so this validates the handle and succeeds - there is no pending
/// operation left running, which is the state the caller wants.
#[hostcall]
pub(super) fn cancel_get_location(_ctx: &mut GuestCtx, st: &mut VitaState, handle: u32) -> i32 {
    do_cancel_get_location(st, handle)
}

/// See [`get_heading`].
fn do_get_heading(ctx: &mut GuestCtx, st: &mut VitaState, handle: u32, addr: u32) -> i32 {
    if st.location.find(handle).is_none() {
        return SCE_LOCATION_ERROR_INVALID_HANDLE;
    }
    if addr == 0 {
        return SCE_LOCATION_ERROR_INVALID_ADDRESS;
    }
    let mut buf = [0u8; HEADING_INFO_BYTES];
    let f = (SCE_LOCATION_DATA_INVALID as f32).to_le_bytes();
    for off in [0x00usize, 0x04, 0x08, 0x0C, 0x10, 0x14] {
        buf[off..off + 4].copy_from_slice(&f);
    }
    ctx.write_bytes(addr, &buf);
    SCE_LOCATION_INFO_UNDETERMINED_LOCATION
}

/// SceInt32 sceLocationGetHeading(SceLocationHandle handle, SceLocationHeadingInfo *headingInfo)
///
/// A heading is a MAGNETOMETER reading - which way the device is pointed - and this
/// emulator has no compass source: [`crate::world::World`] carries position, not
/// orientation. So this reports the API's own "could not determine" rather than deriving
/// a heading from the direction of travel, which is a different quantity and would be
/// wrong whenever the device is held at an angle to its motion, or is not moving at all.
#[hostcall]
pub(super) fn get_heading(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    handle: u32,
    heading_info: Ptr,
) -> i32 {
    do_get_heading(ctx, st, handle, heading_info.addr())
}

/// See [`get_permission`].
fn do_get_permission(ctx: &mut GuestCtx, st: &mut VitaState, handle: u32, addr: u32) -> i32 {
    if st.location.find(handle).is_none() {
        return SCE_LOCATION_ERROR_INVALID_HANDLE;
    }
    if addr == 0 {
        return SCE_LOCATION_ERROR_INVALID_ADDRESS;
    }
    let (main, app) = match st.world.location_permission() {
        // No provider is not a denial by anyone; it is the system switch being off as
        // far as a title can tell, which is what `mainstatus` says.
        LocationPermission::Unavailable => (PERMISSION_DENY, PERMISSION_APPLICATION_DENY),
        LocationPermission::NotAsked | LocationPermission::Pending => {
            (PERMISSION_ALLOW, PERMISSION_APPLICATION_INIT)
        }
        LocationPermission::Granted => (PERMISSION_ALLOW, PERMISSION_APPLICATION_ALLOW),
        LocationPermission::Denied => (PERMISSION_ALLOW, PERMISSION_APPLICATION_DENY),
    };
    let mut buf = [0u8; PERMISSION_INFO_BYTES];
    buf[0x00..0x04].copy_from_slice(&PERMISSION_ALLOW.to_le_bytes()); // parentalstatus
    buf[0x04..0x08].copy_from_slice(&main.to_le_bytes());
    buf[0x08..0x0C].copy_from_slice(&app.to_le_bytes());
    // unk_0x0C / unk_0x10 are undocumented; they stay zero rather than being given a
    // value we cannot justify.
    ctx.write_bytes(addr, &buf);
    0
}

/// SceInt32 sceLocationGetPermission(SceLocationHandle handle, SceLocationPermissionInfo *info)
///
/// Three independent permissions on a console: parental control, the system-wide
/// location switch, and this title's own entry in settings. A host has ONE answer, so
/// the parental and system-wide gates report ALLOW (nothing on this host forbids it) and
/// the per-application status carries the real state - which is the field that actually
/// varies and the one a title acts on.
#[hostcall]
pub(super) fn get_permission(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    handle: u32,
    info: Ptr,
) -> i32 {
    do_get_permission(ctx, st, handle, info.addr())
}

/// SceInt32 sceLocationDenyApplication(void)
///
/// The title revoking its OWN permission. It is a statement to the system, not a
/// question, and it takes effect for this run by releasing the host provider.
#[hostcall]
pub(super) fn deny_application(_ctx: &mut GuestCtx, st: &mut VitaState) -> i32 {
    st.location.confirm_started = false;
    st.world.release_location();
    0
}

/// SceInt32 sceLocationInit(void)
///
/// Library initialisation. All per-run state lives in [`LocationState`], which is
/// constructed with the run, so there is nothing to allocate - but this must SUCCEED
/// rather than be absent, because a title that checks it will not call anything else.
#[hostcall]
pub(super) fn init(_ctx: &mut GuestCtx, _st: &mut VitaState) -> i32 {
    0
}

/// SceInt32 sceLocationTerm(void)
///
/// Library teardown: drops every handle and stops the host provider. A title calling
/// this and then re-initialising gets a clean library, which is what `Term` promises.
#[hostcall]
pub(super) fn term(_ctx: &mut GuestCtx, st: &mut VitaState) -> i32 {
    st.location.handles.clear();
    st.location.confirm_started = false;
    st.world.release_location();
    0
}

/// SceInt32 sceLocationSetThreadParameter(int a1, int a2)
///
/// Sets the priority and affinity of the library's internal worker thread. This
/// implementation has no such thread - fixes arrive from the host, not from a guest-side
/// poller - so there is nothing to configure and the call succeeds. Nothing observable
/// to the guest depends on it: it returns no data and changes no behaviour it can see.
#[hostcall]
pub(super) fn set_thread_parameter(
    _ctx: &mut GuestCtx,
    _st: &mut VitaState,
    _a1: u32,
    _a2: u32,
) -> i32 {
    0
}

#[cfg(test)]
mod tests {
    //! The SceLibLocation surface, driven through the REAL host-call entry points so the
    //! AAPCS marshalling and the guest struct layout are covered as well as the logic.
    //! Every offset asserted here comes from `psp2/location.h`, whose
    //! `VITASDK_BUILD_ASSERT_EQ` lines fix the struct sizes.

    use super::*;
    use crate::host::{SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::REG_COUNT;
    use crate::world::{CtrlFrame, LocationFix, World};
    use std::sync::{Arc, Mutex};

    /// What the test world was ASKED to do, observable while `VitaState` owns it.
    #[derive(Default)]
    struct Calls {
        requests: u32,
        releases: u32,
    }

    /// A world whose location provider is whatever the test says it is.
    struct TestWorld {
        permission: LocationPermission,
        fix: Option<LocationFix>,
        calls: Arc<Mutex<Calls>>,
    }

    impl World for TestWorld {
        fn monotonic_us(&mut self) -> u64 {
            0
        }
        fn wall_us(&mut self) -> u64 {
            0
        }
        fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
            CtrlFrame::default()
        }
        fn fill_random(&mut self, _buf: &mut [u8]) {}
        fn location_permission(&mut self) -> LocationPermission {
            self.permission
        }
        fn request_location(&mut self) {
            self.calls.lock().unwrap().requests += 1;
        }
        fn release_location(&mut self) {
            self.calls.lock().unwrap().releases += 1;
        }
        fn poll_location(&mut self) -> Option<LocationFix> {
            self.fix
        }
    }

    /// A fully-populated fix, so every optional field is exercised as PRESENT.
    fn full_fix() -> LocationFix {
        LocationFix {
            latitude_deg: 35.6586,
            longitude_deg: 139.7454,
            altitude_m: Some(41.5),
            accuracy_m: Some(12.25),
            direction_deg: Some(97.5),
            speed_mps: Some(3.75),
            timestamp_us: 1_700_000_000_000_000,
        }
    }

    /// A fix with only the two mandatory fields, so the INVALID sentinel is exercised.
    fn sparse_fix() -> LocationFix {
        LocationFix {
            latitude_deg: -33.8688,
            longitude_deg: 151.2093,
            altitude_m: None,
            accuracy_m: None,
            direction_deg: None,
            speed_mps: None,
            timestamp_us: 42,
        }
    }

    /// One host call, as the generated wrappers are shaped.
    type Entry = fn(&mut GuestCtx, &mut VitaState);

    /// A run: guest memory, a register file, and a `VitaState` over a `TestWorld`.
    struct Run {
        regs: [u32; REG_COUNT],
        vfp: [u32; VFP_ARG_COUNT],
        bytes: Vec<u8>,
        st: VitaState,
        calls: Arc<Mutex<Calls>>,
    }

    impl Run {
        fn new(permission: LocationPermission, fix: Option<LocationFix>) -> Self {
            let calls = Arc::new(Mutex::new(Calls::default()));
            let w = TestWorld { permission, fix, calls: calls.clone() };
            Run {
                regs: [0; REG_COUNT],
                vfp: [0; VFP_ARG_COUNT],
                bytes: vec![0u8; 0x400],
                st: VitaState::new(0, 0x400, Box::new(w)),
                calls,
            }
        }

        /// Call `f` as the guest would, with `args` in r0.. and the result read from r0.
        fn call(&mut self, f: Entry, args: &[u32]) -> i32 {
            self.regs = [0; REG_COUNT];
            for (i, a) in args.iter().enumerate() {
                self.regs[i] = *a;
            }
            {
                let mut mem = SliceMemory(&mut self.bytes);
                let mut ctx = GuestCtx::new(&mut self.regs, &mut self.vfp, &mut mem, 0);
                f(&mut ctx, &mut self.st);
            }
            self.regs[0] as i32
        }

        fn requests(&self) -> u32 {
            self.calls.lock().unwrap().requests
        }
        fn releases(&self) -> u32 {
            self.calls.lock().unwrap().releases
        }

        fn word(&self, addr: u32) -> u32 {
            u32::from_le_bytes(self.bytes[addr as usize..addr as usize + 4].try_into().unwrap())
        }
        fn f64_at(&self, addr: u32) -> f64 {
            f64::from_le_bytes(self.bytes[addr as usize..addr as usize + 8].try_into().unwrap())
        }
        fn f32_at(&self, addr: u32) -> f32 {
            f32::from_le_bytes(self.bytes[addr as usize..addr as usize + 4].try_into().unwrap())
        }
        fn u64_at(&self, addr: u32) -> u64 {
            u64::from_le_bytes(self.bytes[addr as usize..addr as usize + 8].try_into().unwrap())
        }

        /// Open one handle and return it, asserting the open succeeded. The arguments are
        /// the exact ones the observed retail call site passes: locateMethod 1
        /// (AGPS_AND_3G_AND_WIFI), headingMethod 0 (NONE).
        fn open_handle(&mut self) -> u32 {
            assert_eq!(self.call(open, &[0x100, 1, 0]), 0, "open should succeed");
            self.word(0x100)
        }
    }

    // -- open -------------------------------------------------------------------

    /// A host with no location provider reports the API's own PROVIDER_UNAVAILABLE and
    /// does NOT write a handle - the retail call site passes a word holding 0xffffffff,
    /// and leaving it alone is what keeps a failed open from looking like a successful
    /// one.
    #[test]
    fn open_without_a_provider_reports_unavailable_and_writes_no_handle() {
        let mut r = Run::new(LocationPermission::Unavailable, None);
        r.bytes[0x100..0x104].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert_eq!(r.call(open, &[0x100, 1, 0]), SCE_LOCATION_ERROR_PROVIDER_UNAVAILABLE);
        assert_eq!(r.word(0x100), 0xFFFF_FFFF, "the caller's variable must be untouched");
    }

    #[test]
    fn open_hands_out_a_nonzero_handle() {
        let mut r = Run::new(LocationPermission::NotAsked, None);
        assert_ne!(r.open_handle(), 0, "0 must never be a valid handle");
    }

    #[test]
    fn open_rejects_a_null_handle_pointer() {
        let mut r = Run::new(LocationPermission::NotAsked, None);
        assert_eq!(r.call(open, &[0, 1, 0]), SCE_LOCATION_ERROR_INVALID_ADDRESS);
    }

    /// The two method arguments are validated separately and each names its own error -
    /// a caller passing a bad heading method must not be told its location method is
    /// wrong.
    #[test]
    fn open_validates_each_method_against_its_own_error() {
        let mut r = Run::new(LocationPermission::NotAsked, None);
        assert_eq!(
            r.call(open, &[0x100, MAX_LOCATION_METHOD + 1, 0]),
            SCE_LOCATION_ERROR_INVALID_LOCATION_METHOD
        );
        assert_eq!(
            r.call(open, &[0x100, 1, MAX_HEADING_METHOD + 1]),
            SCE_LOCATION_ERROR_INVALID_HEADING_METHOD
        );
    }

    #[test]
    fn open_reports_too_many_handles_at_the_table_limit() {
        let mut r = Run::new(LocationPermission::NotAsked, None);
        for _ in 0..MAX_HANDLES {
            assert_eq!(r.call(open, &[0x100, 1, 0]), 0);
        }
        assert_eq!(r.call(open, &[0x100, 1, 0]), SCE_LOCATION_ERROR_TOO_MANY_HANDLES);
    }

    /// Two opens that returned the same value would let a close of one silently close
    /// the other.
    #[test]
    fn each_open_returns_a_distinct_handle() {
        let mut r = Run::new(LocationPermission::NotAsked, None);
        let a = r.open_handle();
        let b = r.open_handle();
        assert_ne!(a, b);
    }

    // -- close / term -----------------------------------------------------------

    #[test]
    fn close_rejects_a_handle_it_never_issued() {
        let mut r = Run::new(LocationPermission::NotAsked, None);
        let h = r.open_handle();
        assert_eq!(r.call(close, &[h.wrapping_add(1000)]), SCE_LOCATION_ERROR_INVALID_HANDLE);
        assert_eq!(r.call(close, &[h]), 0);
        assert_eq!(r.call(close, &[h]), SCE_LOCATION_ERROR_INVALID_HANDLE, "already closed");
    }

    /// The provider is released when the LAST handle closes, not the first - two handles
    /// are two views of one device, and stopping acquisition while one is still open
    /// would silently starve it.
    #[test]
    fn the_provider_is_released_only_when_the_last_handle_closes() {
        let mut r = Run::new(LocationPermission::Granted, Some(full_fix()));
        let a = r.open_handle();
        let b = r.open_handle();
        assert_eq!(r.call(close, &[a]), 0);
        assert_eq!(r.releases(), 0, "one handle is still open");
        assert_eq!(r.call(close, &[b]), 0);
        assert_eq!(r.releases(), 1);
    }

    #[test]
    fn term_drops_every_handle_and_releases_the_provider() {
        let mut r = Run::new(LocationPermission::Granted, Some(full_fix()));
        let a = r.open_handle();
        let _b = r.open_handle();
        assert_eq!(r.call(term, &[]), 0);
        assert_eq!(r.releases(), 1);
        assert_eq!(r.call(close, &[a]), SCE_LOCATION_ERROR_INVALID_HANDLE);
    }

    // -- the confirm dialog -----------------------------------------------------

    /// Before any Confirm the dialog is IDLE, and asking for a result reports that none
    /// is stored - as an error AND as the value, so both kinds of caller agree.
    #[test]
    fn the_dialog_is_idle_until_confirm_and_has_no_result() {
        let mut r = Run::new(LocationPermission::NotAsked, None);
        let h = r.open_handle();
        assert_eq!(r.call(confirm_get_status, &[h, 0x200]), 0);
        assert_eq!(r.word(0x200), DIALOG_STATUS_IDLE);
        assert_eq!(r.call(confirm_get_result, &[h, 0x204]), SCE_LOCATION_ERROR_DIALOG_RESULT_NONE);
        assert_eq!(r.word(0x204), DIALOG_RESULT_NONE);
    }

    /// Confirm is what asks the host to start acquiring - which is what raises the
    /// platform's own permission prompt.
    #[test]
    fn confirm_asks_the_host_to_start_acquiring() {
        let mut r = Run::new(LocationPermission::NotAsked, None);
        let h = r.open_handle();
        assert_eq!(r.requests(), 0, "opening must not raise a prompt on its own");
        assert_eq!(r.call(confirm, &[h]), 0);
        assert_eq!(r.requests(), 1);
    }

    /// While the platform prompt is up the guest's dialog reads RUNNING, and there is
    /// still no result.
    #[test]
    fn a_pending_permission_reads_as_a_running_dialog() {
        let mut r = Run::new(LocationPermission::Pending, None);
        let h = r.open_handle();
        assert_eq!(r.call(confirm, &[h]), 0);
        assert_eq!(r.call(confirm_get_status, &[h, 0x200]), 0);
        assert_eq!(r.word(0x200), DIALOG_STATUS_RUNNING);
        assert_eq!(r.call(confirm_get_result, &[h, 0x204]), SCE_LOCATION_ERROR_DIALOG_RESULT_NONE);
    }

    /// A grant finishes the dialog with ENABLE; a refusal finishes it with DISABLE.
    /// These two answers are what the whole family exists to deliver.
    #[test]
    fn a_granted_permission_finishes_the_dialog_with_enable() {
        let mut r = Run::new(LocationPermission::Granted, Some(full_fix()));
        let h = r.open_handle();
        assert_eq!(r.call(confirm, &[h]), 0);
        assert_eq!(r.call(confirm_get_status, &[h, 0x200]), 0);
        assert_eq!(r.word(0x200), DIALOG_STATUS_FINISHED);
        assert_eq!(r.call(confirm_get_result, &[h, 0x204]), 0);
        assert_eq!(r.word(0x204), DIALOG_RESULT_ENABLE);
    }

    #[test]
    fn a_denied_permission_finishes_the_dialog_with_disable() {
        let mut r = Run::new(LocationPermission::Denied, None);
        let h = r.open_handle();
        assert_eq!(r.call(confirm, &[h]), 0);
        assert_eq!(r.call(confirm_get_status, &[h, 0x200]), 0);
        assert_eq!(r.word(0x200), DIALOG_STATUS_FINISHED);
        assert_eq!(r.call(confirm_get_result, &[h, 0x204]), 0);
        assert_eq!(r.word(0x204), DIALOG_RESULT_DISABLE);
    }

    /// A second Confirm while the user has not answered is MULTIPLE_CONFIRM; once they
    /// HAVE answered it is not "multiple" and must succeed.
    #[test]
    fn a_second_confirm_is_rejected_only_while_one_is_pending() {
        let mut r = Run::new(LocationPermission::Pending, None);
        let h = r.open_handle();
        assert_eq!(r.call(confirm, &[h]), 0);
        assert_eq!(r.call(confirm, &[h]), SCE_LOCATION_ERROR_MULTIPLE_CONFIRM);

        let mut g = Run::new(LocationPermission::Granted, Some(full_fix()));
        let h = g.open_handle();
        assert_eq!(g.call(confirm, &[h]), 0);
        assert_eq!(g.call(confirm, &[h]), 0, "the dialog is finished, not multiple");
    }

    /// Aborting returns the dialog to IDLE without inventing an answer.
    #[test]
    fn aborting_the_dialog_returns_it_to_idle() {
        let mut r = Run::new(LocationPermission::Pending, None);
        let h = r.open_handle();
        assert_eq!(r.call(confirm, &[h]), 0);
        assert_eq!(r.call(confirm_abort, &[h]), 0);
        assert_eq!(r.call(confirm_get_status, &[h, 0x200]), 0);
        assert_eq!(r.word(0x200), DIALOG_STATUS_IDLE);
    }

    // -- get_location -----------------------------------------------------------

    /// The struct layout, field by field, at the offsets `psp2/location.h` fixes. This
    /// is the test that catches a mis-sized or mis-ordered field, which is the failure a
    /// title cannot report - it simply drives to the wrong place.
    #[test]
    fn a_fix_lands_at_every_declared_offset_and_width() {
        let f = full_fix();
        let mut r = Run::new(LocationPermission::Granted, Some(f));
        let h = r.open_handle();
        assert_eq!(r.call(get_location, &[h, 0x200]), 0);
        assert_eq!(r.f64_at(0x200), f.latitude_deg);
        assert_eq!(r.f64_at(0x208), f.longitude_deg);
        assert_eq!(r.f64_at(0x210), f.altitude_m.unwrap());
        assert_eq!(r.f32_at(0x218), f.accuracy_m.unwrap());
        assert_eq!(r.f32_at(0x21C), 0.0, "reserve is not a failed measurement");
        assert_eq!(r.f32_at(0x220), f.direction_deg.unwrap());
        assert_eq!(r.f32_at(0x224), f.speed_mps.unwrap());
        assert_eq!(r.u64_at(0x228), f.timestamp_us);
    }

    /// An absent optional field gets the API's INVALID sentinel at its OWN width - not a
    /// zero, which is a real latitude, a real speed and a real heading.
    #[test]
    fn an_absent_field_is_the_invalid_sentinel_not_zero() {
        let f = sparse_fix();
        let mut r = Run::new(LocationPermission::Granted, Some(f));
        let h = r.open_handle();
        assert_eq!(r.call(get_location, &[h, 0x200]), 0);
        assert_eq!(r.f64_at(0x200), f.latitude_deg);
        assert_eq!(r.f64_at(0x210), SCE_LOCATION_DATA_INVALID);
        assert_eq!(r.f32_at(0x218), SCE_LOCATION_DATA_INVALID as f32);
        assert_eq!(r.f32_at(0x220), SCE_LOCATION_DATA_INVALID as f32);
        assert_eq!(r.f32_at(0x224), SCE_LOCATION_DATA_INVALID as f32);
    }

    /// No fix yet is UNDETERMINED, and the struct is written all-INVALID rather than
    /// left holding whatever was on the caller's stack.
    #[test]
    fn no_fix_yet_is_undetermined_and_writes_an_all_invalid_struct() {
        let mut r = Run::new(LocationPermission::Granted, None);
        let h = r.open_handle();
        // Poison the buffer first, so "left alone" would be visible.
        for b in &mut r.bytes[0x200..0x230] {
            *b = 0xAB;
        }
        assert_eq!(r.call(get_location, &[h, 0x200]), SCE_LOCATION_INFO_UNDETERMINED_LOCATION);
        assert_eq!(r.f64_at(0x200), SCE_LOCATION_DATA_INVALID);
        assert_eq!(r.f64_at(0x208), SCE_LOCATION_DATA_INVALID);
        assert_eq!(r.f64_at(0x210), SCE_LOCATION_DATA_INVALID);
        assert_eq!(r.u64_at(0x228), 0);
    }

    #[test]
    fn a_refusal_reports_denied_by_user_and_leaks_no_position() {
        let mut r = Run::new(LocationPermission::Denied, Some(full_fix()));
        let h = r.open_handle();
        assert_eq!(r.call(get_location, &[h, 0x200]), SCE_LOCATION_INFO_DENIED_BY_USER);
        assert_eq!(r.f64_at(0x200), SCE_LOCATION_DATA_INVALID);
    }

    /// The timeout variant is the same call, so it must give the same answer - a
    /// separate NID that silently behaved differently would be the worst kind of bug.
    #[test]
    fn the_timeout_variant_matches_the_plain_one() {
        let mut r = Run::new(LocationPermission::Granted, Some(full_fix()));
        let h = r.open_handle();
        assert_eq!(r.call(get_location, &[h, 0x200]), 0);
        assert_eq!(r.call(get_location_with_timeout, &[h, 0x240, 5000]), 0);
        assert_eq!(r.bytes[0x200..0x230], r.bytes[0x240..0x270]);
    }

    // -- the rest of the surface ------------------------------------------------

    #[test]
    fn get_method_reports_back_what_open_was_given() {
        let mut r = Run::new(LocationPermission::NotAsked, None);
        let h = r.open_handle();
        assert_eq!(r.call(get_method, &[h, 0x200, 0x204]), 0);
        assert_eq!(r.word(0x200), 1);
        assert_eq!(r.word(0x204), 0);
    }

    /// Either out-parameter may be null: a caller wanting only one method is asking a
    /// legal question.
    #[test]
    fn get_method_accepts_a_null_out_parameter() {
        let mut r = Run::new(LocationPermission::NotAsked, None);
        let h = r.open_handle();
        assert_eq!(r.call(get_method, &[h, 0, 0x204]), 0);
        assert_eq!(r.word(0x204), 0);
    }

    #[test]
    fn reopen_replaces_the_methods_and_still_validates_them() {
        let mut r = Run::new(LocationPermission::NotAsked, None);
        let h = r.open_handle();
        assert_eq!(r.call(reopen, &[h, 5, 3]), 0);
        assert_eq!(r.call(get_method, &[h, 0x200, 0x204]), 0);
        assert_eq!(r.word(0x200), 5);
        assert_eq!(r.word(0x204), 3);
        assert_eq!(r.call(reopen, &[h, 99, 0]), SCE_LOCATION_ERROR_INVALID_LOCATION_METHOD);
    }

    /// A heading is a compass reading and this emulator has no compass, so the answer is
    /// the API's "could not determine" - never a heading derived from travel, which is a
    /// different quantity.
    #[test]
    fn get_heading_reports_undetermined_with_no_compass() {
        let mut r = Run::new(LocationPermission::Granted, Some(full_fix()));
        let h = r.open_handle();
        assert_eq!(r.call(get_heading, &[h, 0x200]), SCE_LOCATION_INFO_UNDETERMINED_LOCATION);
        for off in [0x200u32, 0x204, 0x208, 0x20C, 0x210, 0x214] {
            assert_eq!(r.f32_at(off), SCE_LOCATION_DATA_INVALID as f32);
        }
    }

    /// The per-application status carries the real answer; the parental and system-wide
    /// gates are the ones a host cannot speak for.
    #[test]
    fn get_permission_reports_the_application_status_that_varies() {
        for (perm, want_app) in [
            (LocationPermission::NotAsked, PERMISSION_APPLICATION_INIT),
            (LocationPermission::Granted, PERMISSION_APPLICATION_ALLOW),
            (LocationPermission::Denied, PERMISSION_APPLICATION_DENY),
        ] {
            let mut r = Run::new(perm, None);
            let h = r.open_handle();
            assert_eq!(r.call(get_permission, &[h, 0x200]), 0);
            assert_eq!(r.word(0x200), PERMISSION_ALLOW, "parental");
            assert_eq!(r.word(0x204), PERMISSION_ALLOW, "system-wide");
            assert_eq!(r.word(0x208), want_app, "{perm:?}");
        }
    }

    #[test]
    fn deny_application_releases_the_provider() {
        let mut r = Run::new(LocationPermission::Granted, Some(full_fix()));
        let h = r.open_handle();
        assert_eq!(r.call(confirm, &[h]), 0);
        assert_eq!(r.call(deny_application, &[]), 0);
        assert_eq!(r.releases(), 1);
        assert_eq!(r.call(confirm_get_status, &[h, 0x200]), 0);
        assert_eq!(r.word(0x200), DIALOG_STATUS_IDLE);
    }

    /// Every out-parameter-taking call rejects a null pointer rather than writing
    /// through it. Checked in one place so a new entry point cannot quietly skip it.
    #[test]
    fn every_out_parameter_call_rejects_a_null_pointer() {
        let mut r = Run::new(LocationPermission::Granted, Some(full_fix()));
        let h = r.open_handle();
        for f in [
            confirm_get_status as Entry,
            confirm_get_result,
            get_location,
            get_heading,
            get_permission,
        ] {
            assert_eq!(r.call(f, &[h, 0]), SCE_LOCATION_ERROR_INVALID_ADDRESS);
        }
    }

    /// Every handle-taking call rejects a handle it never issued.
    #[test]
    fn every_handle_call_rejects_an_unissued_handle() {
        let mut r = Run::new(LocationPermission::Granted, Some(full_fix()));
        let h = r.open_handle();
        let bad = h + 4242;
        for f in [close as Entry, confirm, confirm_abort, cancel_get_location] {
            assert_eq!(r.call(f, &[bad]), SCE_LOCATION_ERROR_INVALID_HANDLE);
        }
        for f in [
            confirm_get_status as Entry,
            confirm_get_result,
            get_location,
            get_heading,
            get_permission,
        ] {
            assert_eq!(r.call(f, &[bad, 0x200]), SCE_LOCATION_ERROR_INVALID_HANDLE);
        }
    }
}
