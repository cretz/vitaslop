//! SceSystemGesture: gesture recognition layered over the touch panels.
//!
//! **NOTHING about this library's interface is published.** The vitasdk NID database
//! names the functions and stops there: there is no `psp2/systemgesture.h` in
//! vita-headers, no page on psdevwiki or the henkaku wiki, and the only other sources
//! that carry a layout are GPL emulators this project may not read. So every argument
//! shape below is read from the CALLING TITLE'S OWN CODE - the instruction stream that
//! sets up the call, plus the `call:` argument dump the unimplemented-NID hard-fail
//! prints ([`super::describe_call_args`]). Each handler documents the evidence it rests
//! on, and a call whose shape has not been observed yet is deliberately left to that
//! hard-fail rather than guessed at.
//!
//! WHAT IS ESTABLISHED and what is not. The call shapes are (each documented where it is
//! used): the recognizer work area, the gesture TYPE, the touch PORT, the watched
//! RECTANGLE, and - the one field of `SceSystemGestureTouchEvent` this title reads - a
//! POSITION at byte offsets 26 and 28. What is NOT established is what each type VALUE
//! means: the title creates recognizers of types 1, 2, 4 and 8, which is tap / drag /
//! hold / pinch shaped, but nothing here says which is which.
//!
//! So a recognizer reports "a point is inside my rectangle, on my panel" and nothing
//! more, for every type. A title that polls all of its recognizers each frame and
//! dispatches on which one fired is therefore being told that a tap AND a drag AND a
//! pinch all happened at once. That is a real approximation, not a faithful model, and
//! [`note_position_only_event`] says so unconditionally the first time an event is
//! reported. `VITASLOP_GESTURE_TYPE_MASK` narrows which types report, which is how the
//! type mapping can be settled by experiment.

use crate::host::{GuestCtx, VitaState};
use crate::hostcall;

/// Print `msg` once per distinct key, unconditionally - never behind a debug flag. An
/// approximation the screen cannot distinguish from a faithful result has to announce
/// itself or it becomes a wrong claim about what the emulator does.
fn report_once(key: &'static str, msg: &str) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::BTreeSet<&'static str>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::BTreeSet::new()));
    if seen.lock().unwrap().insert(key) {
        eprintln!("{msg}");
    }
}

/// Byte offset of `reportNum` within a `SceTouchData` - the count of points currently
/// down. Same layout [`crate::vita::touch`] writes, which is where these samples come
/// from: the title reads them with `sceTouchRead` and hands them straight to this
/// library.
const TOUCH_DATA_REPORT_NUM_OFF: u32 = 12;

/// Report, once, what this library still approximates once it IS producing events.
///
/// A real recognizer classifies a gesture (tap / drag / hold / pinch) and reports its
/// phase; ours reports a POSITION and nothing else, because the position is the only
/// thing the calling title reads back (see [`get_touch_event_by_index`]). A title that
/// reads the type or phase would get a zeroed field, which is why this says so out loud
/// rather than letting the screen imply the gesture layer is complete.
fn note_position_only_event() {
    report_once(
        "gesture-position-only",
        "sceSystemGesture: reporting POSITION-ONLY gesture events. The gesture TYPE and \
         PHASE fields of SceSystemGestureTouchEvent are left zeroed - the calling title \
         reads only the two coordinates, so those fields' offsets are not established and \
         are not invented. A title that classifies gestures will not behave correctly.",
    );
}

/// `int sceSystemGestureInitializePrimitiveTouchRecognizer(void *param)`
///
/// EVIDENCE for the one-argument shape: the calling title sets up this call with a bare
/// `MOV r0, #0` immediately before the branch, and nothing else - so it passes exactly
/// one argument and that argument is 0. A null parameter block is the library's
/// "defaults" spelling, and there is no out-parameter to fill.
///
/// Initializing the primitive layer against an idle panel has no observable effect
/// beyond succeeding, so returning success is the whole of it - not a stub standing in
/// for work that was skipped.
#[hostcall]
pub(super) fn init_primitive_touch_recognizer(
    _ctx: &mut GuestCtx,
    _st: &mut VitaState,
    _param: u32,
) -> i32 {
    0
}

/// Marker written at the head of a recognizer's guest work area by
/// [`create_touch_recognizer`], so a later call on that area can tell a recognizer this
/// run created from an uninitialized buffer.
///
/// The identity lives IN the work area rather than in a host table keyed by its address
/// because a guest is free to hand back a COPY of a POD work area, and an address-keyed
/// table cannot recognize one (memory `vitaslop-host-call-reference-semantics`).
const RECOGNIZER_MAGIC: u32 = 0x5347_5231; // "SGR1"

/// The library's own error codes are not published either. `0x80280001` is the standard
/// SCE "invalid argument" spelling for a facility (`0x8028xxxx` is SceSystemGesture's
/// range in the published module-error layout), used here only for a caller error that
/// no correct guest should ever provoke.
const SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT: i32 = 0x8028_0001u32 as i32;

/// A recognizer's screen rectangle: four `i16` as `(x, y, width, height)`.
///
/// EVIDENCE, from the calling title building the argument immediately before the call:
/// it emits four 16-bit stores into one 8-byte object - `x` and `y` copied straight from
/// a pair of signed halfwords, then `x1 - x` and `y1 - y` - and passes that object's
/// address. So the field order, their width, and that the last two are EXTENTS rather
/// than a second corner are all read off the construction, not assumed. The values the
/// title passes are `(0, 0, 1919, 1087)`, exactly the front panel extent
/// [`crate::vita::touch::panel_info_bytes`] reports, which independently confirms the
/// reading and the coordinate space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rect {
    x: i16,
    y: i16,
    width: i16,
    height: i16,
}

impl Rect {
    fn read(ctx: &mut GuestCtx, addr: u32) -> Rect {
        let h = |off: u32| ctx.read_u32(addr + off);
        let lo = h(0);
        let hi = h(4);
        Rect {
            x: lo as u16 as i16,
            y: (lo >> 16) as u16 as i16,
            width: hi as u16 as i16,
            height: (hi >> 16) as u16 as i16,
        }
    }
}

/// `int sceSystemGestureCreateTouchRecognizer(SceSystemGestureTouchRecognizer *recognizer,
///                                           int type, SceUInt32 touchPort,
///                                           const SceSystemGestureRectangle *rect,
///                                           const void *param)`
///
/// EVIDENCE for the shape: the calling title sets exactly `r0`-`r3` before the branch
/// (`r0 = <zeroed work area>`, `r1 = <type>`, `r2 = 0 or 1`, `r3 = <the rectangle it just
/// built>`) and the first stack word is 0.
///
/// **`r2` WAS FIRST READ AS A PARAMETER POINTER AND THAT WAS WRONG.** Logging its value
/// across every call settled it: it is literally `0` or `1`, never an address, and the
/// title creates recognizers in both flavours for the same type. On a console with a
/// FRONT and a BACK touch panel, a 0/1 next to a screen rectangle is the panel - the same
/// `SCE_TOUCH_PORT_FRONT`/`_BACK` selector `sceTouchRead` takes, which this title calls
/// once per port immediately before feeding both samples to the primitive recognizer.
/// Reading it as a pointer would have made every recognizer watch the front panel, so a
/// back-panel gesture would fire on a front-panel touch.
///
/// This title's nine recognizers: four of type 1 (ports 0,0,1,1), two of type 2
/// (ports 1,0), one of type 4 (port 1), two of type 8 (ports 1,0) - all over the full
/// panel. Powers of two, in a family that is tap / drag / hold / pinch shaped. **Which
/// value is which gesture is NOT established**, so no behaviour is keyed off it yet; see
/// [`recognizer_events`].
#[hostcall]
pub(super) fn create_touch_recognizer(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    recognizer: Ptr,
    kind: u32,
    port: u32,
    rect: Ptr,
) -> i32 {
    if recognizer.is_null() {
        // A null work area is the caller's error, not a state to model.
        SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT
    } else {
        let r = (!rect.is_null()).then(|| Rect::read(ctx, rect.addr()));
        // The identity goes in the guest's own work area: magic, type, port, and the
        // rectangle it watches, so a later Update/Get call on this area (or on a copy of
        // it) can recover what it is without a host address table.
        ctx.write_bytes(recognizer.addr(), &RECOGNIZER_MAGIC.to_le_bytes());
        ctx.write_bytes(recognizer.addr() + 4, &kind.to_le_bytes());
        let (x, y, w, h) = r.map_or((0, 0, 0, 0), |r| (r.x, r.y, r.width, r.height));
        ctx.write_bytes(recognizer.addr() + 8, &x.to_le_bytes());
        ctx.write_bytes(recognizer.addr() + 10, &y.to_le_bytes());
        ctx.write_bytes(recognizer.addr() + 12, &w.to_le_bytes());
        ctx.write_bytes(recognizer.addr() + 14, &h.to_le_bytes());
        ctx.write_bytes(recognizer.addr() + 16, &port.to_le_bytes());
        // Log every recognizer as it is created, unconditionally and once each. A
        // recognizer only reports a touch inside its rectangle and on its own panel, so
        // this set is the map of where the title is listening - the first thing worth
        // knowing when a scripted tap "has an effect but does not do anything".
        eprintln!(
            "sceSystemGesture: recognizer #{} type={kind} port={port} rect=({x},{y} {w}x{h}) \
             at {:#010x}",
            RECOGNIZER_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            recognizer.addr(),
        );
        0
    }
}

/// How many recognizers this run has created, so each is logged with a stable index.
static RECOGNIZER_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// `int sceSystemGestureUpdatePrimitiveTouchRecognizer(const SceTouchData *pFront,
///                                                    const SceTouchData *pBack)`
///
/// EVIDENCE for the two-argument shape, read straight off the calling title's per-frame
/// update, which reads both panels and then hands the two samples over:
///
/// ```text
///   r0=0, r1=<front buf>, r2=1   BLX   -> sceTouchRead(SCE_TOUCH_PORT_FRONT, front, 1)
///   r0=1, r1=<back buf>,  r2=1   BLX   -> sceTouchRead(SCE_TOUCH_PORT_BACK,  back,  1)
///   r0=<front buf>, r1=<back buf>   BLX   -> HERE
/// ```
///
/// Only `r0` and `r1` are set for this call - the `1`s still in `r2`/`r3` are the
/// leftover `nBufs` of the two reads above, not arguments. The two buffers are `0x90`
/// bytes apart, exactly one `SceTouchData`, which confirms what they are independently
/// of the calls that filled them.
///
/// The primitive layer's job is to turn the two raw panel samples into primitive touch
/// events. Both samples come from `sceTouchRead`, which reads the SAME world touch frame
/// this module reads directly in [`recognizer_events`], so there is no state to carry
/// between the two calls - the recognizers are answered from the world, not from a copy
/// of it taken here. What this call does contribute is the CHECK: if the guest ever hands
/// over a sample that disagrees with the world, the two paths have diverged and one of
/// them is wrong.
#[hostcall]
pub(super) fn update_primitive_touch_recognizer(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    front: Ptr,
    back: Ptr,
) -> i32 {
    let (fa, ba) = (front.addr(), back.addr());
    let mut reported = 0u32;
    for addr in [fa, ba] {
        if addr != 0 {
            reported += ctx.read_u32(addr + TOUCH_DATA_REPORT_NUM_OFF);
        }
    }
    let world = st.world.poll_touch(0).active().len() + st.world.poll_touch(1).active().len();
    if reported as usize != world {
        report_once(
            "gesture-sample-mismatch",
            "sceSystemGestureUpdatePrimitiveTouchRecognizer: the SceTouchData the title \
             passed reports a different number of points than the world does this frame. \
             The recognizers are answered from the world, so they would disagree with the \
             samples the title thinks it fed in.",
        );
    }
    0
}

/// `int sceSystemGestureUpdateTouchRecognizer(SceSystemGestureTouchRecognizer *recognizer)`
///
/// EVIDENCE for the one-argument shape: the title calls this once per recognizer in a
/// straight run of calls that each set only `r0` to a different offset in its own state
/// block (`r4 + 1312`, `+3768`, `+6224`, ...), the first of which is exactly the work
/// area it passed to [`create_touch_recognizer`]. Nothing else is set up between them.
///
/// The recognizer advances against the primitive layer's events; there were none, so it
/// recognizes nothing. Its work area is left as [`create_touch_recognizer`] wrote it.
#[hostcall]
pub(super) fn update_touch_recognizer(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    recognizer: Ptr,
) -> i32 {
    if recognizer.is_null() || ctx.read_u32(recognizer.addr()) != RECOGNIZER_MAGIC {
        // Updating something this run never created is a guest bug or a wrong reading of
        // the argument shape - either way it must not pass silently.
        report_once(
            "gesture-update-unknown",
            "sceSystemGestureUpdateTouchRecognizer: called on a work area that no \
             sceSystemGestureCreateTouchRecognizer in this run initialized. Either the \
             title creates recognizers by some path not seen yet, or this argument shape \
             is wrong.",
        );
        SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT
    } else {
        0
    }
}

/// Byte offsets, within `SceSystemGestureTouchEvent`, of the two fields the calling
/// title reads. **These are the ONLY two fields of the struct that are established.**
///
/// EVIDENCE, read off the consumer loop that runs immediately after
/// `sceSystemGestureGetTouchEventByIndex` returns:
///
/// ```text
///   r2 = sp                      ; the event buffer is on the caller's stack
///   r0 = recognizer, r1 = index
///   BLX  -> sceSystemGestureGetTouchEventByIndex(recognizer, index, &event)
///   LDRSH r2, [sp, #26]          ; <- the only two loads from the event
///   LDRSH r3, [sp, #28]
///   STRH  r2, [r1, r0]           ; stored as a HALFWORD PAIR into a per-touch array
///   STRH  r3, [r14, #2]
/// ```
///
/// Two adjacent SIGNED halfwords, kept as a pair, is a coordinate; the recognizer's own
/// rectangle is in panel space, so these are panel coordinates. Nothing else in the
/// struct is read here, so nothing else is claimed - the rest is zeroed, and
/// [`note_position_only_event`] says so.
const EVENT_X_OFF: u32 = 26;
const EVENT_Y_OFF: u32 = 28;

/// Bytes of the event struct this writes. The two established fields end at byte 30, so
/// the struct is at least that big and writing 30 bytes cannot run past it - which
/// matters, because the buffer is a stack slot in the CALLER's frame and overrunning it
/// would corrupt the caller's locals.
const EVENT_WRITE_BYTES: u32 = 30;

/// Recognizer types allowed to report events (`VITASLOP_GESTURE_TYPE_MASK`, a bitmask
/// over the type values).
///
/// # Why the default is type 1 alone, and not every type
/// A title polls every recognizer it created each frame and dispatches on which one
/// fired, so reporting a point to ALL of them says a tap AND a drag AND a hold AND a
/// pinch happened at the same instant. That is not a harmless over-report: it is
/// CONTRADICTORY, and a UI that has to choose resolves it the wrong way.
///
/// MEASURED on the retail title this library was RE'd from. Its campaign map is
/// tap-to-select and drag-to-pan. With every type reporting, a 35-position tap sweep
/// across the map selected nothing at all - the map only panned. With the mask
/// narrowed to type 1 the identical sweep started an event (the title spawned its
/// loading thread); masks 2, 4 and 8 each reproduced the all-types outcome exactly,
/// bit for bit. So type 1 is the recognizer whose event a tap-driven UI consumes, and
/// the other three were actively suppressing it.
///
/// It is also the reading that needs no gesture the world cannot support: the world
/// model delivers ONE stationary touch point, which genuinely is a tap and genuinely
/// is NOT a drag (no movement), a hold (no duration model) or a pinch (one point).
/// Reporting only the tap is therefore the faithful answer as well as the working one.
/// Type 1 meaning "tap" is still an inference from behaviour rather than a published
/// fact, so the mask stays overridable and [`note_type_mask`] states it at runtime.
const DEFAULT_GESTURE_TYPE_MASK: u32 = 1;

fn type_mask() -> u32 {
    static CELL: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_GESTURE_TYPE_MASK")
            .ok()
            .and_then(|s| {
                let s = s.trim();
                match s.strip_prefix("0x") {
                    Some(h) => u32::from_str_radix(h, 16).ok(),
                    None => s.parse().ok(),
                }
            })
            .unwrap_or(DEFAULT_GESTURE_TYPE_MASK)
    })
}

/// Say, once, which recognizer types report events and which are silent - because a
/// recognizer that never fires is indistinguishable from one whose gesture the player
/// never made, and a title that relies on a drag would look merely unresponsive.
fn note_type_mask(kind: u32) {
    let mask = type_mask();
    if mask & kind == 0 {
        report_once(
            "gesture-type-suppressed",
            "sceSystemGesture: a recognizer of a type OUTSIDE VITASLOP_GESTURE_TYPE_MASK \
             was polled; it reports no events. The world delivers a single stationary \
             touch point, which is a tap and is not a drag, a hold or a pinch - see \
             `type_mask`. A title needing one of those will not respond.",
        );
    }
}

/// The gesture events a recognizer reports this frame: one per touch point on ITS PANEL
/// and inside its rectangle, as `(x, y)` in panel coordinates.
///
/// Containment plus panel is the whole of the model. What it does NOT do is classify:
/// every type reports the same "there is a point here", so a title that distinguishes a
/// tap from a drag from a pinch is being told all three happened. That is the honest
/// state of the RE - the type values are known, their meanings are not - and
/// [`note_position_only_event`] says so at runtime rather than letting the screen imply
/// otherwise. `VITASLOP_GESTURE_TYPE_MASK` narrows it for an experiment.
fn recognizer_events(ctx: &mut GuestCtx, st: &mut VitaState, recognizer: u32) -> Vec<(i16, i16)> {
    if recognizer == 0 || ctx.read_u32(recognizer) != RECOGNIZER_MAGIC {
        return Vec::new();
    }
    let kind = ctx.read_u32(recognizer + 4);
    if type_mask() & kind == 0 {
        note_type_mask(kind);
        return Vec::new();
    }
    let half = |off: u32| -> i16 {
        let w = ctx.read_u32(recognizer + (off & !3));
        (if off & 2 == 0 { w as u16 } else { (w >> 16) as u16 }) as i16
    };
    let (x, y, w, h) = (half(8), half(10), half(12), half(14));
    let port = ctx.read_u32(recognizer + 16);
    let (x0, y0) = (x as i32, y as i32);
    let (x1, y1) = (x0 + w as i32, y0 + h as i32);
    let mut out = Vec::new();
    let frame = st.world.poll_touch(port);
    let points = frame.active();
    for p in points.iter() {
        let (px, py) = (p.x as i32, p.y as i32);
        if (x0..=x1).contains(&px) && (y0..=y1).contains(&py) {
            out.push((px as i16, py as i16));
        }
    }
    // Diagnostic (`RUST_LOG=vitaslop::input=trace`): the recognizer's rectangle, the
    // points offered to it, and how many landed inside.
    //
    // A tap that is delivered, is inside the right rectangle, and still selects nothing
    // is a completely different bug from a tap that misses the rectangle or never
    // arrives - and the screen is identical in all three cases.
    if !points.is_empty() && tracing::enabled!(target: "vitaslop::input", tracing::Level::TRACE) {
        tracing::trace!(
            target: "vitaslop::input",
            "gesture recognizer kind {kind} rect ({x0},{y0})..({x1},{y1}) port {port}: \
             {} point(s) offered, {} inside",
            points.len(),
            out.len(),
        );
    }
    out
}

/// `SceUInt32 sceSystemGestureGetTouchEventsCount(const SceSystemGestureTouchRecognizer *r)`
///
/// EVIDENCE for the shape AND for the count being the RETURN VALUE rather than an
/// out-parameter: the title passes only the recognizer work area (the argument dump shows
/// `r0` holding the very block [`create_touch_recognizer`] stamped, magic and rectangle
/// intact, while `r1`-`r3` still hold unrelated small integers left over from the
/// preceding code), and the instruction immediately after the branch is `MOV r8, r0` - it
/// keeps the returned value as a number and never tests it as an error code. The title
/// then loops `index` from 0 while `index < count`, which is what makes the count the
/// bound on [`get_touch_event_by_index`].
#[hostcall]
pub(super) fn get_touch_events_count(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    recognizer: Ptr,
) -> i32 {
    if recognizer.is_null() || ctx.read_u32(recognizer.addr()) != RECOGNIZER_MAGIC {
        report_once(
            "gesture-count-unknown",
            "sceSystemGestureGetTouchEventsCount: called on a work area that no \
             sceSystemGestureCreateTouchRecognizer in this run initialized - the argument \
             shape read for one of the two calls is wrong.",
        );
        0
    } else {
        recognizer_events(ctx, st, recognizer.addr()).len() as i32
    }
}

/// `int sceSystemGestureGetTouchEventByIndex(const SceSystemGestureTouchRecognizer *r,
///                                          SceUInt32 index, SceSystemGestureTouchEvent *ev)`
///
/// EVIDENCE for the argument order: the caller sets `r0 = <the recognizer it just counted
/// events on>`, `r1 = <its loop counter>` and `r2 = sp` immediately before the branch, and
/// then reads the event back out of that stack buffer.
///
/// Only [`EVENT_X_OFF`] and [`EVENT_Y_OFF`] are written with meaning; see their evidence.
/// An out-of-range `index` cannot happen while the title honours the count, so it is an
/// error rather than a silently empty event.
#[hostcall]
pub(super) fn get_touch_event_by_index(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    recognizer: Ptr,
    index: u32,
    event: Ptr,
) -> i32 {
    let events = recognizer_events(ctx, st, recognizer.addr());
    match events.get(index as usize) {
        None => SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT,
        Some(&(x, y)) if event.is_null() => {
            let _ = (x, y);
            SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT
        }
        Some(&(x, y)) => {
            note_position_only_event();
            // Zero only as far as the established fields reach - see EVENT_WRITE_BYTES.
            ctx.write_bytes(event.addr(), &[0u8; EVENT_WRITE_BYTES as usize]);
            ctx.write_bytes(event.addr() + EVENT_X_OFF, &x.to_le_bytes());
            ctx.write_bytes(event.addr() + EVENT_Y_OFF, &y.to_le_bytes());
            0
        }
    }
}
