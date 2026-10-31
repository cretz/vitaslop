//! SceSystemGesture: gesture recognition layered over the touch panels.
//!
//! Two layers, exactly as the library's own API splits them:
//!
//! - the PRIMITIVE layer, one per process (no handle on any of its calls). The title feeds
//!   it the two `SceTouchData` samples it just read with `sceTouchRead`, once per frame, via
//!   [`update_primitive_touch_recognizer`]; it turns each touch report into a primitive that
//!   goes BEGIN -> ACTIVE -> END and keeps the table the primitive getters read. It lives in
//!   [`VitaState::gesture`] ([`PrimitiveLayer`]).
//! - the RECOGNIZERS: TAP (1), DRAG (2), TAP_AND_HOLD (4), PINCH_OUT_IN (8), each in a
//!   2456-byte block the CALLER owns. [`update_touch_recognizer`] advances one recognizer
//!   against the current primitive table and rebuilds its event list; every getter only
//!   reads that list, so calling one twice in a frame answers the same events.
//!
//! WHERE THE FACTS COME FROM. The clean-room fact sheet for this library (working-area
//! `scesystemgesture-spec.md`) records the 17 NIDs, every prototype, the enums, the error
//! codes and every struct layout used here, each cross-checked against the calling titles'
//! own machine code (the argument dumps the unimplemented-NID hard-fail printed, and the
//! loads the consumers make out of the returned events - quoted at the handlers below).
//! What NO source gives is the recognizers' BEHAVIOUR: slop distances, tap and hold times,
//! the double-tap window, the numeric value of CANCELLED, the state a tap event carries.
//! Each of those is ONE documented `const` in the "NOT ESTABLISHED" block below, chosen by
//! the fact sheet's recommended model, and each is announced once at runtime through
//! [`report_once`] the first time it shapes an event - so a recognizer that disagrees with
//! the console says which guess it rests on instead of passing as faithful.

use crate::host::{GuestCtx, Ptr, VitaState};
use crate::hostcall;

/// Print `msg` once per distinct key, unconditionally - never behind a debug flag. An
/// approximation the screen cannot distinguish from a faithful result has to announce
/// itself or it becomes a wrong claim about what the emulator does.
fn report_once(key: &'static str, msg: &str) {
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<std::collections::BTreeSet<&'static str>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::BTreeSet::new()));
    if seen.lock().unwrap().insert(key) {
        // A warning, on the tracing channel like every other approximation report: it
        // reaches the browser panel and the desktop's stderr, and it is a fix owed.
        tracing::warn!(target: "vitaslop::input", "{msg}");
    }
}

// ---------------------------------------------------------------------------------------
// Error codes, gesture types, touch states - the library's own numbers.
// ---------------------------------------------------------------------------------------

/// The library's error facility is `0x8089xxxx` (public SCE error-code table, corroborated
/// independently). An earlier reading used `0x80280001`, a guess from the generic module
/// layout, which no title compares against.
const SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT: i32 = 0x8089_0001u32 as i32;
const SCE_SYSTEM_GESTURE_ERROR_NOT_SUPPORTED_GESTURE: i32 = 0x8089_0002u32 as i32;
const SCE_SYSTEM_GESTURE_ERROR_NOT_INITIALIZED: i32 = 0x8089_0003u32 as i32;
const SCE_SYSTEM_GESTURE_ERROR_INDEX_OUT_OF_ARRAY: i32 = 0x8089_0004u32 as i32;
const SCE_SYSTEM_GESTURE_ERROR_EVENT_DATA_NOT_FOUND: i32 = 0x8089_0005u32 as i32;

/// `SceSystemGestureType`, bit values. Consistent with the calling titles: one creates four
/// type-1, two type-2, one type-4 and two type-8 recognizers (tap, drag, hold and pinch on
/// both panels), and a second ANDs the event's `gestureType` word with 1 or 2 before it
/// accepts an event.
const TYPE_TAP: i32 = 1;
const TYPE_DRAG: i32 = 2;
const TYPE_TAP_AND_HOLD: i32 = 4;
const TYPE_PINCH_OUT_IN: i32 = 8;
/// `SCE_SYSTEM_GESTURE_TYPE_ROTATION`. A recognizer of this type is CREATED and polled, but
/// its detection is not modelled: it reports no events (see [`create_touch_recognizer`]).
const TYPE_ROTATION: i32 = 16;

/// `SceSystemGestureTouchState`, shared by primitive and gesture events.
const STATE_BEGIN: i32 = 1;
const STATE_ACTIVE: i32 = 2;
const STATE_END: i32 = 3;

// ---------------------------------------------------------------------------------------
// NOT ESTABLISHED: the recognizer model's numbers. Every one of these is the fact sheet's
// recommended model, NOT a measured or published value; each is reported once at runtime
// (see the `note_*` calls) the first time it decides an event.
// ---------------------------------------------------------------------------------------

/// **NOT ESTABLISHED.** The numeric value of `SCE_SYSTEM_GESTURE_TOUCH_STATE_CANCELLED`.
/// The NAME is real (titles compare against it, after END in their switch order) but no
/// declaration gives its number; 4 follows END. Written only for a gesture whose primitive
/// vanished WITHOUT an END - i.e. after `sceSystemGestureResetPrimitiveTouchRecognizer`
/// dropped it mid-gesture - so a title that never resets the primitive layer never sees it.
const STATE_CANCELLED: i32 = 4;

/// **NOT ESTABLISHED.** How far (panel units, Euclidean) a finger may wander from its press
/// point and still count as a TAP, or as a HOLD that has not yet fired. "A few tens of
/// panel units" is all the model says; 40 units is 20 screen pixels on the 1920-wide front
/// panel.
const STATIONARY_SLOP: i32 = 40;

/// **NOT ESTABLISHED.** The longest press that is still a TAP, microseconds. It must stay
/// below the default hold time so one press is never both. A scripted tap is counted in
/// PRESENTS, not 1/60 s: a 20-frame hold on a 30 fps screen is ~670 ms and is not a tap here
/// (a racer's pre-race checkmark ignored it); playbooks tap for ~6 frames, as a finger does.
const TAP_MAX_DURATION_US: u64 = 500_000;

/// **NOT ESTABLISHED.** The `eventState` a TAP event carries. A tap is recognized on
/// RELEASE, so END; every observed consumer only requires that it is not CANCELLED, and
/// some read no state at all.
const TAP_EVENT_STATE: i32 = STATE_END;

/// **NOT ESTABLISHED.** Whether `property.tap.position` is the RELEASE point (true) or the
/// PRESS point (false). Within [`STATIONARY_SLOP`] of each other by construction.
const TAP_POSITION_AT_RELEASE: bool = true;

/// **NOT ESTABLISHED.** The multi-tap window for a recognizer created with
/// `tap.maxTapCount > 1`: a tap whose PRESS comes within this long of the previous tap's
/// RELEASE, and within [`MULTI_TAP_SLOP`] of it, continues the count (1, then 2, ...).
const MULTI_TAP_INTERVAL_US: u64 = 300_000;
/// **NOT ESTABLISHED.** The distance (panel units) half of the multi-tap window.
const MULTI_TAP_SLOP: i32 = 100;

/// How far a primitive must move from its press point before a DRAG begins: NOT AT ALL - a
/// drag begins with the press itself. MEASURED on a shooter whose front end creates exactly
/// ONE recognizer, a DRAG over the whole front panel, and whose buttons are pressed by
/// tapping them on the console: a stationary tap has to produce that recognizer's events, or
/// its menus could never be used. Begun only on movement, every tap on them was ignored.
/// A negative value means no slop test at all, so a primitive that has not moved begins one.
const DRAG_SLOP: i32 = -1;

/// [`DRAG_SLOP`], or `VITASLOP_GESTURE_DRAG_SLOP=<panel units>` to begin a drag only once the
/// primitive has moved that far - the A/B for a title whose taps stop landing while a drag
/// recognizer shares their area.
fn drag_slop() -> i32 {
    static V: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        crate::knobs::var("VITASLOP_GESTURE_DRAG_SLOP")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(DRAG_SLOP)
    })
}

/// **NOT ESTABLISHED.** The hold time when the recognizer was created without a parameter
/// (or with `timeToInvokeEvent = 0`): "on the order of 0.5 s".
const HOLD_DEFAULT_US: u64 = 500_000;

/// **NOT ESTABLISHED.** The UNIT of `tapAndHold.timeToInvokeEvent`. The one observed value
/// is 30, implausible as microseconds; read as 60 Hz panel periods it is exactly the 0.5 s
/// default. So a value BELOW [`HOLD_TIME_TICKS_BELOW`] counts 60 Hz panel periods of
/// [`HOLD_TIME_TICK_US`] each, and a larger one is taken as microseconds like every other
/// time in this API. One choice, two numbers.
const HOLD_TIME_TICKS_BELOW: u64 = 1000;
const HOLD_TIME_TICK_US: u64 = 16_667;

/// **NOT ESTABLISHED.** Whether a rectangle's far edge is inside: `x <= px <= x + width`.
/// Inclusive, so the `(0, 0, 1919, 1087)` rectangle a title passes for the front panel
/// covers the whole 0..=1919 x 0..=1087 panel rather than missing its last row and column.
const RECT_FAR_EDGE_INCLUSIVE: bool = true;

/// **NOT ESTABLISHED.** What a NULL `SceTouchData` pointer to
/// `sceSystemGestureUpdatePrimitiveTouchRecognizer` means for that panel: "no data", read
/// as an EMPTY sample, so a finger the panel was tracking ENDs rather than staying down
/// forever.
const NULL_SAMPLE_IS_EMPTY_PANEL: bool = true;

/// **NOT ESTABLISHED.** Whether the calls that need the primitive layer (its Update, Reset,
/// Finalize and getters, and `sceSystemGestureUpdateTouchRecognizer`, which reads its
/// table) refuse with NOT_INITIALIZED before `sceSystemGestureInitializePrimitiveTouch
/// Recognizer` / after Finalize - inferred from the error's name alone. Calls that only
/// touch a caller-owned recognizer block (Create, Reset, Rectangle, the event getters) do
/// not depend on the global layer and never refuse for it.
const REQUIRE_INITIALIZE: bool = true;

fn note_model(key: &'static str, what: &str) {
    report_once(
        key,
        &format!(
            "sceSystemGesture: {what} - a NOT ESTABLISHED model value (no published source \
             gives it; see the const in vita::gesture). A title tuned to the console's \
             real value may react differently."
        ),
    );
}

// ---------------------------------------------------------------------------------------
// The primitive layer.
// ---------------------------------------------------------------------------------------

/// Layout of `SceTouchData` as [`super::touch`] writes it and the title hands it over:
/// `timeStamp` u64 at 0, `reportNum` u32 at 12, then `report[8]` of 16 bytes from 16, each
/// `id` u8, `force` u8, `x` u16, `y` u16.
const TOUCH_DATA_SIZE: usize = 0x90;
const TOUCH_DATA_REPORT_NUM_OFF: usize = 12;
const TOUCH_REPORT_BASE: usize = 16;
const TOUCH_REPORT_SIZE: usize = 16;
const TOUCH_REPORT_MAX: usize = 8;

/// One report out of a `SceTouchData`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Report {
    id: u8,
    force: u8,
    x: i16,
    y: i16,
}

/// One panel's sample as the title passed it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Sample {
    ts: u64,
    reports: Vec<Report>,
}

fn read_sample(ctx: &GuestCtx, addr: u32) -> Sample {
    let b = ctx.read_bytes(addr, TOUCH_DATA_SIZE);
    if b.len() < TOUCH_DATA_SIZE {
        return Sample::default();
    }
    let ts = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let n = (get_u32(&b, TOUCH_DATA_REPORT_NUM_OFF) as usize).min(TOUCH_REPORT_MAX);
    let reports = (0..n)
        .map(|j| {
            let o = TOUCH_REPORT_BASE + j * TOUCH_REPORT_SIZE;
            Report {
                id: b[o],
                force: b[o + 1],
                x: get_u16(&b, o + 2) as i16,
                y: get_u16(&b, o + 4) as i16,
            }
        })
        .collect();
    Sample { ts, reports }
}

/// One tracked primitive - one `SceSystemGesturePrimitiveTouchEvent` (0x60 bytes): state at
/// 0x00, primitiveID 0x04, pressed position+force 0x06/0x0A, current 0x0C/0x10, delta
/// 0x12/0x16, deltaTime 0x18, elapsedTime 0x20, 56 reserved. A retail consumer's two reads
/// of a primitive event, `+0x0a` and `+0x10`, land on the two force fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Primitive {
    state: i32,
    id: u16,
    port: u8,
    /// The panel's own report id, which is what ties a report to the primitive it continues.
    report_id: u8,
    pressed: (i16, i16, i16),
    current: (i16, i16, i16),
    delta: (i16, i16, i16),
    delta_time: u64,
    elapsed: u64,
    press_us: u64,
}

/// The process-wide primitive layer (`VitaState::gesture`).
#[derive(Debug, Default)]
pub struct PrimitiveLayer {
    initialized: bool,
    /// This update's primitives, oldest first: BEGIN and ACTIVE ones, and the ENDs of
    /// fingers that lifted since the previous update (kept for exactly this update).
    table: Vec<Primitive>,
    /// Each panel's sample timestamp at the previous update, for `deltaTime`.
    port_last_us: [Option<u64>; 2],
    /// The time of the latest update - the `updatedTime` of what a recognizer derives.
    updated_us: u64,
    /// Bumped on every update. A recognizer records the one it consumed, so a second
    /// `UpdateTouchRecognizer` against the SAME table (no primitive update between) does
    /// not re-derive it - which would report one released finger as a second tap.
    seq: u32,
}

/// A primitive's id IS the panel report's own id (`SceTouchReport.id`), not a counter of
/// this layer's. MEASURED on a shooter's touch-driven front end: its touch layer records the
/// raw report id of the finger that pressed a widget, and the widget then accepts only the
/// gesture events whose `primitiveID` equals that byte. A private counter (1 for the first
/// finger, against a report id of 0) failed that comparison on every event, and no button
/// could be pressed. The panels' ids already come from one counter shared by both ports (see
/// `touch::panel_ids`), so front and back cannot collide in the one table.
fn primitive_id(report_id: u8) -> u16 {
    u16::from(report_id)
}

impl PrimitiveLayer {
    /// Advance against one sample per panel (`None` = no sample for that panel, see
    /// [`NULL_SAMPLE_IS_EMPTY_PANEL`]). A report whose id was not tracked is a new primitive
    /// in BEGIN; a tracked one is ACTIVE with its movement since; a tracked one absent from
    /// the sample is an END at its last position, kept for this update only.
    fn advance(&mut self, samples: [Option<Sample>; 2], fallback_now: u64) {
        let now = samples
            .iter()
            .flatten()
            .map(|s| s.ts)
            .max()
            .unwrap_or(fallback_now);
        let prev: Vec<Primitive> = std::mem::take(&mut self.table)
            .into_iter()
            .filter(|p| p.state != STATE_END)
            .collect();
        self.table = prev.clone();
        let mut next = Vec::new();
        for (port, sample) in samples.into_iter().enumerate() {
            let sample = match sample {
                Some(s) => s,
                None if NULL_SAMPLE_IS_EMPTY_PANEL => Sample { ts: now, reports: Vec::new() },
                None => {
                    next.extend(prev.iter().filter(|p| p.port as usize == port).copied());
                    continue;
                }
            };
            let ts = sample.ts;
            let dt = self.port_last_us[port].map_or(0, |last| ts.saturating_sub(last));
            self.port_last_us[port] = Some(ts);
            for q in prev.iter().filter(|q| q.port as usize == port) {
                match sample.reports.iter().find(|r| r.id == q.report_id) {
                    Some(r) => {
                        let cur = (r.x, r.y, i16::from(r.force));
                        next.push(Primitive {
                            state: STATE_ACTIVE,
                            current: cur,
                            delta: (
                                cur.0.wrapping_sub(q.current.0),
                                cur.1.wrapping_sub(q.current.1),
                                cur.2.wrapping_sub(q.current.2),
                            ),
                            delta_time: dt,
                            elapsed: ts.saturating_sub(q.press_us),
                            ..*q
                        });
                    }
                    None => next.push(Primitive {
                        state: STATE_END,
                        delta: (0, 0, 0),
                        delta_time: dt,
                        elapsed: ts.saturating_sub(q.press_us),
                        ..*q
                    }),
                }
            }
            for r in &sample.reports {
                if prev.iter().any(|q| q.port as usize == port && q.report_id == r.id) {
                    continue;
                }
                let id = primitive_id(r.id);
                let cur = (r.x, r.y, i16::from(r.force));
                let p = Primitive {
                    state: STATE_BEGIN,
                    id,
                    port: port as u8,
                    report_id: r.id,
                    pressed: cur,
                    current: cur,
                    delta: (0, 0, 0),
                    delta_time: 0,
                    elapsed: 0,
                    press_us: ts,
                };
                self.table.push(p);
                next.push(p);
            }
        }
        // Oldest first: tracked primitives keep their order, new ones follow.
        next.sort_by_key(|p| (p.state == STATE_BEGIN, p.press_us));
        self.table = next;
        self.updated_us = now;
        self.seq = self.seq.wrapping_add(1).max(1);
    }
}

const PRIMITIVE_EVENT_SIZE: u32 = 0x60;

fn write_primitive(ctx: &mut GuestCtx, at: u32, p: &Primitive) {
    let mut b = [0u8; PRIMITIVE_EVENT_SIZE as usize];
    b[0..4].copy_from_slice(&p.state.to_le_bytes());
    b[4..6].copy_from_slice(&p.id.to_le_bytes());
    for (off, (x, y, f)) in [(0x06, p.pressed), (0x0C, p.current), (0x12, p.delta)] {
        b[off..off + 2].copy_from_slice(&x.to_le_bytes());
        b[off + 2..off + 4].copy_from_slice(&y.to_le_bytes());
        b[off + 4..off + 6].copy_from_slice(&f.to_le_bytes());
    }
    b[0x18..0x20].copy_from_slice(&p.delta_time.to_le_bytes());
    b[0x20..0x28].copy_from_slice(&p.elapsed.to_le_bytes());
    ctx.write_bytes(at, &b);
}

/// The NOT_INITIALIZED refusal ([`REQUIRE_INITIALIZE`]) for a call that needs the layer.
fn require_init(st: &VitaState, call: &'static str) -> Result<(), i32> {
    if !REQUIRE_INITIALIZE || st.gesture.initialized {
        return Ok(());
    }
    report_once(
        call,
        &format!(
            "{call}: refused with SCE_SYSTEM_GESTURE_ERROR_NOT_INITIALIZED - the primitive \
             layer is not initialized (no sceSystemGestureInitializePrimitiveTouchRecognizer, \
             or it was finalized). Whether the console refuses here is inferred from the \
             error's name only (REQUIRE_INITIALIZE)."
        ),
    );
    Err(SCE_SYSTEM_GESTURE_ERROR_NOT_INITIALIZED)
}

/// `int sceSystemGestureInitializePrimitiveTouchRecognizer(
///        SceSystemGesturePrimitiveTouchRecognizerParameter *parameter)`
///
/// EVIDENCE for the one-argument shape: the calling title sets up this call with a bare
/// `MOV r0, #0` immediately before the branch - one argument, NULL, the defaults. The 64-byte
/// parameter is all reserved bytes, so there is nothing in it to read.
#[hostcall]
pub(super) fn init_primitive_touch_recognizer(
    _ctx: &mut GuestCtx,
    st: &mut VitaState,
    _param: u32,
) -> i32 {
    st.gesture.initialized = true;
    st.gesture.table.clear();
    st.gesture.port_last_us = [None, None];
    0
}

/// `int sceSystemGestureFinalizePrimitiveTouchRecognizer(void)`: tear the layer down.
#[hostcall]
pub(super) fn finalize_primitive_touch_recognizer(_ctx: &mut GuestCtx, st: &mut VitaState) -> i32 {
    finalize_primitive_touch_recognizer_impl(st)
}

// The `_impl` split: a `#[hostcall]` body is inlined into the generated wrapper, so an early
// `return` there would return from the wrapper rather than produce r0.
fn finalize_primitive_touch_recognizer_impl(st: &mut VitaState) -> i32 {
    if let Err(e) = require_init(st, "sceSystemGestureFinalizePrimitiveTouchRecognizer") {
        return e;
    }
    st.gesture.initialized = false;
    st.gesture.table.clear();
    st.gesture.port_last_us = [None, None];
    0
}

/// `int sceSystemGestureResetPrimitiveTouchRecognizer(void)`: forget every tracked primitive.
/// A finger still down after this is a NEW primitive at the next update; a recognizer that
/// was following one of the dropped primitives reports its gesture CANCELLED.
#[hostcall]
pub(super) fn reset_primitive_touch_recognizer(_ctx: &mut GuestCtx, st: &mut VitaState) -> i32 {
    reset_primitive_touch_recognizer_impl(st)
}

fn reset_primitive_touch_recognizer_impl(st: &mut VitaState) -> i32 {
    if let Err(e) = require_init(st, "sceSystemGestureResetPrimitiveTouchRecognizer") {
        return e;
    }
    st.gesture.table.clear();
    st.gesture.port_last_us = [None, None];
    0
}

/// `int sceSystemGestureUpdatePrimitiveTouchRecognizer(const SceTouchData *pFrontData,
///                                                    const SceTouchData *pBackData)`
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
/// Only `r0` and `r1` are set for this call - the `1`s still in `r2`/`r3` are the leftover
/// `nBufs` of the two reads above. The two buffers are `0x90` bytes apart, exactly one
/// `SceTouchData`.
///
/// The primitives are derived from THESE samples - the bytes the title passed, with their
/// own timestamps - not from a second look at the host's touch state, so a title that
/// filters, offsets or replays its samples before handing them over is recognized on what
/// it handed over, as on the console.
#[hostcall]
pub(super) fn update_primitive_touch_recognizer(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    front: Ptr,
    back: Ptr,
) -> i32 {
    update_primitive_touch_recognizer_impl(ctx, st, front, back)
}

fn update_primitive_touch_recognizer_impl(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    front: Ptr,
    back: Ptr,
) -> i32 {
    if let Err(e) = require_init(st, "sceSystemGestureUpdatePrimitiveTouchRecognizer") {
        return e;
    }
    let read = |addr: u32| (addr != 0).then(|| read_sample(ctx, addr));
    let samples = [read(front.addr()), read(back.addr())];
    // Diagnostic (`RUST_LOG=vitaslop::input=trace`): what the title actually handed over, per
    // panel - a title that copies its own samples before feeding them can lose the touch on
    // the way, and nothing downstream can tell that from a touch that never happened.
    if tracing::enabled!(target: "vitaslop::input", tracing::Level::TRACE) {
        for (port, s) in samples.iter().enumerate() {
            if let Some(s) = s.as_ref().filter(|s| !s.reports.is_empty()) {
                tracing::trace!(
                    target: "vitaslop::input",
                    "gesture primitive update port {port}: ts {} reports {:?}",
                    s.ts,
                    s.reports.iter().map(|r| (r.id, r.x, r.y)).collect::<Vec<_>>()
                );
            }
        }
    }
    if samples.iter().any(Option::is_none) {
        note_model(
            "gesture-null-sample",
            "a NULL SceTouchData was passed for one panel and is read as an EMPTY sample \
             (NULL_SAMPLE_IS_EMPTY_PANEL)",
        );
    }
    let fallback = st.guest_mono_us();
    st.gesture.advance(samples, fallback);
    0
}

/// `int sceSystemGestureGetPrimitiveTouchEvents(SceSystemGesturePrimitiveTouchEvent *buffer,
///     SceUInt32 capacityOfBuffer, SceUInt32 *numberOfEvent)`: this update's primitives in
/// table order, `min(count, capacity)` of them, and the number copied.
#[hostcall]
pub(super) fn get_primitive_touch_events(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    events: Ptr,
    capacity: u32,
    count: Ptr,
) -> i32 {
    get_primitive_touch_events_impl(ctx, st, events, capacity, count)
}

fn get_primitive_touch_events_impl(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    events: Ptr,
    capacity: u32,
    count: Ptr,
) -> i32 {
    if let Err(e) = require_init(st, "sceSystemGestureGetPrimitiveTouchEvents") {
        return e;
    }
    if events.is_null() || count.is_null() || capacity == 0 {
        return SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT;
    }
    let n = st.gesture.table.len().min(capacity as usize);
    for (i, p) in st.gesture.table.iter().take(n).enumerate() {
        write_primitive(ctx, events.addr() + i as u32 * PRIMITIVE_EVENT_SIZE, p);
    }
    ctx.write_u32(count.addr(), n as u32);
    0
}

/// `int sceSystemGestureGetPrimitiveTouchEventsCount(void)`: the number of tracked
/// primitives, ENDs of this update included - the count is the return value.
#[hostcall]
pub(super) fn get_primitive_touch_events_count(_ctx: &mut GuestCtx, st: &mut VitaState) -> i32 {
    match require_init(st, "sceSystemGestureGetPrimitiveTouchEventsCount") {
        Ok(()) => st.gesture.table.len() as i32,
        Err(e) => e,
    }
}

/// `int sceSystemGestureGetPrimitiveTouchEventByIndex(SceUInt32 index,
///     SceSystemGesturePrimitiveTouchEvent *primitiveTouchEvent)`
#[hostcall]
pub(super) fn get_primitive_touch_event_by_index(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    index: u32,
    event: Ptr,
) -> i32 {
    get_primitive_touch_event_by_index_impl(ctx, st, index, event)
}

fn get_primitive_touch_event_by_index_impl(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    index: u32,
    event: Ptr,
) -> i32 {
    if let Err(e) = require_init(st, "sceSystemGestureGetPrimitiveTouchEventByIndex") {
        return e;
    }
    if event.is_null() {
        return SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT;
    }
    match st.gesture.table.get(index as usize) {
        Some(p) => {
            write_primitive(ctx, event.addr(), p);
            0
        }
        None => SCE_SYSTEM_GESTURE_ERROR_INDEX_OUT_OF_ARRAY,
    }
}

/// `int sceSystemGestureGetPrimitiveTouchEventByPrimitiveID(SceUInt16 primitiveID,
///     SceSystemGesturePrimitiveTouchEvent *primitiveTouchEvent)`
///
/// EVIDENCE for the shape, from a retail consumer and from the argument dump the
/// unimplemented-NID hard-fail printed the first time its tap path reached this call: `r0` is
/// the halfword the title just loaded out of a gesture event's `property.*.primitiveID`
/// (event + 0x18), and `r1` a buffer in its own frame that it reads back only on a ZERO
/// return:
///
/// ```text
///   LDRH  r0, [sp, #0xd8]         ; event + 0x18
///   BLX   -> HERE
///   MOVS  lr, r0                  ; the return
///   BNE   +6                      ; non-zero: skip the load below
///   LDRSH r4, [sp, #0xa]          ; r1 + 0x0a (pressedForce) - only read when this returned 0
/// ```
///
/// An id no tracked primitive carries is EVENT_DATA_NOT_FOUND and the buffer is left alone.
#[hostcall]
pub(super) fn get_primitive_touch_event_by_primitive_id(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    primitive_id: u32,
    event: Ptr,
) -> i32 {
    get_primitive_touch_event_by_primitive_id_impl(ctx, st, primitive_id, event)
}

fn get_primitive_touch_event_by_primitive_id_impl(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    primitive_id: u32,
    event: Ptr,
) -> i32 {
    if let Err(e) = require_init(st, "sceSystemGestureGetPrimitiveTouchEventByPrimitiveID") {
        return e;
    }
    if event.is_null() {
        return SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT;
    }
    // A SceUInt16 argument: only the low halfword of r0 is the id.
    let id = primitive_id as u16;
    match st.gesture.table.iter().find(|p| p.id == id) {
        Some(p) => {
            write_primitive(ctx, event.addr(), p);
            0
        }
        None => SCE_SYSTEM_GESTURE_ERROR_EVENT_DATA_NOT_FOUND,
    }
}

// ---------------------------------------------------------------------------------------
// Recognizers: state in the caller's block.
// ---------------------------------------------------------------------------------------

/// `sizeof(SceSystemGestureTouchRecognizer)`: `uint64 reserved[307]`. A retail title lays its
/// recognizers out at `+1312, +3768, +6224, ...` - a stride of exactly this.
const RECOGNIZER_SIZE: usize = 2456;

/// Marker at the head of a recognizer block, written by [`create_touch_recognizer`], so a
/// later call can tell a recognizer from an uninitialized buffer.
///
/// ALL of a recognizer's state - identity, rectangle, parameter, in-progress gestures, the
/// published event list - lives IN this block rather than in a host table keyed by its
/// address, because a guest is free to hand back a COPY of a POD work area, and an
/// address-keyed table cannot recognize one (memory `vitaslop-host-call-reference-semantics`).
/// The library allocates nothing per recognizer either.
const RECOGNIZER_MAGIC: u32 = 0x5347_5232; // "SGR2": the stateful layout below

/// Our layout inside the opaque block (the console's own is unknown and unobservable - the
/// title never reads the block, it only passes it back):
///
/// ```text
///   0   u32  magic           4   i32  gestureType     8  i16 x4 rectangle
///   16  u32  port            20  u32  last eventID    24 u64 updatedTime
///   32  u32  primitive seq   36  u32  #instances      40 u32 #events
///   44  u8   tap chain count 46  i16 x2 chain position  56 u64 chain release time
///   64  [64] the creation parameter (the SceSystemGestureTouchRecognizerParameter union)
///   128 instances: MAX_INSTANCES x INSTANCE_SIZE
///   ..  events:    MAX_EVENTS x EVENT_REC_SIZE
/// ```
const OFF_PARAM: usize = 64;
const OFF_INSTANCES: usize = 128;
const INSTANCE_SIZE: usize = 16;
/// In-progress gestures one recognizer tracks: every report of one panel (8) plus the ENDs
/// of as many that lifted this update.
const MAX_INSTANCES: usize = 16;
const OFF_EVENTS: usize = OFF_INSTANCES + MAX_INSTANCES * INSTANCE_SIZE;
/// One published event, compact: eventID, eventState, updatedTime, and the 32 bytes of the
/// property union any gesture type uses (pinch is the largest, exactly 32).
const EVENT_REC_SIZE: usize = 48;
const PROP_SIZE: usize = 32;
/// Events one update can publish. The console's own maximum is not known (titles pass a
/// capacity of 6); this is one per possible instance, and an overflow is reported.
const MAX_EVENTS: usize = 16;
const RECOGNIZER_USED: usize = OFF_EVENTS + MAX_EVENTS * EVENT_REC_SIZE;
const _: () = assert!(RECOGNIZER_USED <= RECOGNIZER_SIZE);

/// `SceSystemGestureRectangle`: four `i16` as `(x, y, width, height)`.
///
/// EVIDENCE, from the calling title building the argument immediately before the call: four
/// 16-bit stores into one 8-byte object - `x` and `y` copied straight from a pair of signed
/// halfwords, then `x1 - x` and `y1 - y` - so the last two are EXTENTS, not a second corner.
/// The values it passes are `(0, 0, 1919, 1087)`, the front panel extent
/// [`crate::vita::touch::panel_info_bytes`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rect {
    x: i16,
    y: i16,
    width: i16,
    height: i16,
}

impl Rect {
    fn read(ctx: &GuestCtx, addr: u32) -> Rect {
        let b = ctx.read_bytes(addr, 8);
        let h = |o: usize| b.get(o..o + 2).map_or(0, |s| i16::from_le_bytes([s[0], s[1]]));
        Rect {
            x: h(0),
            y: h(2),
            width: h(4),
            height: h(6),
        }
    }

    /// >>> A NULL RECTANGLE IS "THE WHOLE PANEL", NOT "NOWHERE". A retail title passes 0 for
    /// every one of its ten recognizers, and a title comment elsewhere calls the
    /// NULL-rectangle recognizer the one that "covers the whole front pad". The extent comes
    /// from `touch` so the two cannot drift.
    fn whole_panel(port: u32) -> Rect {
        Rect {
            x: 0,
            y: 0,
            width: super::touch::MAX_ACTIVE_X,
            height: super::touch::max_active_y(port),
        }
    }

    fn contains(&self, x: i16, y: i16) -> bool {
        let (x0, y0) = (i32::from(self.x), i32::from(self.y));
        let (x1, y1) = (x0 + i32::from(self.width), y0 + i32::from(self.height));
        let (px, py) = (i32::from(x), i32::from(y));
        if RECT_FAR_EDGE_INCLUSIVE {
            (x0..=x1).contains(&px) && (y0..=y1).contains(&py)
        } else {
            (x0..x1).contains(&px) && (y0..y1).contains(&py)
        }
    }
}

/// Phase of one in-progress gesture.
const PHASE_CANDIDATE: u8 = 0;
const PHASE_BEGUN: u8 = 1;
const PHASE_FAILED: u8 = 2;

/// One in-progress gesture instance: the primitive(s) it follows and, once it has begun,
/// the eventID it reports under.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Instance {
    event_id: u32,
    phase: u8,
    prim_a: u16,
    /// Pinch only: the second finger.
    prim_b: u16,
    /// Pinch only: the finger distance at BEGIN, the denominator of `scale`.
    base_dist: f32,
}

/// One published event.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Event {
    id: u32,
    state: i32,
    updated_us: u64,
    prop: [u8; PROP_SIZE],
}

/// A recognizer block, decoded.
#[derive(Clone, Debug, PartialEq)]
struct Rec {
    kind: i32,
    rect: Rect,
    port: u32,
    last_event_id: u32,
    updated_us: u64,
    seq: u32,
    chain_count: u8,
    chain_pos: (i16, i16),
    chain_release_us: u64,
    param: [u8; 64],
    instances: Vec<Instance>,
    events: Vec<Event>,
}

fn get_u16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn get_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn get_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
fn put(b: &mut [u8], o: usize, v: &[u8]) {
    b[o..o + v.len()].copy_from_slice(v);
}

impl Rec {
    fn new(kind: i32, port: u32, rect: Rect, param: [u8; 64]) -> Rec {
        Rec {
            kind,
            rect,
            port,
            last_event_id: 0,
            updated_us: 0,
            seq: 0,
            chain_count: 0,
            chain_pos: (0, 0),
            chain_release_us: 0,
            param,
            instances: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Decode the block at `addr`, or refuse with INVALID_ARGUMENT for a NULL pointer or a
    /// block no `sceSystemGestureCreateTouchRecognizer` initialized (which is a guest bug or
    /// a wrong argument reading - either way it must not pass silently).
    fn load(ctx: &GuestCtx, addr: u32, call: &'static str) -> Result<Rec, i32> {
        if addr == 0 {
            return Err(SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT);
        }
        let b = ctx.read_bytes(addr, RECOGNIZER_USED);
        let (ni, ne) = if b.len() == RECOGNIZER_USED && get_u32(&b, 0) == RECOGNIZER_MAGIC {
            (get_u32(&b, 36) as usize, get_u32(&b, 40) as usize)
        } else {
            (usize::MAX, usize::MAX)
        };
        if ni > MAX_INSTANCES || ne > MAX_EVENTS {
            report_once(
                call,
                &format!(
                    "{call}: called on a work area ({addr:#010x}, head word {:#010x}) that no \
                     sceSystemGestureCreateTouchRecognizer in this run initialized - refused with \
                     INVALID_ARGUMENT. Either the title creates recognizers by a path not seen \
                     yet, or this argument shape is wrong.",
                    b.first_chunk::<4>().map_or(0, |w| u32::from_le_bytes(*w)),
                ),
            );
            return Err(SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT);
        }
        let instances = (0..ni)
            .map(|i| {
                let o = OFF_INSTANCES + i * INSTANCE_SIZE;
                Instance {
                    event_id: get_u32(&b, o),
                    phase: b[o + 4],
                    prim_a: get_u16(&b, o + 6),
                    prim_b: get_u16(&b, o + 8),
                    base_dist: f32::from_bits(get_u32(&b, o + 12)),
                }
            })
            .collect();
        let events = (0..ne)
            .map(|i| {
                let o = OFF_EVENTS + i * EVENT_REC_SIZE;
                Event {
                    id: get_u32(&b, o),
                    state: get_u32(&b, o + 4) as i32,
                    updated_us: get_u64(&b, o + 8),
                    prop: b[o + 16..o + 16 + PROP_SIZE].try_into().unwrap(),
                }
            })
            .collect();
        Ok(Rec {
            kind: get_u32(&b, 4) as i32,
            rect: Rect {
                x: get_u16(&b, 8) as i16,
                y: get_u16(&b, 10) as i16,
                width: get_u16(&b, 12) as i16,
                height: get_u16(&b, 14) as i16,
            },
            port: get_u32(&b, 16),
            last_event_id: get_u32(&b, 20),
            updated_us: get_u64(&b, 24),
            seq: get_u32(&b, 32),
            chain_count: b[44],
            chain_pos: (get_u16(&b, 46) as i16, get_u16(&b, 48) as i16),
            chain_release_us: get_u64(&b, 56),
            param: b[OFF_PARAM..OFF_PARAM + 64].try_into().unwrap(),
            instances,
            events,
        })
    }

    fn store(&self, ctx: &mut GuestCtx, addr: u32) {
        let mut b = vec![0u8; RECOGNIZER_USED];
        put(&mut b, 0, &RECOGNIZER_MAGIC.to_le_bytes());
        put(&mut b, 4, &self.kind.to_le_bytes());
        put(&mut b, 8, &self.rect.x.to_le_bytes());
        put(&mut b, 10, &self.rect.y.to_le_bytes());
        put(&mut b, 12, &self.rect.width.to_le_bytes());
        put(&mut b, 14, &self.rect.height.to_le_bytes());
        put(&mut b, 16, &self.port.to_le_bytes());
        put(&mut b, 20, &self.last_event_id.to_le_bytes());
        put(&mut b, 24, &self.updated_us.to_le_bytes());
        put(&mut b, 32, &self.seq.to_le_bytes());
        put(&mut b, 36, &(self.instances.len() as u32).to_le_bytes());
        put(&mut b, 40, &(self.events.len() as u32).to_le_bytes());
        b[44] = self.chain_count;
        put(&mut b, 46, &self.chain_pos.0.to_le_bytes());
        put(&mut b, 48, &self.chain_pos.1.to_le_bytes());
        put(&mut b, 56, &self.chain_release_us.to_le_bytes());
        put(&mut b, OFF_PARAM, &self.param);
        for (i, ins) in self.instances.iter().take(MAX_INSTANCES).enumerate() {
            let o = OFF_INSTANCES + i * INSTANCE_SIZE;
            put(&mut b, o, &ins.event_id.to_le_bytes());
            b[o + 4] = ins.phase;
            put(&mut b, o + 6, &ins.prim_a.to_le_bytes());
            put(&mut b, o + 8, &ins.prim_b.to_le_bytes());
            put(&mut b, o + 12, &ins.base_dist.to_bits().to_le_bytes());
        }
        for (i, e) in self.events.iter().take(MAX_EVENTS).enumerate() {
            let o = OFF_EVENTS + i * EVENT_REC_SIZE;
            put(&mut b, o, &e.id.to_le_bytes());
            put(&mut b, o + 4, &e.state.to_le_bytes());
            put(&mut b, o + 8, &e.updated_us.to_le_bytes());
            put(&mut b, o + 16, &e.prop);
        }
        ctx.write_bytes(addr, &b);
    }

    /// The next eventID: a per-recognizer counter from 1, never 0 (titles use 0 as "no
    /// gesture tracked").
    fn mint_event_id(&mut self) -> u32 {
        self.last_event_id = self.last_event_id.wrapping_add(1).max(1);
        self.last_event_id
    }

    /// `tap.maxTapCount` (byte 0 of the parameter union); 0 or a NULL parameter = 1.
    fn max_tap_count(&self) -> u8 {
        self.param[0].max(1)
    }

    /// `tapAndHold.timeToInvokeEvent` (u64 at 0 of the union), in microseconds - see
    /// [`HOLD_TIME_TICKS_BELOW`] for the unit reading.
    fn hold_time_us(&self) -> u64 {
        let raw = u64::from_le_bytes(self.param[0..8].try_into().unwrap());
        if raw == 0 {
            note_model(
                "gesture-hold-default",
                "a TAP_AND_HOLD recognizer has no timeToInvokeEvent; its hold time is \
                 HOLD_DEFAULT_US = 500 ms",
            );
            HOLD_DEFAULT_US
        } else if raw < HOLD_TIME_TICKS_BELOW {
            note_model(
                "gesture-hold-unit",
                "a TAP_AND_HOLD timeToInvokeEvent below 1000 is counted in 60 Hz panel \
                 periods (HOLD_TIME_TICKS_BELOW / HOLD_TIME_TICK_US); its unit is unknown",
            );
            raw * HOLD_TIME_TICK_US
        } else {
            raw
        }
    }
}

fn dist2(a: (i16, i16), b: (i16, i16)) -> i64 {
    let dx = i64::from(a.0) - i64::from(b.0);
    let dy = i64::from(a.1) - i64::from(b.1);
    dx * dx + dy * dy
}

fn beyond(a: (i16, i16), b: (i16, i16), slop: i32) -> bool {
    dist2(a, b) > i64::from(slop) * i64::from(slop)
}

fn xy(p: (i16, i16, i16)) -> (i16, i16) {
    (p.0, p.1)
}

fn put_vec2(prop: &mut [u8; PROP_SIZE], o: usize, v: (i16, i16)) {
    put(prop, o, &v.0.to_le_bytes());
    put(prop, o + 2, &v.1.to_le_bytes());
}

/// `property.tap`: primitiveID +0, position +2, tappedCount +6.
fn tap_prop(p: &Primitive, count: u8) -> [u8; PROP_SIZE] {
    let mut b = [0u8; PROP_SIZE];
    put(&mut b, 0, &p.id.to_le_bytes());
    let pos = if TAP_POSITION_AT_RELEASE { p.current } else { p.pressed };
    put_vec2(&mut b, 2, xy(pos));
    b[6] = count;
    b
}

/// `property.drag`: primitiveID +0, deltaVector +2, currentPosition +6, pressedPosition +10.
fn drag_prop(p: &Primitive) -> [u8; PROP_SIZE] {
    let mut b = [0u8; PROP_SIZE];
    put(&mut b, 0, &p.id.to_le_bytes());
    put_vec2(&mut b, 2, xy(p.delta));
    put_vec2(&mut b, 6, xy(p.current));
    put_vec2(&mut b, 10, xy(p.pressed));
    b
}

/// `property.tapAndHold`: primitiveID +0, pressedPosition +2.
fn hold_prop(p: &Primitive) -> [u8; PROP_SIZE] {
    let mut b = [0u8; PROP_SIZE];
    put(&mut b, 0, &p.id.to_le_bytes());
    put_vec2(&mut b, 2, xy(p.pressed));
    b
}

/// `property.pinchOutIn`: scale f32 +0, then two 14-byte primitive records at +4 and +18,
/// each primitiveID, currentPosition, deltaVector, pairedPosition (the OTHER finger).
fn pinch_prop(scale: f32, a: &Primitive, b: &Primitive) -> [u8; PROP_SIZE] {
    let mut out = [0u8; PROP_SIZE];
    put(&mut out, 0, &scale.to_le_bytes());
    for (o, p, q) in [(4, a, b), (18, b, a)] {
        put(&mut out, o, &p.id.to_le_bytes());
        put_vec2(&mut out, o + 2, xy(p.current));
        put_vec2(&mut out, o + 6, xy(p.delta));
        put_vec2(&mut out, o + 10, xy(q.current));
    }
    out
}

fn finger_dist(a: &Primitive, b: &Primitive) -> f32 {
    (dist2(xy(a.current), xy(b.current)) as f32).sqrt()
}

/// Advance one recognizer against the primitive table (`prims`, already this update's) and
/// rebuild its event list: one event per live gesture instance this update, oldest first.
/// A gesture keeps its eventID from BEGIN to END/CANCELLED; the update AFTER the one that
/// reported END drops it, so a by-ID lookup then answers EVENT_DATA_NOT_FOUND.
fn advance_recognizer(rec: &mut Rec, prims: &[Primitive], now: u64) {
    let port = rec.port;
    let mine: Vec<&Primitive> = prims.iter().filter(|p| u32::from(p.port) == port).collect();
    let find = |id: u16| mine.iter().copied().find(|p| p.id == id);
    let mut out: Vec<(u32, i32, [u8; PROP_SIZE])> = Vec::new();
    let mut kept = Vec::new();
    let old = std::mem::take(&mut rec.instances);

    if rec.kind == TYPE_PINCH_OUT_IN {
        let mut active = false;
        for ins in old {
            match (find(ins.prim_a), find(ins.prim_b)) {
                (Some(a), Some(b)) if a.state == STATE_END || b.state == STATE_END => {
                    let scale = finger_dist(a, b) / ins.base_dist;
                    out.push((ins.event_id, STATE_END, pinch_prop(scale, a, b)));
                }
                (Some(a), Some(b)) => {
                    let scale = finger_dist(a, b) / ins.base_dist;
                    out.push((ins.event_id, STATE_ACTIVE, pinch_prop(scale, a, b)));
                    kept.push(ins);
                    active = true;
                }
                _ => cancelled(&mut out, &ins),
            }
        }
        if !active {
            let mut down = mine
                .iter()
                .filter(|p| p.state != STATE_END && rec.rect.contains(p.pressed.0, p.pressed.1));
            if let (Some(a), Some(b)) = (down.next(), down.next()) {
                let id = rec.mint_event_id();
                let ins = Instance {
                    event_id: id,
                    phase: PHASE_BEGUN,
                    prim_a: a.id,
                    prim_b: b.id,
                    // Two fingers on one point have no distance to scale by; 1 unit keeps
                    // `scale` finite.
                    base_dist: finger_dist(a, b).max(1.0),
                };
                out.push((id, STATE_BEGIN, pinch_prop(1.0, a, b)));
                kept.push(ins);
            }
        }
    } else {
        // Every primitive that already had a gesture: one whose gesture just finished or
        // failed must not be seeded afresh in the same update, or a failed tap whose finger
        // drifted back before lifting would be re-judged as a clean one.
        let seen: Vec<u16> = old.iter().map(|i| i.prim_a).collect();
        for ins in old {
            match find(ins.prim_a) {
                Some(p) => step(rec, ins, p, &mut out, &mut kept),
                None => cancelled(&mut out, &ins),
            }
        }
        for p in &mine {
            if seen.contains(&p.id) || !rec.rect.contains(p.pressed.0, p.pressed.1) {
                continue;
            }
            if kept.len() >= MAX_INSTANCES {
                report_once(
                    "gesture-instances-full",
                    "sceSystemGestureUpdateTouchRecognizer: more simultaneous touches than \
                     one recognizer tracks (16); the extra ones are ignored.",
                );
                break;
            }
            let ins = Instance {
                prim_a: p.id,
                ..Instance::default()
            };
            step(rec, ins, p, &mut out, &mut kept);
        }
    }

    if out.len() > MAX_EVENTS {
        report_once(
            "gesture-events-full",
            "sceSystemGestureUpdateTouchRecognizer: more gesture events in one update than \
             this recognizer holds (16); the newest are dropped.",
        );
        out.truncate(MAX_EVENTS);
    }
    // `updatedTime` is the update in which the event last CHANGED: an ACTIVE drag that did
    // not move, or a held finger, keeps the time it last changed (one title only refreshes
    // a value when this differs from last frame's).
    let events: Vec<Event> = out
        .into_iter()
        .map(|(id, state, prop)| {
            let updated_us = rec
                .events
                .iter()
                .find(|e| e.id == id && e.state == state && e.prop == prop)
                .map_or(now, |e| e.updated_us);
            Event {
                id,
                state,
                updated_us,
                prop,
            }
        })
        .collect();
    if events != rec.events {
        rec.updated_us = now;
    }
    rec.events = events;
    rec.instances = kept;
}

/// A gesture whose primitive vanished without an END (the primitive layer was reset under
/// it): CANCELLED if it had begun, nothing if it was still a candidate.
fn cancelled(out: &mut Vec<(u32, i32, [u8; PROP_SIZE])>, ins: &Instance) {
    if ins.phase == PHASE_BEGUN {
        note_model(
            "gesture-cancelled",
            "a gesture lost its primitive without a release and is reported CANCELLED = 4 \
             (STATE_CANCELLED); the value of that state is not published",
        );
        out.push((ins.event_id, STATE_CANCELLED, [0u8; PROP_SIZE]));
    }
}

/// One per-primitive gesture (tap, drag, hold) against its primitive's state this update.
fn step(
    rec: &mut Rec,
    mut ins: Instance,
    p: &Primitive,
    out: &mut Vec<(u32, i32, [u8; PROP_SIZE])>,
    kept: &mut Vec<Instance>,
) {
    let ended = p.state == STATE_END;
    match rec.kind {
        TYPE_TAP => {
            let moved = beyond(xy(p.current), xy(p.pressed), STATIONARY_SLOP);
            if ins.phase == PHASE_CANDIDATE && (moved || p.elapsed > TAP_MAX_DURATION_US) {
                ins.phase = PHASE_FAILED;
            }
            if !ended {
                kept.push(ins);
                return;
            }
            if ins.phase != PHASE_CANDIDATE {
                return;
            }
            note_model(
                "gesture-tap",
                "a TAP is a press held at most 500 ms (TAP_MAX_DURATION_US) within 40 panel \
                 units (STATIONARY_SLOP), reported on release with state END \
                 (TAP_EVENT_STATE) at the release point (TAP_POSITION_AT_RELEASE)",
            );
            let pos = xy(if TAP_POSITION_AT_RELEASE { p.current } else { p.pressed });
            let max = rec.max_tap_count();
            let count = if max > 1
                && rec.chain_count > 0
                && rec.chain_count < max
                && p.press_us.saturating_sub(rec.chain_release_us) <= MULTI_TAP_INTERVAL_US
                && !beyond(pos, rec.chain_pos, MULTI_TAP_SLOP)
            {
                note_model(
                    "gesture-multi-tap",
                    "a multi-tap continues within 300 ms (MULTI_TAP_INTERVAL_US) and 100 \
                     panel units (MULTI_TAP_SLOP) of the previous tap",
                );
                rec.chain_count + 1
            } else {
                1
            };
            rec.chain_count = count;
            rec.chain_pos = pos;
            rec.chain_release_us = p.press_us + p.elapsed;
            let id = rec.mint_event_id();
            out.push((id, TAP_EVENT_STATE, tap_prop(p, count)));
        }
        TYPE_DRAG => {
            if ins.phase == PHASE_CANDIDATE {
                if ended {
                    return;
                }
                let slop = drag_slop();
                if slop >= 0 && !beyond(xy(p.current), xy(p.pressed), slop) {
                    kept.push(ins);
                    return;
                }
                ins.phase = PHASE_BEGUN;
                ins.event_id = rec.mint_event_id();
                out.push((ins.event_id, STATE_BEGIN, drag_prop(p)));
                kept.push(ins);
            } else if ended {
                out.push((ins.event_id, STATE_END, drag_prop(p)));
            } else {
                out.push((ins.event_id, STATE_ACTIVE, drag_prop(p)));
                kept.push(ins);
            }
        }
        TYPE_TAP_AND_HOLD => match ins.phase {
            PHASE_CANDIDATE => {
                if ended {
                    return;
                }
                if beyond(xy(p.current), xy(p.pressed), STATIONARY_SLOP) {
                    ins.phase = PHASE_FAILED;
                } else if p.elapsed >= rec.hold_time_us() {
                    note_model(
                        "gesture-hold",
                        "a TAP_AND_HOLD fires once the press stays within 40 panel units \
                         (STATIONARY_SLOP) for its hold time",
                    );
                    ins.phase = PHASE_BEGUN;
                    ins.event_id = rec.mint_event_id();
                    out.push((ins.event_id, STATE_BEGIN, hold_prop(p)));
                }
                kept.push(ins);
            }
            PHASE_BEGUN if ended => out.push((ins.event_id, STATE_END, hold_prop(p))),
            PHASE_BEGUN => {
                out.push((ins.event_id, STATE_ACTIVE, hold_prop(p)));
                kept.push(ins);
            }
            _ if ended => {}
            _ => kept.push(ins),
        },
        _ => {}
    }
}

/// How many recognizers this run has created, so each is logged with a stable index.
static RECOGNIZER_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// `int sceSystemGestureCreateTouchRecognizer(SceSystemGestureTouchRecognizer *touchRecognizer,
///        SceSystemGestureType gestureType, SceUInt8 touchPanelPortID,
///        const SceSystemGestureRectangle *rectangle,
///        const SceSystemGestureTouchRecognizerParameter *touchRecognizerParameter)`
///
/// EVIDENCE for the shape: the calling title sets `r0 = <zeroed work area>`, `r1 = <type>`,
/// `r2 = 0 or 1`, `r3 = <the rectangle it just built>`, and the first stack word (the
/// parameter) is 0. `r2` is the panel - literally 0 or 1 across every call, never an
/// address, the same `SCE_TOUCH_PORT_FRONT`/`_BACK` selector `sceTouchRead` takes.
///
/// Both the rectangle and the parameter may be NULL: the whole panel, and the defaults.
/// Observed parameters: `tap.maxTapCount = 2` for a double-tap recognizer (the title then
/// checks `tappedCount == 2`), and `tapAndHold.timeToInvokeEvent = 30`.
#[hostcall]
pub(super) fn create_touch_recognizer(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    recognizer: Ptr,
    kind: i32,
    port: u32,
    rect: Ptr,
    param: Ptr,
) -> i32 {
    create_touch_recognizer_impl(ctx, recognizer, kind, port, rect, param)
}

fn create_touch_recognizer_impl(
    ctx: &mut GuestCtx,
    recognizer: Ptr,
    kind: i32,
    port: u32,
    rect: Ptr,
    param: Ptr,
) -> i32 {
    if recognizer.is_null() {
        return SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT;
    }
    // >>> ROTATION (16) IS A GESTURE THIS CONSOLE'S LIBRARY CREATES. It was refused here on an
    // inference that only the later console family has it, and a retail title disproves that:
    // it creates a rotation recognizer per panel next to its tap/drag/hold/pinch ones and polls
    // it every frame without checking the create's result. Refused, the block stayed
    // uninitialised, every poll returned INVALID_ARGUMENT, and the title read that error as an
    // UNSIGNED event count - appending ~2^31 garbage events, growing its vector until the heap
    // it shares with its loading thread corrupted (a frame-0 crash). FLICK (32) has no such
    // evidence and stays refused.
    if ![TYPE_TAP, TYPE_DRAG, TYPE_TAP_AND_HOLD, TYPE_PINCH_OUT_IN, TYPE_ROTATION].contains(&kind) {
        return SCE_SYSTEM_GESTURE_ERROR_NOT_SUPPORTED_GESTURE;
    }
    if kind == TYPE_ROTATION {
        report_once(
            "gesture-rotation",
            "sceSystemGestureCreateTouchRecognizer: a ROTATION recognizer - created, but its              detection is NOT modelled (no source gives its event layout), so it reports no              events. A title that turns things with two fingers will not see the turn.",
        );
    }
    // A SceUInt8 argument: only the low byte of r2 is the port.
    let port = port & 0xff;
    if port > 1 {
        return SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT;
    }
    let r = if rect.is_null() {
        Rect::whole_panel(port)
    } else {
        Rect::read(ctx, rect.addr())
    };
    let mut p = [0u8; 64];
    if !param.is_null() {
        let b = ctx.read_bytes(param.addr(), 64);
        p[..b.len()].copy_from_slice(&b);
    }
    // The whole block is the library's: zero it, then write the recognizer into it.
    ctx.write_bytes(recognizer.addr(), &[0u8; RECOGNIZER_SIZE]);
    Rec::new(kind, port, r, p).store(ctx, recognizer.addr());
    if RECT_FAR_EDGE_INCLUSIVE {
        note_model(
            "gesture-rect-edge",
            "a recognizer rectangle's far edge is INSIDE it (RECT_FAR_EDGE_INCLUSIVE)",
        );
    }
    // Log every recognizer as it is created, unconditionally and once each. A recognizer
    // only reports a touch that began inside its rectangle on its own panel, so this set is
    // the map of where the title is listening - the first thing worth knowing when a
    // scripted tap "has an effect but does not do anything".
    tracing::info!(
        target: "vitaslop::status",
        "sceSystemGesture: recognizer #{} type={kind} port={port} rect=({},{} {}x{}) \
         param[0]={} from rect_ptr={:#010x} at {:#010x}",
        RECOGNIZER_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        r.x,
        r.y,
        r.width,
        r.height,
        p[0],
        rect.addr(),
        recognizer.addr(),
    );
    // An EMPTY rectangle can never contain a touch, so this recognizer will never report:
    // say so rather than let every scripted tap read as "the input did nothing".
    if r.width <= 0 || r.height <= 0 {
        report_once(
            "gesture-empty-rect",
            "sceSystemGesture: a recognizer was created with an EMPTY rectangle, so no touch \
             can begin inside it and it will report NO events until the rectangle is \
             updated - see the `rect_ptr` in the recognizer lines above.",
        );
    }
    0
}

/// `int sceSystemGestureUpdateTouchRecognizer(SceSystemGestureTouchRecognizer *touchRecognizer)`
///
/// EVIDENCE for the ONE-argument shape: the title calls this once per recognizer in a
/// straight run of calls that each set only `r0` to a different offset in its own state
/// block (`r4 + 1312`, `+3768`, `+6224`, ... - a 2456 stride). A dump showing more
/// registers set is the previous call's leftovers; only `r0` is read.
#[hostcall]
pub(super) fn update_touch_recognizer(
    ctx: &mut GuestCtx,
    st: &mut VitaState,
    recognizer: Ptr,
) -> i32 {
    update_touch_recognizer_impl(ctx, st, recognizer)
}

fn update_touch_recognizer_impl(ctx: &mut GuestCtx, st: &mut VitaState, recognizer: Ptr) -> i32 {
    let mut rec = match Rec::load(ctx, recognizer.addr(), "sceSystemGestureUpdateTouchRecognizer") {
        Ok(r) => r,
        Err(e) => return e,
    };
    if let Err(e) = require_init(st, "sceSystemGestureUpdateTouchRecognizer") {
        return e;
    }
    let layer = &st.gesture;
    if rec.seq == layer.seq {
        // No primitive update since this recognizer last advanced: its events stand.
        return 0;
    }
    rec.seq = layer.seq;
    advance_recognizer(&mut rec, &layer.table, layer.updated_us);
    // Diagnostic (`RUST_LOG=vitaslop::input=trace`): the recognizer, the primitives offered
    // to it and what it reported. A tap that is delivered, inside the right rectangle, and
    // still selects nothing is a different bug from one that misses the rectangle or never
    // arrives - and the screen is identical in all three cases.
    if !layer.table.is_empty() && tracing::enabled!(target: "vitaslop::input", tracing::Level::TRACE) {
        tracing::trace!(
            target: "vitaslop::input",
            "gesture recognizer type {} port {} rect ({},{} {}x{}): {} primitive(s), {} event(s) {:?}",
            rec.kind,
            rec.port,
            rec.rect.x,
            rec.rect.y,
            rec.rect.width,
            rec.rect.height,
            layer.table.len(),
            rec.events.len(),
            rec.events.iter().map(|e| (e.id, e.state)).collect::<Vec<_>>(),
        );
    }
    rec.store(ctx, recognizer.addr());
    0
}

/// `int sceSystemGestureResetTouchRecognizer(SceSystemGestureTouchRecognizer *touchRecognizer)`
///
/// EVIDENCE for the one-argument shape: the caller sets only `r0` before the branch. Titles
/// call it after suspend/resume and after a by-ID lookup found their gesture gone. Discards
/// every event and in-progress gesture (and the multi-tap chain); the type, panel, rectangle
/// and parameter stay. The primitive update it last consumed is kept, so a reset followed by
/// an update in the same frame does not re-derive that frame's table.
#[hostcall]
pub(super) fn reset_touch_recognizer(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    recognizer: Ptr,
) -> i32 {
    match Rec::load(ctx, recognizer.addr(), "sceSystemGestureResetTouchRecognizer") {
        Ok(mut rec) => {
            rec.instances.clear();
            rec.events.clear();
            rec.chain_count = 0;
            rec.store(ctx, recognizer.addr());
            0
        }
        Err(e) => e,
    }
}

/// `int sceSystemGestureUpdateTouchRecognizerRectangle(
///        SceSystemGestureTouchRecognizer *touchRecognizer, const SceSystemGestureRectangle *rectangle)`
///
/// Replaces the watched rectangle in place; titles grow or shrink an active area at runtime
/// with it. Gestures already in progress continue (the rectangle is tested at PRESS time).
/// A NULL rectangle means the whole panel, as it does at creation.
#[hostcall]
pub(super) fn update_touch_recognizer_rectangle(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    recognizer: Ptr,
    rect: Ptr,
) -> i32 {
    match Rec::load(ctx, recognizer.addr(), "sceSystemGestureUpdateTouchRecognizerRectangle") {
        Ok(mut rec) => {
            rec.rect = if rect.is_null() {
                Rect::whole_panel(rec.port)
            } else {
                Rect::read(ctx, rect.addr())
            };
            rec.store(ctx, recognizer.addr());
            0
        }
        Err(e) => e,
    }
}

/// `sizeof(SceSystemGestureTouchRecognizerInformation)`: gestureType i32 at 0,
/// touchPanelPortID u32 at 4, rectangle at 8, updatedTime u64 at 16, 256 reserved from 24.
const INFORMATION_SIZE: usize = 280;

/// `int sceSystemGestureGetTouchRecognizerInformation(
///        const SceSystemGestureTouchRecognizer *touchRecognizer,
///        SceSystemGestureTouchRecognizerInformation *information)`
///
/// EVIDENCE for the two-argument shape: a title's setup calls this once per recognizer it
/// just created, setting only `r0 = <recognizer>` and `r1 = <one shared stack buffer>`.
/// Fills the type, panel, current rectangle and the time of the recognizer's last update
/// that changed its events; the reserved tail is zeroed, the whole struct being the
/// library's to fill.
#[hostcall]
pub(super) fn get_touch_recognizer_information(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    recognizer: Ptr,
    information: Ptr,
) -> i32 {
    get_touch_recognizer_information_impl(ctx, recognizer, information)
}

fn get_touch_recognizer_information_impl(ctx: &mut GuestCtx, recognizer: Ptr, information: Ptr) -> i32 {
    let rec = match Rec::load(ctx, recognizer.addr(), "sceSystemGestureGetTouchRecognizerInformation") {
        Ok(r) => r,
        Err(e) => return e,
    };
    if information.is_null() {
        return SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT;
    }
    let mut b = [0u8; INFORMATION_SIZE];
    put(&mut b, 0, &rec.kind.to_le_bytes());
    put(&mut b, 4, &rec.port.to_le_bytes());
    put(&mut b, 8, &rec.rect.x.to_le_bytes());
    put(&mut b, 10, &rec.rect.y.to_le_bytes());
    put(&mut b, 12, &rec.rect.width.to_le_bytes());
    put(&mut b, 14, &rec.rect.height.to_le_bytes());
    put(&mut b, 16, &rec.updated_us.to_le_bytes());
    ctx.write_bytes(information.addr(), &b);
    0
}

/// `sizeof(SceSystemGestureTouchEvent)` = 0x90.
///
/// EVIDENCE: the one title that calls [`get_touch_events`] passes a capacity of 6 and a
/// 0x360-byte stack buffer, and copies each returned event out with a 0x90-byte `memcpy` -
/// six elements of 0x90 is exactly the buffer.
const TOUCH_EVENT_SIZE: u32 = 0x90;

/// Serialize one event as a `SceSystemGestureTouchEvent`:
///
/// ```text
///   0  u32 eventID     4  i32 eventState   8  i32 gestureType   12  pad
///   16 u64 updatedTime 24 [64] property     88 [56] reserved
/// ```
///
/// Consumers read: a halfword at +0x18 (`property.*.primitiveID`) handed to
/// `GetPrimitiveTouchEventByPrimitiveID`; signed halfwords at +26/+28 as a coordinate
/// (`property.tap.position`); the word at +8 ANDed with TAP/DRAG; the byte at +0x1e
/// (`property.tap.tappedCount`) tested `> 1`. The whole element is written: a stale byte of
/// the caller's stack would otherwise read as part of the event.
fn event_bytes(kind: i32, e: &Event) -> [u8; TOUCH_EVENT_SIZE as usize] {
    let mut b = [0u8; TOUCH_EVENT_SIZE as usize];
    put(&mut b, 0, &e.id.to_le_bytes());
    put(&mut b, 4, &e.state.to_le_bytes());
    put(&mut b, 8, &kind.to_le_bytes());
    put(&mut b, 16, &e.updated_us.to_le_bytes());
    put(&mut b, 24, &e.prop);
    b
}

/// `int sceSystemGestureGetTouchEventsCount(const SceSystemGestureTouchRecognizer *touchRecognizer)`
///
/// EVIDENCE for the count being the RETURN VALUE: the title passes only the recognizer, and
/// the instruction after the branch is `MOV r8, r0` - kept as a number and used as the bound
/// of its `GetTouchEventByIndex` loop.
#[hostcall]
pub(super) fn get_touch_events_count(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    recognizer: Ptr,
) -> i32 {
    match Rec::load(ctx, recognizer.addr(), "sceSystemGestureGetTouchEventsCount") {
        Ok(rec) => rec.events.len() as i32,
        Err(e) => e,
    }
}

/// `int sceSystemGestureGetTouchEvents(const SceSystemGestureTouchRecognizer *touchRecognizer,
///        SceSystemGestureTouchEvent *eventBuffer, SceUInt32 capacityOfBuffer,
///        SceUInt32 *numberOfEvent)`
///
/// EVIDENCE for the shape: the caller sets `r0 = <recognizer>`, `r1 = <stack buffer>`,
/// `r2 = 6`, `r3 = <a local count>`, right after `UpdateTouchRecognizer` on the same
/// recognizer. Copies `min(count, capacity)` events in the order [`get_touch_event_by_index`]
/// uses, and writes the number copied; elements past it are left alone.
#[hostcall]
pub(super) fn get_touch_events(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    recognizer: Ptr,
    buffer: Ptr,
    capacity: u32,
    count: Ptr,
) -> i32 {
    get_touch_events_impl(ctx, recognizer, buffer, capacity, count)
}

fn get_touch_events_impl(ctx: &mut GuestCtx, recognizer: Ptr, buffer: Ptr, capacity: u32, count: Ptr) -> i32 {
    let rec = match Rec::load(ctx, recognizer.addr(), "sceSystemGestureGetTouchEvents") {
        Ok(r) => r,
        Err(e) => return e,
    };
    if buffer.is_null() || count.is_null() || capacity == 0 {
        return SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT;
    }
    let n = rec.events.len().min(capacity as usize);
    for (i, e) in rec.events.iter().take(n).enumerate() {
        ctx.write_bytes(buffer.addr() + i as u32 * TOUCH_EVENT_SIZE, &event_bytes(rec.kind, e));
    }
    ctx.write_u32(count.addr(), n as u32);
    0
}

/// `int sceSystemGestureGetTouchEventByIndex(const SceSystemGestureTouchRecognizer *touchRecognizer,
///        SceUInt32 index, SceSystemGestureTouchEvent *touchEvent)`
///
/// EVIDENCE for the argument order: the caller sets `r0 = <the recognizer it just counted
/// events on>`, `r1 = <its loop counter>`, `r2 = sp`, then reads the event out of that stack
/// buffer (`LDRSH r2, [sp, #26]` / `LDRSH r3, [sp, #28]`: `property.tap.position`).
#[hostcall]
pub(super) fn get_touch_event_by_index(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    recognizer: Ptr,
    index: u32,
    event: Ptr,
) -> i32 {
    get_touch_event_by_index_impl(ctx, recognizer, index, event)
}

fn get_touch_event_by_index_impl(ctx: &mut GuestCtx, recognizer: Ptr, index: u32, event: Ptr) -> i32 {
    let rec = match Rec::load(ctx, recognizer.addr(), "sceSystemGestureGetTouchEventByIndex") {
        Ok(r) => r,
        Err(e) => return e,
    };
    if event.is_null() {
        return SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT;
    }
    match rec.events.get(index as usize) {
        Some(e) => {
            ctx.write_bytes(event.addr(), &event_bytes(rec.kind, e));
            0
        }
        None => SCE_SYSTEM_GESTURE_ERROR_INDEX_OUT_OF_ARRAY,
    }
}

/// `int sceSystemGestureGetTouchEventByEventID(const SceSystemGestureTouchRecognizer *touchRecognizer,
///        SceUInt32 eventID, SceSystemGestureTouchEvent *touchEvent)`
///
/// How a title follows ONE gesture across frames: it keeps the eventID from BEGIN and looks
/// it up each frame. An ID the recognizer no longer holds is EVENT_DATA_NOT_FOUND, which
/// titles read as "the gesture I was tracking is gone" (and reset the recognizer).
#[hostcall]
pub(super) fn get_touch_event_by_event_id(
    ctx: &mut GuestCtx,
    _st: &mut VitaState,
    recognizer: Ptr,
    event_id: u32,
    event: Ptr,
) -> i32 {
    get_touch_event_by_event_id_impl(ctx, recognizer, event_id, event)
}

fn get_touch_event_by_event_id_impl(ctx: &mut GuestCtx, recognizer: Ptr, event_id: u32, event: Ptr) -> i32 {
    let rec = match Rec::load(ctx, recognizer.addr(), "sceSystemGestureGetTouchEventByEventID") {
        Ok(r) => r,
        Err(e) => return e,
    };
    if event.is_null() {
        return SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT;
    }
    match rec.events.iter().find(|e| e.id == event_id) {
        Some(e) => {
            ctx.write_bytes(event.addr(), &event_bytes(rec.kind, e));
            0
        }
        None => SCE_SYSTEM_GESTURE_ERROR_EVENT_DATA_NOT_FOUND,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{CtrlFrame, World};
    use crate::{SliceMemory, VFP_ARG_COUNT};
    use vitaslop_transpiler::abi::{REG_COUNT, SP};

    struct Idle;
    impl World for Idle {
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
    }

    const MEM: usize = 0x8000;
    const STACK: u32 = 0x7F00;
    const FRONT: u32 = 0x100;
    const BACK: u32 = FRONT + TOUCH_DATA_SIZE as u32;
    const RECT: u32 = 0x300;
    const PARAM: u32 = 0x340;
    const COUNT: u32 = 0x3F0;
    const OUT: u32 = 0x400;
    const BUF: u32 = 0x600;
    const REC_A: u32 = 0x1000;
    const REC_B: u32 = 0x2000;
    const FRAME_US: u64 = 16_667;

    /// A guest in miniature: memory, the host state, and the guest's own touch clock.
    struct Guest {
        st: VitaState,
        mem: Vec<u8>,
        ts: u64,
    }

    impl Guest {
        fn new() -> Guest {
            let mut g = Guest {
                st: VitaState::new(0, MEM as u32, Box::new(Idle)),
                mem: vec![0xAAu8; MEM],
                ts: 1_000_000,
            };
            assert_eq!(g.call(init_primitive_touch_recognizer, &[0]), 0);
            g
        }

        fn call(&mut self, f: fn(&mut GuestCtx, &mut VitaState), args: &[u32]) -> i32 {
            let mut regs = [0u32; REG_COUNT];
            regs[..args.len().min(4)].copy_from_slice(&args[..args.len().min(4)]);
            regs[SP] = STACK;
            for (i, a) in args.iter().enumerate().skip(4) {
                let at = STACK as usize + 4 * (i - 4);
                self.mem[at..at + 4].copy_from_slice(&a.to_le_bytes());
            }
            let mut vfp = [0u32; VFP_ARG_COUNT];
            let mut mem = SliceMemory(&mut self.mem);
            let mut ctx = GuestCtx::new(&mut regs, &mut vfp, &mut mem, 0);
            f(&mut ctx, &mut self.st);
            regs[0] as i32
        }

        fn create(&mut self, at: u32, kind: i32, port: u32, rect: Option<(i16, i16, i16, i16)>, param: Option<[u8; 8]>) -> i32 {
            if let Some((x, y, w, h)) = rect {
                for (i, v) in [x, y, w, h].iter().enumerate() {
                    let o = RECT as usize + 2 * i;
                    self.mem[o..o + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            if let Some(p) = param {
                self.mem[PARAM as usize..PARAM as usize + 64].fill(0);
                self.mem[PARAM as usize..PARAM as usize + 8].copy_from_slice(&p);
            }
            let r = if rect.is_some() { RECT } else { 0 };
            let p = if param.is_some() { PARAM } else { 0 };
            self.call(create_touch_recognizer, &[at, kind as u32, port, r, p])
        }

        fn write_sample(&mut self, at: u32, ts: u64, points: &[(u8, i16, i16)]) {
            let mut b = [0u8; TOUCH_DATA_SIZE];
            b[0..8].copy_from_slice(&ts.to_le_bytes());
            b[12..16].copy_from_slice(&(points.len() as u32).to_le_bytes());
            for (j, &(id, x, y)) in points.iter().enumerate() {
                let o = TOUCH_REPORT_BASE + j * TOUCH_REPORT_SIZE;
                b[o] = id;
                b[o + 1] = 64;
                b[o + 2..o + 4].copy_from_slice(&x.to_le_bytes());
                b[o + 4..o + 6].copy_from_slice(&y.to_le_bytes());
            }
            self.mem[at as usize..at as usize + TOUCH_DATA_SIZE].copy_from_slice(&b);
        }

        /// One frame: the title's sequence - both panels' samples into the primitive layer,
        /// then every recognizer updated.
        fn frame(&mut self, front: &[(u8, i16, i16)], back: &[(u8, i16, i16)], recs: &[u32]) {
            self.ts += FRAME_US;
            let ts = self.ts;
            self.write_sample(FRONT, ts, front);
            self.write_sample(BACK, ts, back);
            assert_eq!(self.call(update_primitive_touch_recognizer, &[FRONT, BACK]), 0);
            for &r in recs {
                assert_eq!(self.call(update_touch_recognizer, &[r]), 0);
            }
        }

        /// Every event `rec` reports, through `GetTouchEvents`, as raw 0x90-byte elements.
        fn events(&mut self, rec: u32) -> Vec<Vec<u8>> {
            assert_eq!(self.call(get_touch_events, &[rec, BUF, 16, COUNT]), 0);
            let n = self.word(COUNT) as usize;
            assert_eq!(self.call(get_touch_events_count, &[rec]), n as i32);
            (0..n)
                .map(|i| {
                    let at = BUF as usize + i * TOUCH_EVENT_SIZE as usize;
                    self.mem[at..at + TOUCH_EVENT_SIZE as usize].to_vec()
                })
                .collect()
        }

        fn word(&self, at: u32) -> u32 {
            get_u32(&self.mem, at as usize)
        }
    }

    fn i16_at(b: &[u8], o: usize) -> i16 {
        get_u16(b, o) as i16
    }
    fn id(e: &[u8]) -> u32 {
        get_u32(e, 0)
    }
    fn state(e: &[u8]) -> i32 {
        get_u32(e, 4) as i32
    }

    /// A press and release is ONE tap, reported on the release only, carrying the position,
    /// a count of 1 and the primitive it came from - which the primitive getter still holds.
    #[test]
    fn a_tap_is_one_event_on_release() {
        let mut g = Guest::new();
        assert_eq!(g.create(REC_A, TYPE_TAP, 0, Some((0, 0, 1919, 1087)), None), 0);
        g.frame(&[(5, 500, 400)], &[], &[REC_A]);
        assert!(g.events(REC_A).is_empty(), "a finger still down is not a tap yet");
        g.frame(&[(5, 502, 401)], &[], &[REC_A]);
        assert!(g.events(REC_A).is_empty());
        g.frame(&[], &[], &[REC_A]);
        let ev = g.events(REC_A);
        assert_eq!(ev.len(), 1);
        let e = &ev[0];
        assert_ne!(id(e), 0);
        assert_eq!(state(e), TAP_EVENT_STATE);
        assert_eq!(get_u32(e, 8) as i32, TYPE_TAP);
        assert_eq!(get_u64(e, 16), g.ts, "updatedTime is the sample time");
        assert_eq!((i16_at(e, 26), i16_at(e, 28)), (502, 401));
        assert_eq!(e[30], 1, "tappedCount");
        assert!(e[32..].iter().all(|&b| b == 0), "the rest of the element is zeroed");
        let prim = get_u16(e, 24);
        assert_eq!(g.call(get_primitive_touch_event_by_primitive_id, &[u32::from(prim), OUT]), 0);
        assert_eq!(g.word(OUT) as i32, STATE_END);
        // By-index and repeated reads answer the same event; nothing is consumed by reading.
        assert_eq!(g.call(get_touch_event_by_index, &[REC_A, 0, OUT]), 0);
        assert_eq!(&g.mem[OUT as usize..OUT as usize + 0x90], &e[..]);
        assert_eq!(g.events(REC_A).len(), 1);
        assert_eq!(
            g.call(get_touch_event_by_index, &[REC_A, 1, OUT]),
            SCE_SYSTEM_GESTURE_ERROR_INDEX_OUT_OF_ARRAY
        );
        g.frame(&[], &[], &[REC_A]);
        assert!(g.events(REC_A).is_empty(), "a tap lives for one update");
        assert_eq!(
            g.call(get_primitive_touch_event_by_primitive_id, &[u32::from(prim), OUT]),
            SCE_SYSTEM_GESTURE_ERROR_EVENT_DATA_NOT_FOUND
        );
    }

    /// A press held past the tap time, or one that wanders past the slop, is no tap.
    #[test]
    fn a_long_or_moving_press_is_not_a_tap() {
        let mut g = Guest::new();
        assert_eq!(g.create(REC_A, TYPE_TAP, 0, None, None), 0);
        let frames = (TAP_MAX_DURATION_US / FRAME_US) as usize + 2;
        for _ in 0..frames {
            g.frame(&[(1, 100, 100)], &[], &[REC_A]);
        }
        g.frame(&[], &[], &[REC_A]);
        assert!(g.events(REC_A).is_empty());
        g.frame(&[(2, 100, 100)], &[], &[REC_A]);
        g.frame(&[(2, 100 + STATIONARY_SLOP as i16 + 1, 100)], &[], &[REC_A]);
        // Drifting back before the lift does not make it a tap again.
        g.frame(&[(2, 100, 100)], &[], &[REC_A]);
        g.frame(&[], &[], &[REC_A]);
        assert!(g.events(REC_A).is_empty());
    }

    /// With `maxTapCount = 2` a second quick tap close to the first reports tappedCount 2.
    #[test]
    fn a_double_tap_counts_two() {
        let mut g = Guest::new();
        assert_eq!(g.create(REC_A, TYPE_TAP, 0, None, Some([2, 0, 0, 0, 0, 0, 0, 0])), 0);
        g.frame(&[(1, 300, 300)], &[], &[REC_A]);
        g.frame(&[], &[], &[REC_A]);
        let first = g.events(REC_A);
        assert_eq!(first[0][30], 1);
        g.frame(&[], &[], &[REC_A]);
        g.frame(&[(2, 310, 305)], &[], &[REC_A]);
        g.frame(&[], &[], &[REC_A]);
        let second = g.events(REC_A);
        assert_eq!(second.len(), 1);
        assert_eq!(second[0][30], 2);
        assert_ne!(id(&first[0]), id(&second[0]), "each tap is its own event");
    }

    /// A drag is BEGIN on the press (a stationary finger included), ACTIVE with each update's
    /// delta, END on the release - all under ONE eventID - and then gone.
    #[test]
    fn a_drag_is_begin_active_end_under_one_id() {
        let mut g = Guest::new();
        assert_eq!(g.create(REC_A, TYPE_DRAG, 0, None, None), 0);
        g.frame(&[(3, 100, 100)], &[], &[REC_A]);
        let e = g.events(REC_A).remove(0);
        let eid = id(&e);
        assert_ne!(eid, 0);
        assert_eq!((state(&e), get_u32(&e, 8) as i32), (STATE_BEGIN, TYPE_DRAG));
        let drag = |e: &[u8]| {
            [26, 28, 30, 32, 34, 36].map(|o| i16_at(e, o))
        };
        assert_eq!(drag(&e), [0, 0, 100, 100, 100, 100], "a press begins the drag where it is");
        g.frame(&[(3, 130, 90)], &[], &[REC_A]);
        let e = g.events(REC_A).remove(0);
        assert_eq!((id(&e), state(&e)), (eid, STATE_ACTIVE));
        assert_eq!(drag(&e), [30, -10, 130, 90, 100, 100], "delta, current, pressed");
        let begun_at = get_u64(&e, 16);
        g.frame(&[(3, 150, 90)], &[], &[REC_A]);
        let e = g.events(REC_A).remove(0);
        assert_eq!((id(&e), state(&e)), (eid, STATE_ACTIVE));
        assert_eq!(drag(&e), [20, 0, 150, 90, 100, 100]);
        g.frame(&[(3, 150, 90)], &[], &[REC_A]);
        let still = g.events(REC_A).remove(0);
        assert_eq!((id(&still), state(&still)), (eid, STATE_ACTIVE));
        assert_eq!(drag(&still)[..2], [0, 0]);
        assert!(get_u64(&e, 16) > begun_at);
        g.frame(&[(3, 150, 90)], &[], &[REC_A]);
        assert_eq!(
            get_u64(&g.events(REC_A)[0], 16),
            get_u64(&still, 16),
            "an unchanged event keeps its updatedTime"
        );
        g.frame(&[], &[], &[REC_A]);
        let e = g.events(REC_A).remove(0);
        assert_eq!((id(&e), state(&e)), (eid, STATE_END));
        assert_eq!(g.call(get_touch_event_by_event_id, &[REC_A, eid, OUT]), 0);
        assert_eq!(id(&g.mem[OUT as usize..]), eid);
        g.frame(&[], &[], &[REC_A]);
        assert!(g.events(REC_A).is_empty());
        assert_eq!(
            g.call(get_touch_event_by_event_id, &[REC_A, eid, OUT]),
            SCE_SYSTEM_GESTURE_ERROR_EVENT_DATA_NOT_FOUND
        );
    }

    /// A hold fires BEGIN once the press has lasted the hold time (the observed parameter 30
    /// is read as thirty 60 Hz periods), then ACTIVE, then END on release.
    #[test]
    fn a_hold_begins_after_the_hold_time() {
        let mut g = Guest::new();
        assert_eq!(g.create(REC_A, TYPE_TAP_AND_HOLD, 1, None, Some(30u64.to_le_bytes())), 0);
        let hold = 30 * HOLD_TIME_TICK_US;
        g.frame(&[], &[(9, 800, 600)], &[REC_A]);
        let pressed = g.ts;
        while g.ts + FRAME_US < pressed + hold {
            g.frame(&[], &[(9, 800, 600)], &[REC_A]);
            assert!(g.events(REC_A).is_empty(), "not held long enough yet");
        }
        g.frame(&[], &[(9, 805, 600)], &[REC_A]);
        let e = g.events(REC_A).remove(0);
        assert_eq!((state(&e), get_u32(&e, 8) as i32), (STATE_BEGIN, TYPE_TAP_AND_HOLD));
        assert_eq!((i16_at(&e, 26), i16_at(&e, 28)), (800, 600), "pressedPosition");
        let eid = id(&e);
        g.frame(&[], &[(9, 805, 600)], &[REC_A]);
        assert_eq!(state(&g.events(REC_A)[0]), STATE_ACTIVE);
        g.frame(&[], &[], &[REC_A]);
        let e = g.events(REC_A).remove(0);
        assert_eq!((id(&e), state(&e)), (eid, STATE_END));
    }

    /// Two fingers make one pinch: BEGIN at scale 1, ACTIVE with the distance ratio, each
    /// record carrying the other finger as its pair.
    #[test]
    fn a_pinch_scales_by_finger_distance() {
        let mut g = Guest::new();
        assert_eq!(g.create(REC_A, TYPE_PINCH_OUT_IN, 0, None, None), 0);
        g.frame(&[(1, 900, 500)], &[], &[REC_A]);
        assert!(g.events(REC_A).is_empty(), "one finger is no pinch");
        g.frame(&[(1, 900, 500), (2, 1000, 500)], &[], &[REC_A]);
        let e = g.events(REC_A).remove(0);
        assert_eq!(state(&e), STATE_BEGIN);
        assert_eq!(f32::from_le_bytes(e[24..28].try_into().unwrap()), 1.0);
        g.frame(&[(1, 850, 500), (2, 1050, 500)], &[], &[REC_A]);
        let e = g.events(REC_A).remove(0);
        assert_eq!(state(&e), STATE_ACTIVE);
        assert_eq!(f32::from_le_bytes(e[24..28].try_into().unwrap()), 2.0);
        // primitive[0]: current (850,500), delta (-50,0), paired (1050,500).
        assert_eq!([30, 32, 34, 36, 38, 40].map(|o| i16_at(&e, o)), [850, 500, -50, 0, 1050, 500]);
        g.frame(&[(1, 850, 500)], &[], &[REC_A]);
        assert_eq!(state(&g.events(REC_A)[0]), STATE_END);
        g.frame(&[(1, 850, 500)], &[], &[REC_A]);
        assert!(g.events(REC_A).is_empty());
    }

    /// A recognizer only hears its own panel, and only touches that BEGIN inside its
    /// rectangle; moving the rectangle takes effect for the next press.
    #[test]
    fn rectangles_and_ports_filter() {
        let mut g = Guest::new();
        assert_eq!(g.create(REC_A, TYPE_TAP, 1, None, None), 0);
        assert_eq!(g.create(REC_B, TYPE_TAP, 0, Some((1000, 0, 100, 100)), None), 0);
        let tap = |g: &mut Guest, x: i16, y: i16| {
            g.frame(&[(1, x, y)], &[], &[REC_A, REC_B]);
            g.frame(&[], &[], &[REC_A, REC_B]);
        };
        tap(&mut g, 500, 400);
        assert!(g.events(REC_A).is_empty(), "a back-panel recognizer ignores the front");
        assert!(g.events(REC_B).is_empty(), "outside the rectangle");
        tap(&mut g, 1100, 100);
        assert_eq!(g.events(REC_B).len(), 1, "the far edge is inside");
        for (i, v) in [0i16, 0, 1919, 1087].iter().enumerate() {
            let o = RECT as usize + 2 * i;
            g.mem[o..o + 2].copy_from_slice(&v.to_le_bytes());
        }
        assert_eq!(g.call(update_touch_recognizer_rectangle, &[REC_B, RECT]), 0);
        tap(&mut g, 500, 400);
        assert_eq!(g.events(REC_B).len(), 1);
        assert_eq!(g.call(get_touch_recognizer_information, &[REC_B, OUT]), 0);
        let info = &g.mem[OUT as usize..OUT as usize + INFORMATION_SIZE];
        assert_eq!(get_u32(info, 0) as i32, TYPE_TAP);
        assert_eq!(get_u32(info, 4), 0);
        assert_eq!([8, 10, 12, 14].map(|o| i16_at(info, o)), [0, 0, 1919, 1087]);
        assert_eq!(get_u64(info, 16), g.ts);
    }

    /// GetTouchEvents copies at most `capacity` events, writes the number copied, and leaves
    /// the buffer past it alone.
    #[test]
    fn capacity_truncates() {
        let mut g = Guest::new();
        assert_eq!(g.create(REC_A, TYPE_DRAG, 0, None, None), 0);
        let fingers = |dx: i16| [(1, 100 + dx, 100), (2, 400 + dx, 100), (3, 700 + dx, 100)];
        g.frame(&fingers(0), &[], &[REC_A]);
        g.frame(&fingers(10), &[], &[REC_A]);
        assert_eq!(g.call(get_touch_events_count, &[REC_A]), 3);
        g.mem[BUF as usize..BUF as usize + 3 * 0x90].fill(0xAA);
        assert_eq!(g.call(get_touch_events, &[REC_A, BUF, 2, COUNT]), 0);
        assert_eq!(g.word(COUNT), 2);
        assert_eq!(i16_at(&g.mem, BUF as usize + 0x90 + 30), 410, "oldest first");
        assert_eq!(g.mem[BUF as usize + 2 * 0x90], 0xAA, "no third element");
        assert_eq!(
            g.call(get_touch_events, &[REC_A, BUF, 0, COUNT]),
            SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT
        );
        let ids: Vec<u32> = g.events(REC_A).iter().map(|e| id(e)).collect();
        assert_eq!(ids, [1, 2, 3], "one independent drag per finger");
    }

    /// The whole recognizer lives in its block: a byte copy taken mid-drag continues the
    /// same gesture under the same id, and Reset discards it.
    #[test]
    fn a_copied_work_area_carries_its_gesture() {
        let mut g = Guest::new();
        assert_eq!(g.create(REC_A, TYPE_DRAG, 0, None, None), 0);
        g.frame(&[(4, 100, 100)], &[], &[REC_A]);
        g.frame(&[(4, 120, 100)], &[], &[REC_A]);
        let eid = id(&g.events(REC_A)[0]);
        g.mem.copy_within(REC_A as usize..REC_A as usize + RECOGNIZER_SIZE, REC_B as usize);
        g.frame(&[(4, 140, 100)], &[], &[REC_B]);
        let e = g.events(REC_B).remove(0);
        assert_eq!((id(&e), state(&e)), (eid, STATE_ACTIVE));
        assert_eq!(g.call(reset_touch_recognizer, &[REC_B]), 0);
        assert!(g.events(REC_B).is_empty());
        assert_eq!(
            g.call(get_touch_event_by_event_id, &[REC_B, eid, OUT]),
            SCE_SYSTEM_GESTURE_ERROR_EVENT_DATA_NOT_FOUND
        );
    }

    /// A ROTATION recognizer is created and polled like any other, and - its detection not
    /// being modelled - reports no events: a count of ZERO, never an error a title could read
    /// as an unsigned count.
    #[test]
    fn a_rotation_recognizer_is_created_and_reports_no_events() {
        let mut g = Guest::new();
        assert_eq!(g.create(REC_A, TYPE_ROTATION, 0, None, None), 0);
        g.frame(&[(4, 140, 100)], &[], &[REC_A]);
        g.frame(&[(4, 300, 100), (5, 140, 300)], &[], &[REC_A]);
        assert_eq!(g.call(update_touch_recognizer, &[REC_A]), 0);
        assert_eq!(g.call(get_touch_events_count, &[REC_A]), 0);
        assert!(g.events(REC_A).is_empty());
    }

    /// The library's own error codes for the caller errors and the unknown.
    #[test]
    fn errors_are_the_librarys_own() {
        let mut g = Guest::new();
        assert_eq!(g.create(REC_A, 32, 0, None, None), SCE_SYSTEM_GESTURE_ERROR_NOT_SUPPORTED_GESTURE);
        assert_eq!(g.create(0, TYPE_TAP, 0, None, None), SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT);
        assert_eq!(g.call(update_touch_recognizer, &[0]), SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT);
        assert_eq!(
            g.call(get_touch_events_count, &[REC_B]),
            SCE_SYSTEM_GESTURE_ERROR_INVALID_ARGUMENT,
            "a block no Create initialized"
        );
        assert_eq!(g.create(REC_A, TYPE_TAP, 0, None, None), 0);
        assert_eq!(
            g.call(get_touch_event_by_event_id, &[REC_A, 77, OUT]),
            SCE_SYSTEM_GESTURE_ERROR_EVENT_DATA_NOT_FOUND
        );
        assert_eq!(
            g.call(get_primitive_touch_event_by_index, &[0, OUT]),
            SCE_SYSTEM_GESTURE_ERROR_INDEX_OUT_OF_ARRAY
        );
        assert_eq!(g.call(finalize_primitive_touch_recognizer, &[]), 0);
        assert_eq!(
            g.call(update_primitive_touch_recognizer, &[FRONT, BACK]),
            SCE_SYSTEM_GESTURE_ERROR_NOT_INITIALIZED
        );
        assert_eq!(g.call(update_touch_recognizer, &[REC_A]), SCE_SYSTEM_GESTURE_ERROR_NOT_INITIALIZED);
    }

    /// The primitive layer: a press is BEGIN, a held finger that moves is ACTIVE with its
    /// movement and time since, a lift is one END at the last position, then it is gone; a
    /// primitive's id is its report's id (unique across the panels, which share one counter).
    #[test]
    fn primitives_follow_the_samples() {
        let mut g = Guest::new();
        g.frame(&[(7, 100, 200)], &[(8, 50, 60)], &[]);
        assert_eq!(g.call(get_primitive_touch_events_count, &[]), 2);
        assert_eq!(g.call(get_primitive_touch_events, &[BUF, 8, COUNT]), 0);
        assert_eq!(g.word(COUNT), 2);
        let p0 = g.mem[BUF as usize..BUF as usize + 0x60].to_vec();
        let p1 = g.mem[BUF as usize + 0x60..BUF as usize + 0xC0].to_vec();
        assert_eq!(get_u32(&p0, 0) as i32, STATE_BEGIN);
        assert_eq!((get_u16(&p0, 4), get_u16(&p1, 4)), (7, 8), "each id is the panel report's own");
        assert_eq!([6, 8, 10, 12, 14, 16].map(|o| i16_at(&p0, o)), [100, 200, 64, 100, 200, 64]);
        g.frame(&[(7, 130, 190)], &[(8, 50, 60)], &[]);
        assert_eq!(g.call(get_primitive_touch_event_by_index, &[0, OUT]), 0);
        let p = g.mem[OUT as usize..OUT as usize + 0x60].to_vec();
        assert_eq!(get_u32(&p, 0) as i32, STATE_ACTIVE);
        assert_eq!([0x12, 0x14, 0x16].map(|o| i16_at(&p, o)), [30, -10, 0]);
        assert_eq!(get_u64(&p, 0x18), FRAME_US, "deltaTime");
        assert_eq!(get_u64(&p, 0x20), FRAME_US, "elapsedTime");
        g.frame(&[], &[(8, 50, 60)], &[]);
        assert_eq!(g.call(get_primitive_touch_event_by_index, &[0, OUT]), 0);
        let p = g.mem[OUT as usize..OUT as usize + 0x60].to_vec();
        assert_eq!(get_u32(&p, 0) as i32, STATE_END);
        assert_eq!([0x0C, 0x0E].map(|o| i16_at(&p, o)), [130, 190]);
        g.frame(&[], &[(8, 50, 60)], &[]);
        assert_eq!(g.call(get_primitive_touch_events_count, &[]), 1, "an END is reported once");
    }
}
