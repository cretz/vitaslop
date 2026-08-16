//! The determinism seam: the single small trait through which every
//! non-deterministic external input a Vita program can observe enters the
//! emulator. Handlers translate NID semantics and only ever ask `World` for
//! abstract inputs, so this trait stays small and stable as the NID surface
//! grows. See `projects/vitaslop-runtime/README.md`.
//!
//! Everything else (thread scheduling, allocation addresses) is made
//! deterministic by construction, so it never appears here.

/// One frame of controller state, port-agnostic. Buttons is the Vita
//  `SceCtrlButtons` bitmask; sticks are 0..255 with 128 as neutral.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtrlFrame {
    pub buttons: u32,
    pub lx: u8,
    pub ly: u8,
    pub rx: u8,
    pub ry: u8,
}

impl Default for CtrlFrame {
    /// No buttons held, sticks centered.
    fn default() -> Self {
        CtrlFrame { buttons: 0, lx: 128, ly: 128, rx: 128, ry: 128 }
    }
}

/// The most simultaneous touch points a panel reports (`SceTouchData.report[8]`).
/// A `TouchFrame` carries a fixed array of this size so the poll path never
/// allocates; only the first `count` are live.
pub const MAX_TOUCH_POINTS: usize = 8;

/// One active touch point, in PANEL coordinates (the front panel is 1920x1088,
/// twice the 960x544 screen; back is 1920x890). Mirrors the fields a title reads
/// out of a `SceTouchReport`: an id that is stable while a finger stays down, a
/// force, and the position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TouchPoint {
    pub id: u8,
    pub force: u8,
    pub x: u16,
    pub y: u16,
}

/// One frame of touch-panel state for a port: the set of points currently down.
/// The default is "no finger" (`count == 0`), which a handler reports as a valid
/// sample with `reportNum = 0` - deliberately not the same as "no buffers".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TouchFrame {
    pub points: [TouchPoint; MAX_TOUCH_POINTS],
    pub count: u8,
}

impl TouchFrame {
    /// A single finger held at panel `(x, y)` - the common menu/tap case. Force is
    /// the panel's typical full-press value and the id is 0 (a lone finger).
    pub fn single(x: u16, y: u16) -> Self {
        let mut f = TouchFrame::default();
        f.points[0] = TouchPoint { id: 0, force: 128, x, y };
        f.count = 1;
        f
    }

    /// The live points this frame.
    pub fn active(&self) -> &[TouchPoint] {
        &self.points[..self.count as usize]
    }
}

/// One position fix, exactly as a host location provider reports it.
///
/// # Why the optional fields are `Option` and not a sentinel
/// `SceLocationLocationInfo` marks a field it could not obtain with
/// `SCE_LOCATION_DATA_INVALID` (-9999.0), and the W3C Geolocation API - the browser's
/// provider - marks the same fields `null`. Carrying `Option` through the seam keeps
/// those two statements the SAME statement, and leaves the sentinel where it belongs:
/// in the handler that writes the guest struct. A provider that cannot see altitude
/// must not be able to spell it as a plausible number here.
///
/// Latitude and longitude are not optional: a fix without them is not a fix, and a
/// provider with nothing to say returns `None` from [`World::poll_location`] instead.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocationFix {
    /// Degrees, -90..=90.
    pub latitude_deg: f64,
    /// Degrees, -180..=180.
    pub longitude_deg: f64,
    /// Metres above the WGS84 ellipsoid.
    pub altitude_m: Option<f64>,
    /// Horizontal error in metres (the guest's `accuracy`).
    pub accuracy_m: Option<f32>,
    /// Direction of TRAVEL in degrees clockwise from true north. This is the guest's
    /// `direction` field and the browser's `heading`; it is a property of movement, not
    /// of which way the device is pointed, which is what `SceLocationHeadingInfo` is for.
    pub direction_deg: Option<f32>,
    /// Ground speed in metres per second.
    pub speed_mps: Option<f32>,
    /// When the fix was taken, in microseconds since the Unix epoch (UTC), matching the
    /// `SceRtcTick` the guest struct ends with.
    pub timestamp_us: u64,
}

/// What a host's location provider can currently tell the guest.
///
/// This models the two independent things the Vita API keeps separate and a naive
/// implementation collapses: whether a provider EXISTS at all (hardware), and whether
/// the user has PERMITTED this title to use it (the `sceLocationConfirm` dialog). A host
/// with no provider is not a user who said no, and telling a title the second when the
/// first is true invents a decision nobody made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocationPermission {
    /// This host has no location provider at all - the honest state for a desktop run
    /// with no GPS and no scripted fix. Handlers report the API's own
    /// `SCE_LOCATION_ERROR_PROVIDER_UNAVAILABLE` for it, the same shape
    /// [`crate::vita::camera`] uses for absent camera hardware.
    Unavailable,
    /// A provider exists and has not been asked for permission yet.
    NotAsked,
    /// Permission has been requested and the user has not answered. The browser's
    /// permission prompt is on screen; the guest sees its own dialog RUNNING.
    Pending,
    /// The user permitted it.
    Granted,
    /// The user refused.
    Denied,
}

/// `PositionError.PERMISSION_DENIED` from the W3C Geolocation API: the one code that
/// means the USER refused. The others (`POSITION_UNAVAILABLE` = 2, `TIMEOUT` = 3) mean
/// permission is fine and there is simply no fix - which is the Vita's
/// `SCE_LOCATION_INFO_UNDETERMINED_LOCATION`, not a refusal. Collapsing them would tell a
/// title the user declined when they did not.
pub const W3C_POSITION_ERROR_PERMISSION_DENIED: u16 = 1;

/// What a host location provider currently holds: the permission answer and the latest
/// fix, if any.
///
/// # Why this lives in the runtime and not in the browser crate
/// The rules below are statements about how the W3C Geolocation API maps onto the Vita's
/// location API - which of its error codes is a refusal, what a NaN heading means, when a
/// fix must be dropped. They are facts about the two APIs, not about web-sys bindings, so
/// they belong where they can be tested on any host. The browser crate wraps this in an
/// `Arc<Mutex<_>>` and supplies the DOM half only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HostLocation {
    pub permission: LocationPermission,
    pub fix: Option<LocationFix>,
}

impl Default for HostLocation {
    /// A run starts with a provider that EXISTS but has not been asked. Deliberately not
    /// [`LocationPermission::Unavailable`]: claiming a provider is absent before asking
    /// would report missing hardware on a device that has a GPS. `Unavailable` is set
    /// only when the platform really has no positioning API.
    fn default() -> Self {
        HostLocation { permission: LocationPermission::NotAsked, fix: None }
    }
}

impl HostLocation {
    /// Record a position from a W3C `Position`.
    ///
    /// `timestamp_ms` is `Position.timestamp` (milliseconds since the Unix epoch), which
    /// becomes the microsecond `SceRtcTick` the guest struct ends with; a negative or
    /// non-finite value becomes 0 rather than being wrapped into a nonsense tick.
    ///
    /// A non-finite optional component stays UNKNOWN. The browser reports `heading` and
    /// `speed` as NaN when the device is stationary (a direction of travel is undefined
    /// with no travel) and the spec allows null for a component the device cannot supply.
    /// Both become `None`, which reaches the guest as the API's INVALID sentinel - never
    /// as 0, which would be due north at a standstill.
    pub fn apply_w3c_fix(
        &mut self,
        latitude_deg: f64,
        longitude_deg: f64,
        altitude_m: Option<f64>,
        accuracy_m: Option<f64>,
        heading_deg: Option<f64>,
        speed_mps: Option<f64>,
        timestamp_ms: f64,
    ) {
        let finite = |v: Option<f64>| v.filter(|x| x.is_finite());
        let timestamp_us = if timestamp_ms.is_finite() && timestamp_ms >= 0.0 {
            (timestamp_ms * 1000.0) as u64
        } else {
            0
        };
        // A fix arriving IS the grant: a browser does not deliver a position to a page
        // the user refused.
        self.permission = LocationPermission::Granted;
        self.fix = Some(LocationFix {
            latitude_deg,
            longitude_deg,
            altitude_m: finite(altitude_m),
            accuracy_m: finite(accuracy_m).map(|v| v as f32),
            direction_deg: finite(heading_deg).map(|v| v as f32),
            speed_mps: finite(speed_mps).map(|v| v as f32),
            timestamp_us,
        });
    }

    /// Record a W3C `PositionError` by its `code`.
    pub fn apply_w3c_error(&mut self, code: u16) {
        if code == W3C_POSITION_ERROR_PERMISSION_DENIED {
            self.permission = LocationPermission::Denied;
        } else if self.permission == LocationPermission::Pending {
            // Permission is not the problem, so the dialog is settled - there is just no
            // position yet. Only promote from Pending: a prior refusal must not be undone
            // by a later timeout.
            self.permission = LocationPermission::Granted;
        }
        // Either way there is no current position. A refusal in particular must drop a
        // fix already delivered - a title must not keep reading a position it is no
        // longer permitted to see.
        self.fix = None;
    }

    /// The platform has no positioning API at all (an insecure origin, or a host without
    /// one). Distinct from a refusal: nothing was asked.
    pub fn set_unavailable(&mut self) {
        self.permission = LocationPermission::Unavailable;
        self.fix = None;
    }

    /// The permission prompt is up (or the answer is already known and the first callback
    /// is imminent). The guest reads this as a RUNNING dialog.
    pub fn set_pending(&mut self) {
        self.permission = LocationPermission::Pending;
    }
}

/// The external world a guest observes. The only source of non-determinism the
/// emulator admits. Implementations decide whether time is real, virtual, or
/// replayed, and whether the clocks move together.
pub trait World {
    /// Monotonic time in microseconds. Never goes backward. May be virtual
    /// (frame or instruction derived) so it is not a recorded input.
    fn monotonic_us(&mut self) -> u64;

    /// Wall-clock time in microseconds since the Unix epoch. A genuine external
    /// input, so a recording wrapper logs it.
    fn wall_us(&mut self) -> u64;

    /// Controller state for `port` this poll.
    fn poll_ctrl(&mut self, port: u32) -> CtrlFrame;

    /// Touch-panel state for `port` this poll (port 0 = front, 1 = back). The
    /// default is no finger down, so a time-free or pad-only world needs no touch
    /// source; worlds that script or capture touch override this.
    fn poll_touch(&mut self, _port: u32) -> TouchFrame {
        TouchFrame::default()
    }

    /// Whether this host has a location provider, and what the user has said about it.
    /// The default is [`LocationPermission::Unavailable`]: a world that has not been
    /// given a location source does not have one, and says so rather than reporting a
    /// denial nobody issued.
    fn location_permission(&mut self) -> LocationPermission {
        LocationPermission::Unavailable
    }

    /// Ask the host to begin acquiring position, raising the platform's own permission
    /// prompt if it has not been answered. Called from `sceLocationConfirm`, which is
    /// the guest's request to show exactly that dialog. Idempotent; a world with no
    /// provider ignores it.
    fn request_location(&mut self) {}

    /// Stop acquiring position (the guest closed its last handle). A world with no
    /// provider ignores it.
    fn release_location(&mut self) {}

    /// The latest position fix, or `None` when the provider has not produced one yet.
    /// `None` is a real console state - a Vita indoors reports
    /// `SCE_LOCATION_INFO_UNDETERMINED_LOCATION` for exactly this - so it is never
    /// substituted with a guessed position.
    fn poll_location(&mut self) -> Option<LocationFix> {
        None
    }

    /// Fill `buf` with entropy.
    fn fill_random(&mut self, buf: &mut [u8]);

    /// Notify the world that display frame `frame` (a flip just completed) is now
    /// current, so a frame-keyed input source (a scripted TAS recipe) can advance.
    /// The default ignores it - time-free and input-free worlds do not need it.
    fn set_frame(&mut self, _frame: u64) {}
}

/// A deterministic, input-free world: a virtual monotonic clock advanced only by
/// the host, a fixed wall epoch, no buttons, and a small seeded PRNG. This is the
/// default backing for bring-up and for replay-clean runs.
pub struct DeterministicWorld {
    monotonic_us: u64,
    wall_us: u64,
    rng: u64,
}

impl DeterministicWorld {
    /// A world starting at monotonic 0, the given wall epoch, and a PRNG seed.
    pub fn new(wall_epoch_us: u64, seed: u64) -> Self {
        DeterministicWorld { monotonic_us: 0, wall_us: wall_epoch_us, rng: seed | 1 }
    }

    /// Advance the virtual monotonic clock (e.g. one frame's worth). The host
    /// drives this so time is a pure function of progress, not wall-clock.
    pub fn advance_us(&mut self, delta_us: u64) {
        self.monotonic_us = self.monotonic_us.wrapping_add(delta_us);
        self.wall_us = self.wall_us.wrapping_add(delta_us);
    }
}

impl Default for DeterministicWorld {
    fn default() -> Self {
        // A fixed, arbitrary wall epoch so runs are reproducible by default.
        DeterministicWorld::new(1_500_000_000_000_000, 0x9E3779B97F4A7C15)
    }
}

impl World for DeterministicWorld {
    fn monotonic_us(&mut self) -> u64 {
        self.monotonic_us
    }
    fn wall_us(&mut self) -> u64 {
        self.wall_us
    }
    fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
        CtrlFrame::default()
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        // SplitMix64: deterministic and cheap. Not cryptographic, which is fine
        // for a replayable emulator entropy source.
        for chunk in buf.chunks_mut(8) {
            self.rng = self.rng.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            for (i, b) in chunk.iter_mut().enumerate() {
                *b = (z >> (i * 8)) as u8;
            }
        }
    }
}

/// One recorded non-deterministic answer, in call order. A `Record` wrapper
/// appends these over any inner world; a replay reads them back. This is the
/// answer-level log that makes bug-replay robust even in multi-worker mode.
/// Not `Eq`: [`LocationFix`] carries floats, which have no total equality. Every
/// comparison this enum is used for is an equality assertion in a test, which
/// `PartialEq` serves - and a recorded fix is compared against a byte-identical replay
/// of itself, never against a computed one, so no tolerance is wanted here either.
#[derive(Clone, Debug, PartialEq)]
pub enum WorldEvent {
    Monotonic(u64),
    Wall(u64),
    Ctrl { port: u32, frame: CtrlFrame },
    Touch { port: u32, frame: TouchFrame },
    Random(Vec<u8>),
    /// The permission answer, recorded because a user's grant or refusal is exactly the
    /// kind of external decision a replay must reproduce rather than re-ask.
    LocationPermission(LocationPermission),
    /// A position fix (or its absence), which is the most external input there is.
    Location(Option<LocationFix>),
}

/// Wraps any inner world and logs every answer it gives, in order. This is the
/// opt-in determinism trace: it captures exactly the values that crossed the
/// boundary, so a later replay reproduces the run without needing the inner
/// world at all.
pub struct Record<W: World> {
    inner: W,
    events: std::sync::Arc<std::sync::Mutex<Vec<WorldEvent>>>,
}

impl<W: World> Record<W> {
    pub fn new(inner: W) -> Self {
        Record { inner, events: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())) }
    }

    /// A shared handle to the recorded log. Clone it before boxing the recorder as
    /// the run's world, then read the events back after the run (the recorder
    /// itself is owned by the run). `Arc<Mutex>` rather than `Rc<RefCell>` so a
    /// recorder can back a `Send` world (the async scheduler needs `World: Send`);
    /// there is still only one thread, so the lock never contends.
    pub fn events(&self) -> std::sync::Arc<std::sync::Mutex<Vec<WorldEvent>>> {
        self.events.clone()
    }
}

impl<W: World> World for Record<W> {
    fn monotonic_us(&mut self) -> u64 {
        let v = self.inner.monotonic_us();
        self.events.lock().unwrap().push(WorldEvent::Monotonic(v));
        v
    }
    fn wall_us(&mut self) -> u64 {
        let v = self.inner.wall_us();
        self.events.lock().unwrap().push(WorldEvent::Wall(v));
        v
    }
    fn poll_ctrl(&mut self, port: u32) -> CtrlFrame {
        let frame = self.inner.poll_ctrl(port);
        self.events.lock().unwrap().push(WorldEvent::Ctrl { port, frame });
        frame
    }
    fn poll_touch(&mut self, port: u32) -> TouchFrame {
        let frame = self.inner.poll_touch(port);
        self.events.lock().unwrap().push(WorldEvent::Touch { port, frame });
        frame
    }
    fn location_permission(&mut self) -> LocationPermission {
        let v = self.inner.location_permission();
        self.events.lock().unwrap().push(WorldEvent::LocationPermission(v));
        v
    }
    fn request_location(&mut self) {
        // Not a recorded ANSWER - it is an outbound request, and the permission and fix
        // it leads to are each recorded where they are read. Recording it too would put
        // an event in the log that the replay has no call to consume.
        self.inner.request_location();
    }
    fn release_location(&mut self) {
        self.inner.release_location();
    }
    fn poll_location(&mut self) -> Option<LocationFix> {
        let v = self.inner.poll_location();
        self.events.lock().unwrap().push(WorldEvent::Location(v));
        v
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        self.inner.fill_random(buf);
        self.events.lock().unwrap().push(WorldEvent::Random(buf.to_vec()));
    }
    fn set_frame(&mut self, frame: u64) {
        // A frame tick is not a recorded answer (it is driven by the scheduler, which
        // is deterministic), but the inner world still needs it to advance input.
        self.inner.set_frame(frame);
    }
}

/// Replays an answer-level log. Ignores any real world: each call consumes the
/// next recorded event of the matching kind. Panics on log exhaustion or a kind
/// mismatch, which flags a divergence between record and replay.
pub struct Replay {
    events: std::collections::VecDeque<WorldEvent>,
}

impl Replay {
    pub fn new(events: Vec<WorldEvent>) -> Self {
        Replay { events: events.into() }
    }
    fn next(&mut self) -> WorldEvent {
        self.events.pop_front().expect("replay log exhausted")
    }
}

impl World for Replay {
    fn monotonic_us(&mut self) -> u64 {
        match self.next() {
            WorldEvent::Monotonic(v) => v,
            e => panic!("replay expected Monotonic, got {e:?}"),
        }
    }
    fn wall_us(&mut self) -> u64 {
        match self.next() {
            WorldEvent::Wall(v) => v,
            e => panic!("replay expected Wall, got {e:?}"),
        }
    }
    fn poll_ctrl(&mut self, port: u32) -> CtrlFrame {
        match self.next() {
            WorldEvent::Ctrl { port: p, frame } if p == port => frame,
            e => panic!("replay expected Ctrl(port={port}), got {e:?}"),
        }
    }
    fn poll_touch(&mut self, port: u32) -> TouchFrame {
        match self.next() {
            WorldEvent::Touch { port: p, frame } if p == port => frame,
            e => panic!("replay expected Touch(port={port}), got {e:?}"),
        }
    }
    fn location_permission(&mut self) -> LocationPermission {
        match self.next() {
            WorldEvent::LocationPermission(v) => v,
            e => panic!("replay expected LocationPermission, got {e:?}"),
        }
    }
    fn poll_location(&mut self) -> Option<LocationFix> {
        match self.next() {
            WorldEvent::Location(v) => v,
            e => panic!("replay expected Location, got {e:?}"),
        }
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        match self.next() {
            WorldEvent::Random(bytes) => {
                let n = buf.len().min(bytes.len());
                buf[..n].copy_from_slice(&bytes[..n]);
            }
            e => panic!("replay expected Random, got {e:?}"),
        }
    }
}

#[cfg(test)]
mod host_location_tests {
    //! How the W3C Geolocation API maps onto the Vita's. These are the rules a wrong
    //! answer would be invisible behind: a title told "the user refused" when they did
    //! not, or handed a heading of due north because the device was standing still.

    use super::*;

    /// A run starts with a provider that exists and has not been asked - not one reported
    /// absent, which would be a claim about the device.
    #[test]
    fn the_default_provider_is_present_but_unasked() {
        let l = HostLocation::default();
        assert_eq!(l.permission, LocationPermission::NotAsked);
        assert!(l.fix.is_none());
    }

    /// A fix arriving is itself the grant, and every component lands at its own type.
    #[test]
    fn a_fix_grants_permission_and_converts_every_component() {
        let mut l = HostLocation::default();
        l.apply_w3c_fix(35.5, 139.5, Some(41.5), Some(12.25), Some(97.5), Some(3.75), 1_700_000.0);
        assert_eq!(l.permission, LocationPermission::Granted);
        let f = l.fix.unwrap();
        assert_eq!(f.latitude_deg, 35.5);
        assert_eq!(f.longitude_deg, 139.5);
        assert_eq!(f.altitude_m, Some(41.5));
        assert_eq!(f.accuracy_m, Some(12.25));
        assert_eq!(f.direction_deg, Some(97.5));
        assert_eq!(f.speed_mps, Some(3.75));
        assert_eq!(f.timestamp_us, 1_700_000_000, "milliseconds become microseconds");
    }

    /// A stationary device reports NaN heading and speed. That is UNKNOWN, not zero -
    /// zero would be due north at a standstill, a measurement the device never made.
    #[test]
    fn a_nan_or_absent_component_stays_unknown_rather_than_zero() {
        let mut l = HostLocation::default();
        l.apply_w3c_fix(1.0, 2.0, None, Some(5.0), Some(f64::NAN), Some(f64::NAN), 0.0);
        let f = l.fix.unwrap();
        assert_eq!(f.direction_deg, None);
        assert_eq!(f.speed_mps, None);
        assert_eq!(f.altitude_m, None);
        assert_eq!(f.accuracy_m, Some(5.0), "a component that IS finite still lands");
    }

    /// Only PERMISSION_DENIED is a refusal. POSITION_UNAVAILABLE and TIMEOUT mean
    /// permission was fine and there is simply no fix.
    #[test]
    fn only_permission_denied_counts_as_a_refusal() {
        let mut denied = HostLocation::default();
        denied.set_pending();
        denied.apply_w3c_error(W3C_POSITION_ERROR_PERMISSION_DENIED);
        assert_eq!(denied.permission, LocationPermission::Denied);

        for code in [2u16, 3] {
            let mut l = HostLocation::default();
            l.set_pending();
            l.apply_w3c_error(code);
            assert_eq!(l.permission, LocationPermission::Granted, "code {code} is not a refusal");
            assert!(l.fix.is_none());
        }
    }

    /// A refusal drops a fix already delivered - a title must not keep reading a position
    /// it is no longer permitted to see.
    #[test]
    fn a_refusal_drops_a_fix_already_held() {
        let mut l = HostLocation::default();
        l.apply_w3c_fix(1.0, 2.0, None, None, None, None, 0.0);
        assert!(l.fix.is_some());
        l.apply_w3c_error(W3C_POSITION_ERROR_PERMISSION_DENIED);
        assert!(l.fix.is_none());
        assert_eq!(l.permission, LocationPermission::Denied);
    }

    /// A later timeout must not undo a refusal.
    #[test]
    fn a_timeout_after_a_refusal_does_not_grant_permission() {
        let mut l = HostLocation::default();
        l.apply_w3c_error(W3C_POSITION_ERROR_PERMISSION_DENIED);
        l.apply_w3c_error(3);
        assert_eq!(l.permission, LocationPermission::Denied);
    }

    /// A nonsense timestamp becomes 0 rather than a wrapped tick.
    #[test]
    fn a_non_finite_or_negative_timestamp_becomes_zero() {
        let mut l = HostLocation::default();
        for ts in [f64::NAN, f64::INFINITY, -5.0] {
            l.apply_w3c_fix(1.0, 2.0, None, None, None, None, ts);
            assert_eq!(l.fix.unwrap().timestamp_us, 0, "timestamp {ts}");
        }
    }

    /// An absent platform API is distinct from a refusal: nothing was asked.
    #[test]
    fn an_absent_api_is_unavailable_not_denied() {
        let mut l = HostLocation::default();
        l.apply_w3c_fix(1.0, 2.0, None, None, None, None, 0.0);
        l.set_unavailable();
        assert_eq!(l.permission, LocationPermission::Unavailable);
        assert!(l.fix.is_none());
    }
}
