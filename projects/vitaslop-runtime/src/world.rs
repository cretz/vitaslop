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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorldEvent {
    Monotonic(u64),
    Wall(u64),
    Ctrl { port: u32, frame: CtrlFrame },
    Touch { port: u32, frame: TouchFrame },
    Random(Vec<u8>),
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
