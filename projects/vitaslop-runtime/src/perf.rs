//! Named phase timers for the inside of a host call.
//!
//! The engine-side profiler (`vitaslop-native`'s `perf`) splits a frame into
//! marshalling, handler and remainder, and attributes the handler bucket per NID.
//! That is enough to say "the draw handler costs 5 ms a frame" and no further: a
//! `sceGxmDraw` copies index bytes, scans and rebases them, copies and interleaves
//! vertex bytes, decodes bound textures and reads uniforms, and those have entirely
//! different fixes. This module times those phases by name so the next optimisation
//! is chosen rather than guessed.
//!
//! Gated on `VITASLOP_PERF`, cached in a `OnceLock`, so an ordinary run pays one
//! relaxed atomic load per phase and no clock read at all. On `wasm32` there is no
//! `std::time::Instant` (it panics), so the timers compile to nothing and the
//! counters stay zero - the browser host measures with its own clock.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;

/// The phases worth separating. Deliberately a closed enum rather than string keys:
/// a phase is a decision about what to measure, and adding one should be a visible
/// edit here, not an ad-hoc literal at a call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// The WHOLE of `record_draw`, every other `Draw*` phase nested inside it. The outer
    /// bracket exists so the per-draw capture cost has one number, and so the word reads
    /// that fall in no INNER phase can be attributed to the draw path rather than left in
    /// the difference between a table and a total.
    DrawTotal,
    /// Reading a draw's index buffer out of guest memory.
    DrawIndices,
    /// Scanning indices for their min/max and rebasing them onto the snapshot.
    DrawIndexScan,
    /// Reading (and, for a multi-stream mesh, interleaving) the vertex bytes.
    DrawVertices,
    /// Snapshotting and decoding the draw's bound textures.
    DrawTextures,
    /// The FRAGMENT stage's miss path alone: `snapshot_bound_textures` for a binding list
    /// `snapshot_sets` did not already hold. Entered only on a miss, so its `entries` count
    /// IS the per-frame miss count - which is the number that says whether the fix is a
    /// better cache or a cheaper decode.
    DrawTexFragMiss,
    /// The `get_or_read` inside a texture decode: the snapshot-cache lookup, the
    /// currency check and, when it fails, the re-read of the pixels. Nested inside
    /// [`Phase::DrawTexFragMiss`], so what that phase costs BEYOND this is the decode and
    /// the per-draw allocation around it - and those two have different fixes.
    DrawTexRead,
    /// The VERTEX stage's whole texture block, hit and miss together. Split out because it
    /// is skipped entirely by a draw that binds no vertex texture, so its `entries` count
    /// says how many draws pay it at all - and [`Phase::DrawTextures`] minus these two is
    /// what EVERY draw pays for having textures looked at.
    DrawTexVertex,
    /// Reading the sampler block out of the guest's context, decoding it into bindings,
    /// hashing them and looking the finished list up in `snapshot_sets`.
    ///
    /// Split from [`Phase::DrawTextures`] because it is the part EVERY draw pays even when
    /// the cache hits, while the rest is the miss path. A phase whose hit and miss costs
    /// are added together names neither fix: "cache it harder" and "make the gate cheaper"
    /// are different changes, and the ratio between these two decides which one is worth
    /// anything.
    DrawTextureBind,
    /// Snapshotting the raw shader blobs and the SA uniform BYTES for the GXP->WGSL path.
    /// The blobs are cached per header; the fragment SA bytes are a fresh allocation and
    /// copy on EVERY draw, and this runs whenever `VITASLOP_GXP_LIVE` is set - which is
    /// every browser run, and every run that renders the real shaders.
    DrawGxpCapture,
    /// Assembling the `capture::Draw` record and pushing it into the scene: the struct is
    /// built per draw and owns several `Vec`s and `Arc`s, so this is where per-draw
    /// ALLOCATION lands. Allocation is markedly dearer in wasm than native, so a cost that
    /// rounds to nothing on the desktop profile need not round to nothing on the engine
    /// that ships.
    DrawRecord,
    /// Reading the default uniform buffers and reflecting the material.
    DrawUniforms,
    /// Folding a completed scene into the determinism signature.
    SceneFold,
    /// The scheduler's OWN work per round - picking the next thread, draining the
    /// spawns/wakes a host call queued, and handling an idle tick. Deliberately NOT
    /// the resume itself: resuming runs the guest, so timing it would measure the
    /// guest. This is the part of a frame that is neither guest code nor a host call.
    /// The equality COMPARE that decides a retained texture snapshot is still current.
    /// Split out from [`Phase::DrawTextures`] because it is invisible in that phase's byte
    /// counter: `note_bytes` there counts bytes RE-READ, and the whole point of the compare
    /// is that it usually re-reads nothing - so a phase costing 44% of a race frame
    /// reported "0.0 MB/frame" and read as pure overhead when it is a memcmp of every
    /// bound texture, once per scene.
    DrawTextureCompare,
    SchedOverhead,
    /// Refreshing the HOST MIRROR block before a resume - the gathered snapshot and the one
    /// write of it into guest memory. Nested inside [`Phase::SchedOverhead`], because it is
    /// the only part of a pick that touches guest memory and the rest is a scan of a
    /// thirteen-entry table.
    SchedMirror,
    /// The scheduler's IDLE step - nothing was runnable, so the clock is jumped to the next
    /// deadline and the waiters it passes are woken. Split from [`Phase::SchedOverhead`]
    /// because an idle round and a busy one are different work with different fixes, and the
    /// browser charged both to one number.
    SchedIdle,
    /// The bookkeeping AFTER a resume: charging the guest clock for the work it did, blocking
    /// or cooling the thread, and draining the spawns and wakes its host calls queued.
    SchedBook,
    /// What a display FLIP costs outside the render: closing the capture's frame, advancing
    /// the game clock and the modelled I/O by one frame, and waking everything those pass.
    /// It happens on a scheduler round, so it was charged to the scheduler - a per-FRAME cost
    /// filed under a per-ROUND phase, which is how it stayed invisible.
    FrameBoundary,
    /// Standing up a new guest thread: with one wasm INSTANCE per thread, a spawn is a
    /// full module instantiation, not a cheap stack allocation. A title that runs its
    /// display-queue callback as a guest thread spawns one PER FLIP, so this can be a
    /// per-frame cost hiding in what looks like guest execution.
    ThreadSpawn,
}

impl Phase {
    const COUNT: usize = 20;

    pub(crate) fn index(self) -> usize {
        match self {
            Phase::DrawIndices => 0,
            Phase::DrawIndexScan => 1,
            Phase::DrawVertices => 2,
            Phase::DrawTextures => 3,
            Phase::DrawUniforms => 4,
            Phase::SceneFold => 5,
            Phase::DrawTextureCompare => 6,
            Phase::SchedOverhead => 7,
            Phase::ThreadSpawn => 8,
            Phase::DrawTextureBind => 9,
            Phase::DrawGxpCapture => 10,
            Phase::DrawRecord => 11,
            Phase::DrawTexFragMiss => 12,
            Phase::DrawTexVertex => 13,
            Phase::DrawTotal => 14,
            Phase::DrawTexRead => 15,
            Phase::SchedMirror => 19,
            Phase::SchedIdle => 16,
            Phase::SchedBook => 17,
            Phase::FrameBoundary => 18,
        }
    }

    /// Every phase, in report order.
    pub fn all() -> [Phase; Phase::COUNT] {
        [
            Phase::DrawTotal,
            Phase::DrawIndices,
            Phase::DrawIndexScan,
            Phase::DrawVertices,
            Phase::DrawTextureBind,
            Phase::DrawTextures,
            Phase::DrawTexFragMiss,
            Phase::DrawTexRead,
            Phase::DrawTexVertex,
            Phase::DrawUniforms,
            Phase::DrawGxpCapture,
            Phase::DrawRecord,
            Phase::SceneFold,
            Phase::DrawTextureCompare,
            Phase::SchedOverhead,
            Phase::SchedMirror,
            Phase::SchedIdle,
            Phase::SchedBook,
            Phase::FrameBoundary,
            Phase::ThreadSpawn,
        ]
    }

    /// Short label for a report line.
    pub fn label(self) -> &'static str {
        match self {
            Phase::DrawTotal => "DRAW TOTAL (everything below nests in it)",
            Phase::DrawIndices => "draw: read indices",
            Phase::DrawIndexScan => "draw: scan/rebase indices",
            Phase::DrawVertices => "draw: read+interleave vertices",
            Phase::DrawTextures => "draw: snapshot textures (miss path)",
            Phase::DrawTexFragMiss => "draw:   ...fragment MISS decode",
            Phase::DrawTexRead => "draw:     ...of which get_or_read",
            Phase::DrawTexVertex => "draw:   ...vertex stage (hit+miss)",
            Phase::DrawTextureBind => "draw: decode texture bindings (EVERY draw)",
            Phase::DrawGxpCapture => "draw: gxp blob + SA bytes",
            Phase::DrawRecord => "draw: build record + push scene",
            Phase::DrawUniforms => "draw: uniforms + material",
            Phase::SceneFold => "scene: signature fold",
            Phase::DrawTextureCompare => "draw: texture snapshot compare",
            Phase::SchedOverhead => "scheduler: pick",
            Phase::SchedMirror => "scheduler:   ...of which mirror refresh",
            Phase::SchedIdle => "scheduler: idle step",
            Phase::SchedBook => "scheduler: post-resume bookkeeping",
            Phase::FrameBoundary => "flip: end frame + advance clocks",
            Phase::ThreadSpawn => "scheduler: spawn thread (instantiate)",
        }
    }
}

static NS: [AtomicU64; Phase::COUNT] = [const { AtomicU64::new(0) }; Phase::COUNT];
static HITS: [AtomicU64; Phase::COUNT] = [const { AtomicU64::new(0) }; Phase::COUNT];
static BYTES: [AtomicU64; Phase::COUNT] = [const { AtomicU64::new(0) }; Phase::COUNT];

/// >>> HOW MANY TIMES THE HOST REACHED INTO GUEST MEMORY, split by whether it took one
/// word or a block.
///
/// A `dyn GuestMemory` access is a bounds check and a VIRTUAL CALL, and in the browser it
/// crosses into a `SharedArrayBuffer` view. The bytes are never the cost; the CALLS are.
/// A loop that reads a forty-word structure one word at a time and a single bulk read of
/// the same forty words move identical bytes and differ by 40x in cost, so neither a timer
/// nor the byte counters above can tell them apart - MEASURED, one such loop was 13.7% of
/// a browser frame and another 41-word one sat next to it.
///
/// So: count the accesses. `word_reads` per DRAW is the number that names this defect
/// class on sight, and it needs no clock, which means it is comparable between engines and
/// between runs. Counted only when timing is on, since a run that is not measuring should
/// not pay even an atomic.
static WORD_READS: AtomicU64 = AtomicU64::new(0);
static BULK_READS: AtomicU64 = AtomicU64::new(0);

/// One single-word read of guest memory through the `dyn GuestMemory` boundary.
pub fn note_word_read() {
    if enabled() {
        WORD_READS.fetch_add(1, Relaxed);
    }
}

/// One bulk read (a borrow, or a copy of a whole structure) of guest memory.
pub fn note_bulk_read() {
    if enabled() {
        BULK_READS.fetch_add(1, Relaxed);
    }
}

/// >>> HOW OFTEN THE GUEST-STORE EPOCH WRAPPED, which is a per-FRAME cliff and not a rate.
///
/// The epoch is ONE BYTE ([`crate::host::GuestMemory::bump_dirty_epoch`]), advanced once per
/// SCENE. A race frame is eleven scenes and the browser runs more than one guest frame per
/// present, so the 253 usable values are spent in about ten presented frames - and every wrap
/// zeroes the map and drops every stamp, so the NEXT use of each retained snapshot pays a full
/// copy-and-compare of the whole texture working set. That is a periodic hitch, not a steady
/// cost, and a per-frame mean hides it completely.
///
/// Counted unconditionally: it is a handful of increments over a run, and a cliff nobody can
/// see is how the compare it causes was written up as per-draw overhead.
static EPOCH_WRAPS: AtomicU64 = AtomicU64::new(0);

/// Epoch RENUMBERINGS - a wrap avoided by reclaiming the unused low half of the range.
/// Counted beside the wraps so the two read as the alternatives they are.
static EPOCH_REBASES: AtomicU64 = AtomicU64::new(0);

/// One epoch renumbering. See [`EPOCH_REBASES`].
pub fn note_epoch_rebase() {
    EPOCH_REBASES.fetch_add(1, Relaxed);
}

/// Epoch renumberings since the run started.
pub fn epoch_rebases() -> u64 {
    EPOCH_REBASES.load(Relaxed)
}

/// One guest-store epoch wrap. See [`EPOCH_WRAPS`].
pub fn note_epoch_wrap() {
    EPOCH_WRAPS.fetch_add(1, Relaxed);
}

/// Guest-store epoch wraps since the run started (cumulative - a caller reporting per frame
/// differences it).
pub fn epoch_wraps() -> u64 {
    EPOCH_WRAPS.load(Relaxed)
}

/// `(single-word reads, bulk reads)` since the last [`reset`].
pub fn guest_accesses() -> (u64, u64) {
    (WORD_READS.load(Relaxed), BULK_READS.load(Relaxed))
}

/// >>> WHICH PHASE THE WORD READS HAPPENED IN, because the total names the DEFECT CLASS and
/// not the line to fix.
///
/// The totals above said a a retail race frame took **27,107 single-word reads
/// over 560 draws**
/// - forty-eight per draw, after the four largest per-word readers had already been converted
/// to bulk reads. A number like that is a search, and the search was being done by reading
/// code. Every phase is already bracketed by a [`Scope`], so charging the counter's DELTA over
/// a scope costs one load per scope and turns the search into a table.
///
/// **Nested scopes DOUBLE COUNT** - an inner phase's reads are charged to the outer one too,
/// exactly as the nanoseconds already are. Read a child's share out of its parent, do not sum
/// the column. And a read outside every scope is in no row at all, which is why the total is
/// reported beside the table rather than derived from it.
static WORD_BY_PHASE: [AtomicU64; Phase::COUNT] = [const { AtomicU64::new(0) }; Phase::COUNT];

/// Single-word guest reads charged to `phase` since the last [`reset`]. See
/// [`WORD_BY_PHASE`] for how to read it.
pub fn word_reads(phase: Phase) -> u64 {
    WORD_BY_PHASE[phase.index()].load(Relaxed)
}

/// Charge `n` bytes of guest memory actually MOVED to `phase`.
///
/// Time alone cannot tell a phase that is slow from a phase that is merely large: a
/// millisecond spent copying is a volume problem (copy less) and a millisecond spent
/// per call is an overhead problem (call less), and the two have no fix in common.
///
/// # >>> COUNTED ALWAYS, AND ON EVERY TARGET, WHICH IS THE POINT
/// This used to be gated on [`enabled`], i.e. on `VITASLOP_PERF`, which is an ENVIRONMENT
/// variable - and the browser has no environment ([[vitaslop-browser-has-no-env]]). Timing is
/// gated too and is `#[cfg]`-ed out of wasm entirely, because `std::time::Instant` does not work
/// there. So the ONE engine whose CPU cost decides this project could not report a single phase
/// figure of any kind, and a device capture showed only two totals with nothing between them.
///
/// That is how a defect that spent **44% of every frame comparing 105.8 MB of texture** survived:
/// it was invisible on the engine that paid for it and switched off on the engine that did not.
/// A byte count needs no clock, and there are a handful of these calls per draw against thousands
/// of operations, so it is counted unconditionally on every target. **Volume is a measurement in
/// its own right** - the 105.8 MB would have named the defect on its own, with no timer at all.
pub fn note_bytes(phase: Phase, n: usize) {
    BYTES[phase.index()].fetch_add(n as u64, Relaxed);
}

/// Bytes charged to each phase since the last [`reset`] or [`take_bytes`], and zero them.
///
/// Taken rather than read so a caller can report PER FRAME without keeping its own baseline -
/// a running total looks like a per-frame figure that keeps growing, which reads as a leak.
/// >>> PAIRED WITH ITS PHASE, BECAUSE AN ARRAY INDEXED ONE WAY AND READ THE OTHER MISLABELS
/// EVERY ROW.
///
/// The counters are indexed by [`Phase::index`], which is the order variants were ADDED;
/// [`Phase::all`] returns them in REPORT order, which is the order they read well in. The one
/// caller zipped `all()` against the raw array, so every byte figure the browser has ever
/// printed carried its neighbour's name - a phase moving 5.4 MB a frame was filed under a
/// phase that moves none. Returning the pair removes the chance to get it wrong.
pub fn take_bytes() -> [(Phase, u64); Phase::COUNT] {
    let mut taken = [0u64; Phase::COUNT];
    for (i, slot) in taken.iter_mut().enumerate() {
        *slot = BYTES[i].swap(0, Relaxed);
    }
    Phase::all().map(|p| (p, taken[p.index()]))
}

/// Is phase timing on (`VITASLOP_PERF`)? Read once and cached.
///
/// Routed through [`crate::knobs`] rather than reading the environment directly, because
/// the browser HAS no environment ([[vitaslop-browser-has-no-env]]) and this is now the
/// gate on the browser's phase timers too. That also makes `VITASLOP_PERF=0` mean OFF,
/// which a bare `var_os(..).is_some()` did not.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::knobs::flag("VITASLOP_PERF"))
}

/// Time `f` and charge it to `phase`. Returns whatever `f` returns, so it wraps an
/// expression in place without restructuring the caller.
///
/// When timing is off this is `f()` and one atomic load.
pub fn time<T>(phase: Phase, f: impl FnOnce() -> T) -> T {
    // One implementation for both targets: `scope` already carries the per-target clock,
    // and two copies of this drifted once already - wasm's was a bare `f()`, so a phase
    // wrapped with `time` rather than `scope` was silently untimed on the engine that
    // needed it most.
    let _guard = scope(phase);
    f()
}

/// Charge everything until the returned guard drops to `phase`.
///
/// The counterpart of [`time`] for a region that spans several statements and mutates
/// its surroundings, where wrapping it in a closure would mean restructuring the code
/// to measure it. `None` (and no clock read) when timing is off.
pub fn scope(phase: Phase) -> Option<Scope> {
    if !enabled() {
        return None;
    }
    #[cfg(target_arch = "wasm32")]
    {
        // The clock is OPTIONAL here, and that is deliberate: this used to `?` out when no
        // host had installed one, which silently disabled the COUNTERS as well as the timer -
        // and the browser spent a session with no clock in its run worker and an empty phase
        // table that read as "no phase costs anything".
        Some(Scope { phase, start: clock(), words: WORD_READS.load(Relaxed) })
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        Some(Scope { phase, start: std::time::Instant::now(), words: WORD_READS.load(Relaxed) })
    }
}

/// Charges its phase when dropped. See [`scope`].
#[cfg(not(target_arch = "wasm32"))]
pub struct Scope {
    phase: Phase,
    start: std::time::Instant,
    words: u64,
}

/// On `wasm32` the start is a MILLISECOND reading from the host's installed clock (see
/// [`set_clock`]) rather than an `Instant`, which does not exist there.
#[cfg(target_arch = "wasm32")]
pub struct Scope {
    phase: Phase,
    start: Option<f64>,
    words: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Scope {
    fn drop(&mut self) {
        let i = self.phase.index();
        NS[i].fetch_add(self.start.elapsed().as_nanos() as u64, Relaxed);
        HITS[i].fetch_add(1, Relaxed);
        WORD_BY_PHASE[i].fetch_add(WORD_READS.load(Relaxed).saturating_sub(self.words), Relaxed);
    }
}

#[cfg(target_arch = "wasm32")]
impl Drop for Scope {
    fn drop(&mut self) {
        let i = self.phase.index();
        if let (Some(start), Some(now)) = (self.start, clock()) {
            // Saturating at zero: a coarsened clock can read the same value twice, and a
            // negative elapsed cast to u64 would be an eighteen-quintillion-nanosecond
            // phase - a number that looks like a defect in the thing being measured.
            let ns = ((now - start) * 1.0e6).max(0.0) as u64;
            NS[i].fetch_add(ns, Relaxed);
        }
        HITS[i].fetch_add(1, Relaxed);
        WORD_BY_PHASE[i].fetch_add(WORD_READS.load(Relaxed).saturating_sub(self.words), Relaxed);
    }
}

/// >>> THE HOST'S CLOCK, so `wasm32` can time a phase at all.
///
/// There is no `std::time::Instant` on `wasm32-unknown-unknown`, so this module could
/// count bytes on every target but could only ever time on native - and the engine whose
/// CPU cost decides this project is the browser. The result was that a browser frame
/// reported ONE undifferentiated number, and every browser A/B of a change to one part of
/// it was diluted by all the other parts before anyone read it.
///
/// The runtime stays engine-agnostic: it takes a monotonic millisecond clock as a plain
/// `fn` pointer and the host installs one. The browser passes `performance.now()`.
///
/// Idempotent - a second call is ignored, so a host that initialises twice cannot end up
/// with two clocks whose epochs differ.
#[cfg(target_arch = "wasm32")]
static CLOCK: OnceLock<fn() -> f64> = OnceLock::new();

/// Install the host's monotonic millisecond clock. See [`CLOCK`]. A no-op on native,
/// which has `Instant`, so a host may call it unconditionally.
#[cfg(target_arch = "wasm32")]
pub fn set_clock(f: fn() -> f64) {
    let _ = CLOCK.set(f);
}

/// Install the host's monotonic millisecond clock. Native times with `Instant` and ignores
/// this, so the same host code compiles for both.
#[cfg(not(target_arch = "wasm32"))]
pub fn set_clock(_f: fn() -> f64) {}

#[cfg(target_arch = "wasm32")]
fn clock() -> Option<f64> {
    CLOCK.get().map(|f| f())
}

/// Accumulated `(nanoseconds, times entered, bytes moved)` for `phase` since the last
/// [`reset`]. Bytes are zero for a phase that does not report them.
pub fn read(phase: Phase) -> (u64, u64, u64) {
    let i = phase.index();
    (NS[i].load(Relaxed), HITS[i].load(Relaxed), BYTES[i].load(Relaxed))
}

/// Zero every phase counter, so a benchmark measures its window and not the boot.
pub fn reset() {
    for i in 0..Phase::COUNT {
        NS[i].store(0, Relaxed);
        HITS[i].store(0, Relaxed);
        BYTES[i].store(0, Relaxed);
        WORD_BY_PHASE[i].store(0, Relaxed);
    }
    WORD_READS.store(0, Relaxed);
    BULK_READS.store(0, Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `time` returns the closure's value whether or not timing is on - the property
    /// every call site depends on, since it wraps expressions in place.
    #[test]
    fn passes_the_value_through() {
        assert_eq!(time(Phase::DrawIndices, || 41 + 1), 42);
    }

    /// A reset zeroes the counters, and reading an untouched phase is zero rather
    /// than a panic.
    #[test]
    fn reset_zeroes() {
        note_bytes(Phase::SceneFold, 1234);
        reset();
        assert_eq!(read(Phase::SceneFold), (0, 0, 0));
    }

    /// Every phase has a distinct slot; a duplicated index would silently merge two
    /// measurements into one and misattribute the cost.
    #[test]
    fn phase_indices_are_distinct() {
        let mut seen: Vec<usize> = Phase::all().iter().map(|p| p.index()).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), Phase::COUNT);
    }
}
