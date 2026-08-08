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
    /// Reading a draw's index buffer out of guest memory.
    DrawIndices,
    /// Scanning indices for their min/max and rebasing them onto the snapshot.
    DrawIndexScan,
    /// Reading (and, for a multi-stream mesh, interleaving) the vertex bytes.
    DrawVertices,
    /// Snapshotting and decoding the draw's bound textures.
    DrawTextures,
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
    /// Standing up a new guest thread: with one wasm INSTANCE per thread, a spawn is a
    /// full module instantiation, not a cheap stack allocation. A title that runs its
    /// display-queue callback as a guest thread spawns one PER FLIP, so this can be a
    /// per-frame cost hiding in what looks like guest execution.
    ThreadSpawn,
}

impl Phase {
    const COUNT: usize = 9;

    fn index(self) -> usize {
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
        }
    }

    /// Every phase, in report order.
    pub fn all() -> [Phase; Phase::COUNT] {
        [
            Phase::DrawIndices,
            Phase::DrawIndexScan,
            Phase::DrawVertices,
            Phase::DrawTextures,
            Phase::DrawUniforms,
            Phase::SceneFold,
            Phase::DrawTextureCompare,
            Phase::SchedOverhead,
            Phase::ThreadSpawn,
        ]
    }

    /// Short label for a report line.
    pub fn label(self) -> &'static str {
        match self {
            Phase::DrawIndices => "draw: read indices",
            Phase::DrawIndexScan => "draw: scan/rebase indices",
            Phase::DrawVertices => "draw: read+interleave vertices",
            Phase::DrawTextures => "draw: snapshot textures",
            Phase::DrawUniforms => "draw: uniforms + material",
            Phase::SceneFold => "scene: signature fold",
            Phase::DrawTextureCompare => "draw: texture snapshot compare",
            Phase::SchedOverhead => "scheduler: pick + drain",
            Phase::ThreadSpawn => "scheduler: spawn thread (instantiate)",
        }
    }
}

static NS: [AtomicU64; Phase::COUNT] = [const { AtomicU64::new(0) }; Phase::COUNT];
static HITS: [AtomicU64; Phase::COUNT] = [const { AtomicU64::new(0) }; Phase::COUNT];
static BYTES: [AtomicU64; Phase::COUNT] = [const { AtomicU64::new(0) }; Phase::COUNT];

/// Charge `n` bytes of guest memory actually MOVED to `phase`.
///
/// Time alone cannot tell a phase that is slow from a phase that is merely large: a
/// millisecond spent copying is a volume problem (copy less) and a millisecond spent
/// per call is an overhead problem (call less), and the two have no fix in common.
/// Only counted while timing is on, so the report never mixes a measured window with
/// a stale total.
pub fn note_bytes(phase: Phase, n: usize) {
    if enabled() {
        BYTES[phase.index()].fetch_add(n as u64, Relaxed);
    }
}

/// Is phase timing on (`VITASLOP_PERF` set)? Read once and cached.
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VITASLOP_PERF").is_some())
}

/// Time `f` and charge it to `phase`. Returns whatever `f` returns, so it wraps an
/// expression in place without restructuring the caller.
///
/// When timing is off this is `f()` and one atomic load.
pub fn time<T>(phase: Phase, f: impl FnOnce() -> T) -> T {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = phase;
        f()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if !enabled() {
            return f();
        }
        let t = std::time::Instant::now();
        let out = f();
        let i = phase.index();
        NS[i].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        HITS[i].fetch_add(1, Relaxed);
        out
    }
}

/// Charge everything until the returned guard drops to `phase`.
///
/// The counterpart of [`time`] for a region that spans several statements and mutates
/// its surroundings, where wrapping it in a closure would mean restructuring the code
/// to measure it. `None` (and no clock read) when timing is off.
pub fn scope(phase: Phase) -> Option<Scope> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = phase;
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        enabled().then(|| Scope { phase, start: std::time::Instant::now() })
    }
}

/// Charges its phase when dropped. See [`scope`].
#[cfg(not(target_arch = "wasm32"))]
pub struct Scope {
    phase: Phase,
    start: std::time::Instant,
}

/// A `wasm32` build never constructs one (there is no clock), but the type must exist
/// for [`scope`]'s signature to be the same on both targets.
#[cfg(target_arch = "wasm32")]
pub struct Scope {
    _phase: Phase,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Scope {
    fn drop(&mut self) {
        let i = self.phase.index();
        NS[i].fetch_add(self.start.elapsed().as_nanos() as u64, Relaxed);
        HITS[i].fetch_add(1, Relaxed);
    }
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
    }
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
