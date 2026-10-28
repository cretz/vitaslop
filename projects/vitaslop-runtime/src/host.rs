//! The host-call boundary: how a guest NID import trap becomes a typed Rust
//! handler. `GuestCtx` marshals AAPCS arguments (r0..r3 then stack) and guest
//! memory in and out; `VitaEnv` owns the per-run state (allocator, handles,
//! capture, world) and routes a dense import index to a per-module handler.
//! See `projects/vitaslop-runtime/README.md`.

// The per-draw maps below are hashed with FxHash rather than SipHash. See `crate::fasthash` for
// why, and for what that trade actually is.
use crate::fasthash::{FxHashMap, FxHashSet};
use std::sync::Arc;

use vitaslop_transpiler::abi::{REG_COUNT, SP};

use crate::capture::Capture;
use crate::vita::lwwork;
use crate::world::{DeterministicWorld, World};
use crate::{SvcOutcome, vita};

/// A GXM opaque handle we hand back to the guest. Kept below the guest image base
/// so it never collides with a real guest address, and nonzero so guest NULL
/// checks pass. The guest treats these as opaque and never dereferences them.
pub const HANDLE_BASE: u32 = 0x4000_0000;

/// The wall-clock instant the guest boots at, in microseconds since the Unix epoch.
///
/// A fixed epoch rather than the host's clock, because every run must be reproducible and
/// a title keying a save or a trophy timestamp off it must see the same value twice. The
/// same number the worlds have always used as their epoch; it lives here now because
/// [`VitaState::guest_wall_us`] is what serves it to the guest.
pub const GUEST_WALL_EPOCH_US: u64 = 1_500_000_000_000_000;

/// Base of the SceUID range module ids come from. Modules are placed once at link
/// time and never unloaded, so a module's load-order index is a stable identity; the
/// base keeps those ids clear of the small ones threads and sync objects mint.
const MODULE_UID_BASE: i32 = 0x0100_0000;

/// The program-header slot of a default uniform buffer binding the TITLE named with
/// `sceGxmSet{Vertex,Fragment}DefaultUniformBuffer`: context state that belongs to no one
/// program, read at the size of whichever program draws. Never a program header address.
/// See [`VitaState::set_default_uniform_buffer`].
pub(crate) const USER_UNIFORM_BUFFER: u32 = u32::MAX;

/// What `sceKernelGetThreadInfo` reports about one thread. A borrowed view rather
/// than a copy: the caller writes it straight into the guest's struct.
pub struct ThreadInfo<'a> {
    pub name: &'a str,
    pub attr: u32,
    pub status: u32,
    pub entry: u32,
    pub stack_base: u32,
    pub stack_size: i32,
    pub init_priority: i32,
    pub current_priority: i32,
    pub cpu_affinity: i32,
    pub exit_status: i32,
}

/// `SceThreadStatus` bits (vitasdk `psp2common/kernel/threadmgr.h`).
const SCE_THREAD_RUNNING: u32 = 1;
const SCE_THREAD_READY: u32 = 2;
const SCE_THREAD_DORMANT: u32 = 16;
const SCE_THREAD_DEAD: u32 = 32;

/// The `SceThreadStatus` of a thread record. A created-but-never-started thread and a
/// finished one are distinguished the same way [`VitaState::delete_thread`] does it:
/// neither has an exit code until it has actually run.
fn thread_status(t: &ThreadRec, current: i32) -> u32 {
    if t.exit_code.is_some() {
        SCE_THREAD_DEAD
    } else if !t.started {
        SCE_THREAD_DORMANT
    } else if t.uid == current {
        SCE_THREAD_RUNNING
    } else {
        SCE_THREAD_READY
    }
}

/// A guest thread the program created: its SceUID, entry address (Thumb bit
/// cleared, so it names the transpiled `f_<addr>` export), its own stack top, and
/// its return value once it has run.
struct ThreadRec {
    uid: i32,
    /// The name the guest passed to `sceKernelCreateThread`, if any. A worker's own
    /// name ("threadFile", "NU::File::RequestManager", "auto_save") is the fastest
    /// way to read a stalled thread dump, so it is kept rather than only logged.
    name: String,
    entry: u32,
    stack_top: u32,
    /// The stack allocation this thread owns: `(base, size)`. Kept so
    /// [`VitaState::delete_thread`] can hand it back for reuse - the guest allocator here
    /// is a bump allocator with no free, so a title that churns threads would otherwise
    /// march through the arena until it collided with something else.
    stack: (u32, u32),
    /// Whether the thread has ever been started. A created-but-never-started thread is
    /// DORMANT and may be deleted; a running one may not, and both have no exit code, so
    /// the exit code alone cannot tell them apart.
    started: bool,
    exit_code: Option<u32>,
    /// SceKernel thread priority (lower number = higher priority). The scheduler
    /// runs the highest-priority runnable thread, matching the real kernel.
    priority: i32,
    /// The priority the thread was CREATED with, before any
    /// `sceKernelChangeThreadPriority`. Reported separately from the current one by
    /// `sceKernelGetThreadInfo`, and the value a title restores when it temporarily
    /// boosts a worker.
    init_priority: i32,
    /// The `attr` and `cpuAffinityMask` from `sceKernelCreateThread`. Neither steers
    /// this scheduler (one core, and the attribute bits select stack/VFP options we
    /// always provide), but both are reported verbatim by `sceKernelGetThreadInfo`,
    /// so they are kept rather than dropped at the seam.
    attr: u32,
    cpu_affinity: i32,
    /// Signals delivered by `sceKernelSendSignal` and not yet consumed by
    /// `sceKernelWaitSignal`. A counter, not a flag: the kernel counts sends, and a
    /// producer that signals twice before the consumer runs must not lose one.
    signals: u32,
}

/// >>> EVERY GUEST STACK THE HOST HAS HANDED OUT, as `[lo, hi)` guest addresses: the main
/// thread's reserve, each thread's stack, each fiber's context, the two host callback stacks.
///
/// Exists for one reader: the transpiler does not mark SP-relative stores, so
/// [`TextureSnapshots::stamp_stacks`] stamps these pages at every epoch instead (see
/// [`GuestMemory::stamp_dirty_pages`]). A LEAF lock of its own, never held while anything else
/// is taken: a stack is registered under the host lock (thread creation), and read under the
/// snapshot cache's lock, which a resolving render worker can hold while it waits for the host
/// lock - one shared lock would order those two both ways.
#[derive(Clone, Default)]
pub(crate) struct GuestStacks(Arc<std::sync::Mutex<Vec<(u32, u32)>>>);

impl GuestStacks {
    /// Record `[lo, hi)`, kept sorted and COALESCED with any range it overlaps or touches.
    ///
    /// Coalesced because the list only grows: a service callback gets a fresh stack from the
    /// bump allocator every time it is spawned and nothing is ever handed back (see the
    /// thread-creation comment on why stacks are leaked), so a title that fires callbacks
    /// steadily would otherwise make every epoch's stamping longer. Back-to-back allocations
    /// are adjacent and merge, so the cost tracks the address span, not the spawn count.
    pub(crate) fn note(&self, lo: u32, hi: u32) {
        if lo >= hi {
            return;
        }
        let mut v = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let (mut lo, mut hi) = (lo, hi);
        v.retain(|&(a, b)| {
            let touches = a <= hi && lo <= b;
            if touches {
                lo = lo.min(a);
                hi = hi.max(b);
            }
            !touches
        });
        let at = v.partition_point(|&(a, _)| a < lo);
        v.insert(at, (lo, hi));
    }

    /// A copy, so the caller holds no lock while it stamps.
    fn snapshot(&self) -> Vec<(u32, u32)> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// `SCE_KERNEL_DEFAULT_PRIORITY` - the *sentinel* a title passes to
/// `sceKernelCreateThread` to mean "the default priority", and the base for
/// relative priorities (`base + delta`). It is NOT an actual runnable priority;
/// [`resolve_priority`] maps it (and the relative range around it) to a concrete
/// value before it ever reaches the scheduler.
pub const SCE_KERNEL_DEFAULT_PRIORITY: i32 = 0x1000_0100;

/// The concrete default user-thread priority (`SCE_KERNEL_DEFAULT_PRIORITY_USER`,
/// 0xA0). Absolute user priorities run 0x40 (highest) .. 0xBF (lowest); 0xA0 is the
/// middle default the initial (main) thread and a defaulted worker resolve to.
pub const DEFAULT_THREAD_PRIORITY: i32 = 0xA0;

/// The SceUID of the initial ("main") thread. It is the one thread with no
/// [`ThreadRec`]: `create_thread` records what the GUEST creates, and the loader
/// starts this one before any guest code runs. Anything that has to name it uses
/// this rather than a bare 0.
///
/// >>> IT MUST NOT BE ZERO, AND THAT IS NOT A STYLE POINT. The kernel hands the guest
/// this id through `sceKernelGetThreadId`, and NO SceUID is ever zero on hardware:
/// titles use 0 as "no thread owns this" in their own lock words, and the threadmgr API
/// itself uses a zero `thid` argument to mean "the calling thread". A main thread whose
/// id is 0 makes those two readings collide with a real thread. MEASURED on one title: a
/// hand-rolled recursive lock stores the owning thread id in a word and treats 0 as free,
/// so with `MAIN_THID = 0` every one of the main thread's acquisitions was a no-op that
/// believed it already held the lock, the recursion depth ran away, and a display worker
/// later took the same lightweight mutex for real and never gave it back - a deadlock
/// 1,000 frames after the corruption, with nothing at the stall naming this line.
/// 0x40 is below the `next_uid` floor (0x100), so it can never collide with a created
/// object either.
pub const MAIN_THID: i32 = 0x40;

/// `SCE_KERNEL_ERROR_WAIT_TIMEOUT` - the value a *timed* blocking wait
/// (`sceKernelWaitSema`/`WaitCond`/`WaitLwCond`/`WaitEventFlag` with a non-null
/// timeout) returns when its deadline passes before the wait is satisfied. Delivered
/// to the woken thread's `r0` through the resume-code channel (see
/// [`VitaState::take_resume_code`]); a wait satisfied by a signal returns 0 instead.
pub const SCE_KERNEL_ERROR_WAIT_TIMEOUT: u32 = 0x8002_8005;
/// `SCE_KERNEL_ERROR_WAIT_DELETE` (vitasdk `psp2/kernel/error.h`): the object a parked thread
/// waited on was deleted under it.
pub const SCE_KERNEL_ERROR_WAIT_DELETE: u32 = 0x8002_8006;

/// `SCE_KERNEL_ERROR_NOT_DORMANT` - a thread operation that requires the thread to be
/// stopped was asked of a running one.
///
/// From vitasdk `psp2/kernel/error.h`, NOT the psdevwiki error-code table: that table
/// disagrees with the header on this whole block (it lists NOT_DORMANT at 0x80028002 and
/// WAIT_TIMEOUT at 0x80028008, where the header has 0x80028028 and 0x80028005). The header
/// is the one that matches the value this engine already returns for a timed-out wait, so
/// the table's rows appear to be shifted. Prefer the header for SceKernel error numbers.
pub const SCE_KERNEL_ERROR_NOT_DORMANT: u32 = 0x8002_8028;
/// `SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID` - no thread has that SceUID (same header).
pub const SCE_KERNEL_ERROR_UNKNOWN_THREAD_ID: u32 = 0x8002_8021;
/// Bytes of SceAppUtil safe memory each title has.
pub const SAFE_MEMORY_SIZE: usize = 64 * 1024;
/// `SCE_KERNEL_ERROR_ILLEGAL_PRIORITY` (psp2/kernel/error.h).
pub const SCE_KERNEL_ERROR_ILLEGAL_PRIORITY: u32 = 0x8002_8023;
/// `sceKernelSendSignal` to a thread whose last signal is still unconsumed (a latch of one).
pub const SCE_KERNEL_ERROR_ALREADY_SENT: u32 = 0x8002_8121;

/// Whether a fiber's backing thread runs at its runner's priority (the hardware: a fiber IS its
/// runner) rather than [`DEFAULT_THREAD_PRIORITY`]. OFF by default; `VITASLOP_FIBER_RUNNER_PRIORITY=1`
/// turns it on. MEASURED (phone job 532, with the SMP fiber-near arm `smp::fiber_near_on`
/// also on): a fighting title's intro movie stalled again - main's fibers at main's 0x56 on main's worker outrank the
/// movie player's threads (0x8e-0x96) that inherit the same core, and on a phone that runs guest
/// code several times slower than a Vita they leave the demuxer too little time before the
/// player's first buffering check. Faithful on paper, wrong on this CPU until the A/B says how.
fn fiber_runner_priority() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        crate::knobs::var("VITASLOP_FIBER_RUNNER_PRIORITY")
            .as_deref()
            .map(str::trim)
            == Ok("1")
    })
}

/// Resolve a `sceKernelCreateThread` priority argument to a concrete scheduler
/// priority. Absolute user priorities (small numbers, ~0x40..0xBF) pass through;
/// the relative range around [`SCE_KERNEL_DEFAULT_PRIORITY`] (e.g. the sentinel
/// itself, or `default - 1`) is rebased onto [`DEFAULT_THREAD_PRIORITY`]. Without
/// this the sentinel (a huge number) would read as the lowest possible priority,
/// so any concrete-priority worker would outrank - and starve - the main thread.
pub fn resolve_priority(prio: i32) -> i32 {
    if prio >= 0x1000_0000 {
        DEFAULT_THREAD_PRIORITY + (prio - SCE_KERNEL_DEFAULT_PRIORITY)
    } else {
        prio
    }
}

/// A request to synchronously run guest code (a thread entry) that a host call
/// raised. The engine-agnostic runtime cannot itself re-enter the wasm engine, so
/// it records this intent and the engine host ([`vitaslop_native::Vm`]) executes
/// it - seeding r0/r1 and sp, calling the guest function, and reporting the
/// result back through [`ImportDispatch::set_thread_exit`]. The register file is
/// saved and restored around the call so the re-entry is transparent to the
/// interrupted thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reentry {
    /// Guest function to run (Thumb bit cleared; names the `f_<addr>` export).
    pub entry: u32,
    /// r0 for the entry: the argument-block length (`SceSize args`).
    pub arg_len: u32,
    /// r1 for the entry: the argument-block pointer (`void *argp`).
    pub arg_ptr: u32,
    /// r2 for the entry. Zero for a `sceKernelStartThread` entry (whose ABI is only
    /// `(SceSize args, void *argp)`); nonzero only for a host-delivered callback
    /// whose ABI passes a third register (e.g. an NP service-state callback thunk
    /// that takes its `this`/userdata in r2).
    pub r2: u32,
    /// r3 for the entry - a kernel callback's fourth argument (`void *common`). Zero for
    /// everything else.
    pub r3: u32,
    /// sp for the entry: the top of the thread's own stack.
    pub stack_top: u32,
    /// The thread whose exit code the result becomes.
    pub thid: i32,
    /// The new thread's SceKernel priority (lower = higher), so the scheduler can
    /// place it correctly on the run queue.
    pub priority: i32,
}

/// One `sceGxmCreateDeferredContext` context. Its sticky state is in its own block in guest
/// memory; this is what the host keeps beside it.
#[derive(Default)]
pub struct DeferredContext {
    /// The draws of the command list being recorded, between `sceGxmBeginCommandList` and
    /// `sceGxmEndCommandList`; `None` while not recording.
    pub recording: Option<Vec<RecordedDraw>>,
    /// The VDM, vertex and fragment buffers the title last gave (`params` at create, then
    /// `sceGxmSetDeferredContext*Buffer`), answered by the matching getters.
    pub buffers: [(u32, u32); 3],
}

/// The draw call a command list replays.
#[derive(Clone, Copy, Debug)]
pub enum DeferredDraw {
    /// `sceGxmDraw` (`index_wrap` 0) or `sceGxmDrawInstanced`.
    Indexed {
        primitive: u32,
        index_format: u32,
        index_data: u32,
        index_count: u32,
        index_wrap: u32,
    },
    /// `sceGxmDrawPrecomputed`.
    Precomputed { precomputed: u32 },
}

/// One draw of a command list: the call, the deferred context it was made on, and a copy of
/// that context's block as the draw saw it.
#[derive(Clone)]
pub struct RecordedDraw {
    pub context: u32,
    pub draw: DeferredDraw,
    pub block: Vec<u8>,
    pdraw: Option<(u32, [u32; pdraw::STREAM_SLOTS])>,
}

/// A guest function a host call runs on its CALLER's thread before it returns - see
/// [`SvcOutcome::CallGuest`] and [`VitaState::call_guest`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestCall {
    /// The function, as the title handed it over (the Thumb bit is the engine's to clear).
    pub entry: u32,
    /// Its arguments in AAPCS order: the first four in r0..r3, the rest on the stack. A
    /// 64-bit argument is passed as its two halves, already placed at the slot AAPCS gives
    /// it (an even register pair, or an 8-byte-aligned stack slot).
    pub args: Vec<u32>,
}

/// One `sceKernelRegisterThreadEventHandler` registration.
#[derive(Clone, Debug)]
pub struct ThreadEventHandler {
    pub uid: i32,
    /// The thread it watches, or [`THREAD_EVENT_ALL_IN_PROCESS`].
    pub target: i32,
    /// `SceKernelThreadEventType` bits it asked for.
    pub mask: u32,
    pub handler: u32,
    pub common: u32,
}

/// `SceKernelThreadEventType`: a thread starting / ending (psp2common/kernel/threadmgr.h).
pub const THREAD_EVENT_START: u32 = 0x4;
pub const THREAD_EVENT_EXIT: u32 = 0x8;
/// The target id meaning "every thread in this process" (`SCE_KERNEL_THREAD_ID_PROCESS_ALL_ID`).
pub const THREAD_EVENT_ALL_IN_PROCESS: i32 = 0x10027;

/// The highest key `sceKernelGetTLSAddr` has a slot for: keys `0..=0x100` index one
/// per-thread array of pointer slots.
pub const TLS_KEY_MAX: u32 = 0x100;

/// Bytes kept below the caller's stack pointer before a guest call's own frame starts.
///
/// The call is made from inside a library function the title called, and a real library
/// has a frame of its own at that point; nothing reads this gap, it only keeps the callee's
/// frame clear of anything the engine's call sequence might have left just under `sp`.
const GUEST_CALL_GAP: u32 = 64;

impl GuestCall {
    /// Seed `regs` - the caller's register file, as the host call saw it - for this call:
    /// r0..r3 from the arguments, the rest written to a fresh stack frame below the caller's
    /// `sp` (8-byte aligned at the call, as AAPCS requires), `sp` pointed at it. Every other
    /// register keeps the caller's value, which is what a call from inside a library
    /// function leaves in them.
    pub fn seed(&self, regs: &mut [u32; REG_COUNT], mem: &mut dyn GuestMemory, base: u32) {
        for i in 0..4 {
            regs[i] = self.args.get(i).copied().unwrap_or(0);
        }
        let stacked = self.args.len().saturating_sub(4) as u32;
        let sp = (regs[SP].wrapping_sub(GUEST_CALL_GAP + stacked * 4)) & !7;
        for (i, &w) in self.args.iter().skip(4).enumerate() {
            let off = sp.wrapping_add(i as u32 * 4).wrapping_sub(base) as usize;
            mem.write(off, &w.to_le_bytes());
        }
        regs[SP] = sp;
    }
}

/// What a finished [`GuestCall`] left behind: its r0/r1 (r1 is the high half of a 64-bit
/// return), and the bytes of the region the caller asked to have captured the moment the
/// call returned - see [`VitaState::call_guest`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuestCallResult {
    pub r0: u32,
    pub r1: u32,
    pub bytes: Vec<u8>,
}


/// Lock [`VitaState::texture_snapshots`] through a FIELD path, so the guard borrows that one
/// field and the rest of `self` stays free. The cache has its own lock because the flip's
/// geometry read ([`compute_resolve_job`]) runs off the host lock on the presenting worker.
/// Not re-entrant: a caller holding the guard must not reach another `snaps!`.
macro_rules! snaps {
    ($s:expr) => {{
        // Timed only when it has to WAIT (`Phase::SnapsLockWait`): the resolver worker holds this
        // lock for its whole flip-time read, and a wait here is billed to whichever phase the
        // caller is in unless it is named.
        let mut g = match $s.texture_snapshots.try_lock() {
            Ok(g) => g,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                let _w = crate::perf::scope(crate::perf::Phase::SnapsLockWait);
                // Billed to the line HOLDING it - see `perf::snaps_waited`.
                let (holder, t0) = if crate::perf::enabled() {
                    (crate::perf::snaps_holder(), crate::perf::now_ms())
                } else {
                    (0, None)
                };
                let g = $s
                    .texture_snapshots
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                if let (Some(t0), Some(t1)) = (t0, crate::perf::now_ms()) {
                    crate::perf::snaps_waited(holder, ((t1 - t0).max(0.0) * 1.0e6) as u64);
                }
                g
            }
        };
        crate::perf::snaps_taken(crate::site!());
        // Frame boundaries that arrived while another holder had the cache (see
        // `snaps_frames_pending`), applied before anyone reads it.
        for _ in 0..$s
            .snaps_frames_pending
            .swap(0, std::sync::atomic::Ordering::AcqRel)
        {
            g.begin_frame();
        }
        g
    }};
}

/// Byte layout of the guest `SceGxmPrecomputedDraw` work area. Eleven 32-bit words is
/// what `sceGxmGetPrecomputedDrawSize` reports and what the SDK struct declares
/// (`uint32_t reserved[11]`), and its contents are opaque to the guest, so the fields
/// below are ours to define. Word 0 is a tag: a block that does not carry it was never
/// initialised through our `Init` (or was copied from uninitialised memory), which is a
/// hard error rather than a silently skipped draw.
mod pdraw {
    /// "PDRW" - the initialised-block tag in word 0.
    pub const MAGIC: u32 = 0x5744_5250;
    pub const OFF_MAGIC: u32 = 0;
    pub const OFF_VERTEX_PROGRAM: u32 = 4;
    /// Stream `i`'s buffer pointer, one word each: words 2, 7, 8, 9. Stream 0 keeps word 2
    /// (where it has always been) and the rest use the words that were spare.
    ///
    /// FOUR, not [`super::MAX_VERTEX_STREAMS`]: `SCE_GXM_PRECOMPUTED_DRAW_WORD_COUNT` is 11
    /// and the other seven words are spoken for, so this encoding can carry four stream
    /// pointers and no more. That is a limit of OUR packing (real GXM packs the block
    /// differently), which is why a draw that needs a fifth stream is reported by
    /// `precomputed_draw_set_stream` rather than silently dropping the buffer.
    pub const OFF_STREAM: [u32; STREAM_SLOTS] = [8, 28, 32, 36];
    /// How many stream pointers [`OFF_STREAM`] has room for.
    pub const STREAM_SLOTS: usize = 4;
    pub const OFF_PRIMITIVE: u32 = 12;
    pub const OFF_INDEX_FORMAT: u32 = 16;
    pub const OFF_INDEX_ADDR: u32 = 20;
    pub const OFF_INDEX_COUNT: u32 = 24;
    /// Word 10: the instancing wrap (0 = not instanced). It was the one spare word.
    pub const OFF_INDEX_WRAP: u32 = 40;
    pub const WORDS: u32 = 11;
}

/// `SCE_GXM_MAX_VERTEX_STREAMS`. A vertex program may source its attributes from up to
/// this many separate guest buffers.
///
/// One definition, in [`crate::vita::gxmctx`], because the context block reserves a slot per
/// stream and a count that disagreed with the block's layout would read a neighbour's word.
/// It used to be 4 here - the number of stream pointers a `SceGxmPrecomputedDraw` can carry
/// in our packing, which is a different quantity entirely and is now `pdraw::STREAM_SLOTS`.
pub(crate) use crate::vita::gxmctx::MAX_VERTEX_STREAMS;

/// One reflected uniform parameter: where its components sit in the program's default
/// uniform buffer and how they are packed. `res` is a 4-byte-register offset; each of
/// the `comp` components is an F16 (2 bytes) when `f16`, else an F32.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ParamRef {
    res: u32,
    comp: u8,
    f16: bool,
}

/// Everything the capture needs to know about ONE `SceGxmProgram`, reflected out of
/// its parameter table.
///
/// # Why this is cached
/// A program's parameter table is immutable for the life of the registration: it is
/// part of the compiled shader blob the guest hands to the shader patcher. Every field
/// here is therefore a constant of the program, while the VALUES at these offsets
/// change every draw.
///
/// Reflecting per draw meant five independent walks of the table (world/projection,
/// exposure, shader-expansion, albedo sampler, material) each allocating a `String`
/// per parameter to lowercase and substring-match its name. At a couple of hundred
/// draws a frame that was tens of thousands of allocations per frame and the single
/// most expensive thing a `sceGxmDraw` did. One walk, once per program, replaces it.
///
/// Every field is `Copy`, so a draw takes the whole reflection out of the cache
/// without touching the allocator.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ProgramReflection {
    /// Float offset of the model->world 4x4 (`vsModelToWorldMatrix`).
    world_off: Option<u32>,
    /// Float offset of the world->projection 4x4.
    proj_off: Option<u32>,
    /// Float offset of a SINGLE combined model->clip 4x4 (`worldViewProj`), when the
    /// shader keeps one instead of a model/projection pair. Takes precedence over
    /// [`Self::world_off`]/[`Self::proj_off`] because it is already the whole transform.
    mvp_off: Option<u32>,
    /// Float offset of the scene exposure scalar (`vsCoarseExposureReg`).
    exposure_off: Option<u32>,
    /// Does this vertex program SYNTHESIZE its primitive (point sprite / billboard)?
    shader_expanded: bool,
    /// Texture unit of the sampler that reads as the albedo / base-colour map.
    albedo_unit: Option<u32>,
    /// The base-colour tint multiplying the sampled albedo.
    tint: Option<ParamRef>,
    /// Direction and colour of the single directional light the material model uses.
    light_dir: Option<ParamRef>,
    light_col: Option<ParamRef>,
    /// Size of the program's default uniform buffer, from the table's extent in 32-bit
    /// REGISTERS (each parameter's components packed at its own type's width, not one
    /// register per component). See [`VitaState::reflected_uniform_size_bytes`].
    uniform_size_bytes: u32,
    /// The program header's OWN default-uniform-buffer size field, clamped and scaled -
    /// what [`default_uniform_buffer_bytes`] reads, memoised here.
    ///
    /// It is a word of the container, which is immutable while the program is registered
    /// (the same fact [`ProgramMemos`] and this whole table rest on), and the staleness check
    /// in `record_draw` read it FRESH on every draw of every frame. MEASURED in the browser:
    /// two single-word guest reads per draw, ~2,000 a presented frame, which is where the
    /// draw path's whole remaining word-read count came from. A word read is a bounds check
    /// and a virtual call into a typed array there [[vitaslop-browser-scalar-reads-are-a-typed-array]].
    default_uniform_bytes: u32,
    /// Texture units the program's samplers occupy (one past the highest sampler
    /// resource index). This is the length of the texture array the whole-array
    /// setter `sceGxmPrecomputed*StateSetAllTextures` is given.
    texture_unit_count: u32,
    /// >>> WHICH texture units this program's samplers DECLARE, one bit per unit.
    ///
    /// GXM's context block carries 16 fragment texture slots and a title leaves whatever it
    /// last bound in the ones the current program does not read. Those slots are state no
    /// draw of this program can sample - a program reads the samplers it declares and there
    /// is no other path to a texture unit - so decoding them is work for bytes that cannot
    /// reach a pixel.
    ///
    /// MEASURED on a golf title's gameplay frame: **7,366 texture resolutions across 672
    /// draws** - eleven per draw - against fragment programs that declare a handful. That is
    /// `host::decode_texture` at 10.5% of all busy time in a V8 profile of the worker, and it
    /// is also why the whole-set memo almost never hits: the key folds every bound slot, so a
    /// stale binding in a slot nobody samples makes two otherwise identical draws two
    /// different sets.
    ///
    /// Zero means "declares no samplers", which is a real answer (a program that samples
    /// nothing) - and also what an unreadable parameter table produces, which is why the
    /// narrowing is applied only when the mask is non-zero.
    sampler_units: u32,
}

/// Which stage's uniform-buffer table a lookup reads. The two are separate arrays in the
/// context block because GXM numbers them independently.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum UniformStage {
    Vertex,
    Fragment,
}

impl UniformStage {
    fn name(self) -> &'static str {
        match self {
            UniformStage::Vertex => "vertex",
            UniformStage::Fragment => "fragment",
        }
    }
}

/// Where one program's uniform BUFFERS live in the SA register file - see
/// [`VitaState::sa_uniform_image`].
#[derive(Default)]
struct SaUniformLayout {
    /// First SA register of the DEFAULT uniform buffer, and its extent, from the program's
    /// own container table (0 and the header's declared size when it declares no container).
    default_base: u32,
    default_regs: u32,
    /// Every NON-default uniform buffer the driver copies into the register file.
    buffers: Vec<vitaslop_gxp_shader::container::SaUniformBuffer>,
}

impl SaUniformLayout {
    /// The shared empty layout, so the `header == 0` path allocates nothing.
    fn none() -> std::sync::Arc<Self> {
        use std::sync::{Arc, OnceLock};
        static NONE: OnceLock<Arc<SaUniformLayout>> = OnceLock::new();
        NONE.get_or_init(|| Arc::new(SaUniformLayout::default()))
            .clone()
    }
}

/// Whether the `sceKernelDelayThread(0)` yield elision is on - see
/// [`VitaState::yield_would_repick_this_thread`]. DEFAULT ON, so `VITASLOP_YIELD_ELIDE=0` is
/// the arm back to always suspending, which is what an A/B needs from ONE build.
/// `(elided, taken)` yields - see [`VitaState::yield_would_repick_this_thread`]. Statics rather
/// than state fields so the end-of-run line can be printed from outside a locked host.
static YIELDS_ELIDED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static YIELDS_TAKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws that reached a stage whose default uniform buffer was reserved for a DIFFERENT
/// program, and so rendered with NO uniform bank at all - counted whether or not the
/// deduplicating warning chose to print that one. See [`VitaState::stale_uniforms`].
///
/// A total rather than a rate because the budget on it is ZERO: there is no count of draws
/// rendering without the constants they declare that is acceptable. It is a static for the same
/// reason the yield counters are - the end-of-run line is printed from outside a locked host.
static STALE_UNIFORM_DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many draws rendered with no uniform bank because the bound one belonged to another
/// program. See [`STALE_UNIFORM_DRAWS`].
pub fn stale_uniform_draws() -> u64 {
    STALE_UNIFORM_DRAWS.load(std::sync::atomic::Ordering::Relaxed)
}

/// `(yields elided because nothing else could run, yields taken)`.
pub fn yield_report() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (YIELDS_ELIDED.load(Relaxed), YIELDS_TAKEN.load(Relaxed))
}

/// Print what the yield elision did, unconditionally, at the end of a headless run.
///
/// Both halves, because a guard that never fires and a build where it is switched off look
/// identical from every other number in the report.
pub fn report_yield_elision() {
    let (elided, taken) = yield_report();
    if elided == 0 && taken == 0 {
        return;
    }
    println!(
        "yield elision: {elided} `delay(0)` yields returned to the caller (nothing else was          runnable), {taken} suspended. Each elided one is a full fiber suspend AND resume not          taken, which on the browser is the most expensive thing a scheduler round does.          `VITASLOP_YIELD_ELIDE=0` suspends on all of them."
    );
}

fn yield_elide_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            vitaslop_platform::knobs::var("VITASLOP_YIELD_ELIDE")
                .as_deref()
                .map(str::trim),
            Ok("0") | Ok("false") | Ok("no") | Ok("off")
        )
    })
}

/// Say - once per program - that a draw's program declares an SA-resident uniform buffer the
/// guest has not bound, so those registers read ZERO.
///
/// This is the half of the story the renderer's `report_unfed_uniforms` cannot tell: that one
/// names a program whose buffer nothing FEEDS, and is silent once feeding exists. This names
/// the case where the feeding exists and the guest simply has not bound the buffer at this
/// draw - which renders with zeros just the same, and needs the opposite investigation (a call
/// order, not a missing mechanism).
fn report_unbound_sa_uniform_buffer(header: u32, stage: UniformStage, index: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    // Per draw per unbound buffer: a key this thread has seen returns before the global lock.
    thread_local! {
        static DONE: std::cell::RefCell<crate::fasthash::FxHashSet<(u32, u32)>> = std::cell::RefCell::new(Default::default());
    }
    if !DONE.with(|d| d.borrow_mut().insert((header, index))) {
        return;
    }
    static SEEN: Mutex<Option<HashSet<(u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert((header, index)) {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        program = format_args!("{header:#x}"),
        stage = stage.name(),
        index,
        "the program reads uniform buffer {index} out of the SA register file, but the guest has NOT bound it at this draw - those registers read ZERO"
    );
}

/// Say - once per program - that the header's declared default-uniform size exceeds the
/// extent its own container table gives that buffer, so the tail is not carried.
///
/// Both numbers are read from the blob and they disagree on one retail title's vertex
/// programs (30 declared against a 16-register container, 28 against 14). The container's is
/// the one the layout corroborates - it is contiguous with the DATA container that holds the
/// program's literals, and taking the header's would run the buffer over them - so that is
/// what is used, and the difference is stated here rather than left to be inferred from a
/// wrong colour.
fn report_default_uniform_container_short(header: u32, container_regs: u32, short: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g.get_or_insert_with(HashSet::new).insert(header) {
        return;
    }
    tracing::warn!(
        target: "vitaslop::gxm",
        program = format_args!("{header:#x}"),
        container_regs,
        short,
        "the program's DEFAULT uniform buffer is declared {} registers in the header but its container table gives it only {container_regs} - the container's extent is used (it is contiguous with the DATA container holding this program's literals, so the larger reading would run over them) and the last {short} register(s) of the declared parameter block are NOT carried",
        container_regs + short,
    );
}

/// All host state for one run: the guest allocator, handle tables, the capture
/// stream, the world (determinism seam), and the in-progress scene state.
/// Which path is advancing the virtual game clock, for [`VitaState::clock_sources`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum ClockSource {
    /// A scheduler quantum of guest execution ([`VitaState::charge_cpu_quantum`]).
    Quantum,
    /// The per-flip top-up ([`VitaState::advance_time_frame`]), which is OPT-IN:
    /// `VITASLOP_FRAME_TOPUP=1` turns it on and every other value, including absent, leaves it
    /// OFF. See `frame_topup` - and note the comment on `advance_time_frame` used to say "=0
    /// switches it off", which reads as though the default were ON. It is not, and that wording
    /// cost this project a measurement: a run made to test the topup-on arm was the topup-OFF
    /// arm and produced numbers identical to the run it was meant to be compared against.
    /// [[vitaslop-knob-is-the-gate-not-the-level]]
    FrameTopup,
    /// The scheduler's idle path jumping to the earliest pending deadline - and the
    /// default, so an advance from an untagged path is attributed rather than lost.
    Idle,
    /// The wall-clock floor ([`VitaState::wall_floor_tick`]): the guest ran slower than real
    /// time and the clock was pulled up to it.
    Wall,
}

/// The kind of timed wait that OWNED the deadline an idle jump landed on, plus the thread
/// waiting on it. See [`VitaState::idle_attribution`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IdleOwner {
    pub kind: IdleKind,
    pub thid: i32,
}

/// Which timed-wait family an [`IdleOwner`] belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IdleKind {
    /// `sceKernelDelayThread` and the `sceAudioOutOutput` grain park - a pure sleep.
    Sleep,
    /// A timed `sceKernelWaitLwCond`.
    LwCond,
    /// A timed `sceKernelWaitCond`.
    Cond,
    /// A timed `sceKernelWaitSema`.
    Sema,
    /// A timed `sceKernelWaitEventFlag`.
    EventFlag,
    /// A timed `sceKernelLockLwMutex`.
    LwMutex,
    /// A timed `sceKernelReceiveMsgPipe`.
    MsgPipe,
    /// A timed `sceKernelLockReadRWLock` / `sceKernelLockWriteRWLock`.
    RwLock,
}

impl IdleKind {
    pub fn name(self) -> &'static str {
        match self {
            IdleKind::Sleep => "sleep/delay",
            IdleKind::LwCond => "lwcond",
            IdleKind::Cond => "cond",
            IdleKind::Sema => "sema",
            IdleKind::EventFlag => "evf",
            IdleKind::LwMutex => "lwmutex",
            IdleKind::MsgPipe => "msgpipe",
            IdleKind::RwLock => "rwlock",
        }
    }
}

pub struct VitaState {
    /// >>> A SMALL RENDER TARGET IS COMPLETED AT ITS OWN `sceGxmEndScene`, not at frame end.
    ///
    /// Every scene completes synchronously at `sceGxmEndScene` as far as the guest can tell
    /// (notifications signal there, `sceGxmFinish` has nothing to wait for), but its PIXELS
    /// only reached guest memory at the end of the display frame - so a title that ends a
    /// probe scene and reads a texel of it on the CPU in the same frame read what its
    /// allocator left there. MEASURED on a baseball title: its ambient light is read ONCE, at
    /// load, right after the probe's scene ends; the read returned `0xBAADCAFE` decoded
    /// through the title's log curve (97, 24, 11), it cached that for the run, and every
    /// later write-back landed on a value nobody re-read. The frame stayed washed out with
    /// the write-back in place and the probe painted correctly.
    ///
    /// The frontend installs this: given the scene just ended, render it NOW and return the
    /// small targets' pixels (`(guest address, width, height, RGBA8)`), which are written back
    /// here before `sceGxmEndScene` returns. Only scenes whose colour target fits the
    /// write-back cap take this path; the frame render then skips them
    /// (`Scene::completed_early`). `None` on a frontend without a renderer.
    pub complete_scene_now:
        Option<Box<dyn FnMut(&[crate::capture::Scene]) -> Vec<(u32, u32, u32, Vec<u8>)> + Send>>,
    /// The ASYNC form of the same completion, for a frontend whose readback cannot block (the
    /// browser: a GPU map resolves on a later event-loop turn). When set, a small target's
    /// `sceGxmEndScene` records the batch in `pending_early` and BLOCKS the calling thread;
    /// the frontend's run loop renders the batch, awaits the pixels, writes them back, marks
    /// the scenes completed and wakes the thread (`wake_thread`). `sceGxmFinish` semantics
    /// by another route: the guest cannot run past the end of the scene until its pixels are
    /// in memory.
    pub complete_scene_async: bool,
    /// Every address the guest has named as a display buffer (`sceDisplaySetFrameBuf`) or as
    /// a colour surface's data (`sceGxmColorSurfaceInit`): the set a display-queue flip
    /// request's callback data is matched against - see `gxm::display_queue_add_entry`.
    pub flip_candidates: std::collections::HashSet<u32>,
    /// `sceAppMgrLoadExec`'s target, when the title asked to REPLACE this process with another
    /// executable of its own app. The run halts; the host that owns the process boots that
    /// executable in its place (see `ingest::pipeline::Game::with_main_exec`).
    pub exec_request: Option<String>,
    /// The INITIAL thread's pending `sceKernelSendSignal` latch - it has no create record to
    /// carry one (see [`Self::send_signal`]).
    main_signals: u32,
    /// `(thread id, first scene index, one past the last)` of the batch a blocked
    /// `sceGxmEndScene` is waiting on - see `complete_scene_async`.
    pub pending_early: Option<(i32, usize, usize)>,
    pub base: u32,
    pub mem_bytes: u32,
    /// The guest heap and the memory-block registry on it - see [`GuestArena`], which owns
    /// them.
    arena: GuestArena,
    next_handle: u32,
    next_uid: i32,
    /// The modelled storage device's clock, its accrued debt and the threads parked paying
    /// it - see [`StorageClock`], which owns them.
    storage: StorageClock,
    /// Game-clock time already charged by quantum boundaries since the last display
    /// flip, so the per-flip advance can top the clock up to exactly one frame rather
    /// than double-count. The role [`StorageClock`]'s own per-flip charge plays for the
    /// storage clock. See [`VitaState::charge_cpu_quantum`].
    cpu_charged_since_flip: u64,
    /// Sub-microsecond remainders of the two per-fuel charges, `(game clock, storage
    /// clock)`, carried from one charge to the next.
    ///
    /// # Without these the clock STOPS on a title that yields in small bursts
    /// A charge is `per_quantum * fuel / QUANTUM_FUEL`, and `QUANTUM_FUEL` is five
    /// million. A thread that suspends after a few hundred fuel units - a spin whose loop
    /// body makes one cheap host call - therefore rounds to **zero microseconds**, every
    /// time, forever. That is not a rounding error, it is a stopped clock: measured on a
    /// retail title in the browser, one thread was resumed 198,968 times at frame 2 while
    /// the game clock stood still at 3.44 s, because the storage transfer it was spinning on
    /// could only complete when time passed and no time ever did.
    ///
    /// Carrying the remainder makes the charge exact in the long run - a hundred yields of
    /// a tenth of a microsecond now cost ten microseconds instead of nothing - which is
    /// what lets such a spin terminate. It is the same argument the CPU charge itself
    /// rests on: work the guest really did has to cost time, or something waiting on time
    /// waits for ever.
    charge_rem: (u64, u64),
    /// Scheduler quanta observed, for the calibration diagnostic that sets `QUANTUM_CPU_US`
    /// (a quantum is worth one frame divided by the quanta a rendering frame actually takes -
    /// see [`VitaState::charge_cpu_quantum`]); the flips are [`FlipPacer`]'s count.
    quantum_count: u64,
    /// The scanout's side of a flip - latch queue, vsync floor, queue depth, flip count - see
    /// [`FlipPacer`] and [`VitaState::pace_flip`].
    flip_pacer: FlipPacer,
    /// What the guest told the display controller with `sceDisplaySetFrameBuf` - see
    /// [`DisplayController`].
    display: DisplayController,
    /// Every program registered with the shader patcher and every vertex and fragment program
    /// created from one, with each patcher's user data - see [`ShaderPatcherRecords`].
    shader_patcher: ShaderPatcherRecords,
    /// Everything memoised per registered program - reflection, container bytes, epilogue
    /// blend, memory windows, SA uniform layout - see [`ProgramMemos`], which owns them and
    /// clears them together.
    program_memos: ProgramMemos,
    /// The shader PAIRS the renderer can prepare ahead of their first draw, and the program
    /// lists they come from - see [`PrecompileQueue`], which owns them.
    precompile: PrecompileQueue,
    /// The colour surfaces the guest initialised, and their gamma, clip and dither state -
    /// see [`ColorSurfaces`], which owns them.
    color_surfaces: ColorSurfaces,
    /// Guest address of the displayQueue callback from sceGxmInitialize, and its
    /// data size. Recorded for faithfulness; the present address is captured from
    /// the display queue directly so the callback need not run yet.
    pub display_queue_cb: u32,
    pub display_queue_cb_data_size: u32,
    // In-progress scene (BeginScene..EndScene).
    scene: Option<crate::capture::Scene>,
    /// The guest `SceGxmContext *` the sticky draw state lives in - the `hostMem` the guest
    /// handed `sceGxmCreateContext`, as on hardware. See [`crate::vita::gxmctx`].
    ///
    /// This is the only GXM context state left on the host, and it is an ADDRESS, not a
    /// value: the bound vertex/fragment program, the vertex streams and the whole
    /// [`crate::capture::RenderState`] are read back out of guest memory on demand
    /// ([`Self::bound_vertex_program`] and friends). That is what lets their setters be
    /// inlined into guest code instead of crossing the host boundary once per call.
    ///
    /// Zero until a context exists. A draw before then is reported, not guessed at.
    gxm_context: u32,
    /// `sceGxmCreateDeferredContext` contexts, by address (see [`Self::is_deferred_context`]).
    pub(crate) deferred_contexts: FxHashMap<u32, DeferredContext>,
    /// Ended command lists, by the id their `SceGxmCommandList` carries.
    pub(crate) command_lists: Vec<Vec<RecordedDraw>>,
    /// Whether a draw with no context has already been reported, so the report is once.
    reported_no_gxm_context: bool,
    /// Whether a draw whose fragment program DECLARES samplers but whose binding list came
    /// out empty has already been reported. See [`Self::report_draw_without_bound_textures`].
    reported_draw_without_textures: bool,
    /// >>> WHO HAS WRITTEN THE CONTEXT'S TEXTURE SLOTS, counted for the failure report above.
    ///
    /// Only ever PRINTED by [`Self::report_draw_without_bound_textures`], which fires once per
    /// run on a draw that reaches the renderer with no texture. The counters exist because
    /// that report can otherwise only say the slots are zero, and "nothing ever bound one" and
    /// "something bound one and a later bind CLEARED it" are opposite bugs that leave the same
    /// zeros [[vitaslop-poison-separates-a-guest-zero-from-an-unwritten-one]].
    ///
    /// They count the HOST-ROUTED writers only - an inlined `sceGxmSetFragmentTexture` writes
    /// the slot from emitted code and never reaches this struct
    /// [[vitaslop-inlined-host-calls-escape-both-watchpoints]] - so read them under
    /// `VITASLOP_NO_INLINE_TEXTURE=1` (or `VITASLOP_NO_INLINE_IMPORTS=1`), where every bind is
    /// host-routed and the counts are complete. The report says which.
    texture_writes: TextureSlotWrites,
    /// Whether the draw being recorded arrived through `sceGxmDrawPrecomputed`. Set only
    /// around that call; read only by the empty-bindings report.
    last_draw_was_precomputed: bool,
    /// The `SceGxmPrecomputedDraw` block of that draw, for the per-draw dump - which names the
    /// block so its setters can be traced back (`VITASLOP_HOSTCALL_WATCH`).
    last_precomputed_block: u32,
    /// The vertex program and stream pointers a `sceGxmDrawPrecomputed` draws with, laid
    /// over `record_draw`'s snapshot of the context block for that one draw. Set only around
    /// that call - see `draw_precomputed` for why the context itself is never written.
    pdraw_binding: Option<(u32, [u32; pdraw::STREAM_SLOTS])>,
    /// One entry per draw pushed into the CURRENT scene, in the same order, when
    /// [`defer_geometry`] is on: what that draw's vertex and index read needs, to be performed
    /// at `sceGxmEndScene`. Emptied there.
    deferred_geometry: Vec<DeferredGeometry>,
    /// The scenes already ENDED whose geometry is still unread, each under the serial its
    /// `Scene::deferred_id` carries: `sceGxmEndScene` moves the current scene's descriptions
    /// here, and [`Self::resolve_deferred_geometry`] reads them at the guest's GPU wait or its
    /// flip. An entry whose scene the capture has since evicted is dropped unread.
    pending_geometry: Vec<(u64, Vec<DeferredGeometry>)>,
    /// Under [`async_flip_resolve`]: the last deferred serial a FLIP owes a resolve for, left
    /// for the presenting worker ([`VitaEnv::resolve_flipped_geometry`]) instead of being read
    /// on the flipping guest thread. Any guest sync point resolves it first.
    flip_resolve_upto: Option<u64>,
    /// The last serial handed out - never 0, which is "resolved" in the scene.
    deferred_serial: u64,
    /// Guest address of the fallback SA bank - the uniforms a draw reads when no default
    /// uniform buffer is bound for its stage. Placed once in [`Self::set_alloc_base`],
    /// before any guest code runs, and published to the host-mirror block so an inlined
    /// `sceGxmSetUniformDataF` can reach it; see [`SA_BANK_DATA`] for the layout and for
    /// why it is in guest memory at all. Zero if the arena could not place one.
    sa_bank: u32,
    /// Scratch for [`crate::vita::gxmctx::texture_bindings`], kept so the hottest path in the
    /// engine does not allocate a `Vec` per draw. Never read between draws.
    bound_binding_scratch: Vec<(u32, crate::vita::gxmctx::TexBinding)>,
    // Threads the program created, and any pending synchronous thread run raised
    // by sceKernelStartThread (drained by the engine host after the call).
    threads: Vec<ThreadRec>,
    /// Stacks belonging to deleted threads, as `(base, size)`. NOT a free list - see
    /// [`VitaState::create_thread`] for why they are not recycled. Kept so the leak a
    /// thread-churning title causes is a number someone can look at rather than an
    /// invisible drift toward an allocator collision.
    free_stacks: Vec<(u32, u32)>,
    /// Whether the runaway-thread-creation warning has been emitted (once per run).
    runaway_threads_reported: bool,
    /// Whether "video playback is not implemented" has been reported (once per run).
    /// See `vita::video`.
    pub(crate) reported_no_video: bool,
    /// Whether the "the unit layout is partly inferred" line has been said this run.
    pub(crate) reported_unit_layout: bool,
    /// The movie session `SceMp4` is playing, if any. One at a time: the call surface has
    /// a single handle and a title plays one movie at a time.
    pub(crate) movie: Option<crate::vita::video::MovieSession>,
    /// Every `SceAvPlayer` player alive - see [`crate::vita::avplayer`].
    pub(crate) avplayers: crate::vita::avplayer::Players,
    /// The guest's own H.264 decoders and the codec engine's memory bookkeeping.
    /// See [`crate::vita::avcdec`].
    pub(crate) avcdec: crate::vita::avcdec::AvcdecState,
    /// Memory spaces that have reported an exhausted allocation, so the report is once per
    /// space rather than once per failure - see `vita::libkernel::mspace_alloc`.
    pub(crate) mspace_exhausted: std::collections::HashSet<u32>,
    pending_reentry: Option<Reentry>,
    /// The replay records of host calls that are calling back into the title - see
    /// [`Self::call_guest`] and [`GuestCalls`], which owns them.
    guest_calls: GuestCalls,
    /// The guest buffer a movie read through the title's file functions lands in when the
    /// title gave the library no allocator - one for the run, allocated on first use. See
    /// [`Self::mp4_io_scratch`].
    mp4_io_scratch: u32,
    /// Every kernel synchronization object and every thread parked on one - see
    /// [`SyncObjects`], which owns that state outright.
    sync: SyncObjects,
    /// Virtual timers (`sceKernelCreateTimer` and friends), stopwatches over the same virtual
    /// clock every other guest-visible time call reads ([`now_us`](Self::now_us)) - see
    /// [`KernelTimers`], which owns them.
    timers: KernelTimers,
    /// One bit per open SceCommonDialog family (see `vita::services::DialogFamily`):
    /// set by `*DialogInit`, read by `*DialogGetStatus` (RUNNING for a short while, then
    /// FINISHED - see `vita::services::dialog_get_status`), cleared by `*DialogTerm`.
    pub(crate) open_dialogs: u32,
    /// The display flip count each dialog family was opened at, indexed like
    /// `open_dialogs`' bits.
    pub(crate) dialog_opened_flip: [u64; 16],
    /// Registered FIOS2 path overlays and the threads that opted out of them - see
    /// [`FiosOverlays`], which owns them.
    fios_overlays: FiosOverlays,
    /// The debug GPIO output register (`sceKernelSetGPO`). On a development unit its
    /// low bits drive the board's diagnostic LEDs; retail hardware has none wired, so
    /// nothing observable comes of a write. It is still a register and the value is
    /// still held, so it is held here rather than discarded - a title that toggles it
    /// as a progress marker through boot leaves its last marker readable.
    pub(crate) gpo: u32,
    /// SceTouch's state between calls - see [`crate::vita::touch::TouchState`], which owns it.
    pub(crate) touch: crate::vita::touch::TouchState,
    // Preemptive-mode state (unused in the single-thread model). When `preemptive`
    // is set, blocking primitives actually park the calling thread (`current`) and
    // are woken by another thread's signal/unlock/thread-end; the scheduler drains
    // `pending_spawns` (threads to start as their own fibers) and `pending_wakes`
    // (parked threads made runnable) after each host call. See the runtime README
    // concurrency model and `vitaslop_native::ThreadedScheduler`.
    preemptive: bool,
    current: i32,
    /// Threads parked in `sceKernelWaitSignal`. A plain id list: the wait has no
    /// count or pattern to match, only "a signal arrived for me".
    signal_waiters: Vec<i32>,
    /// Display-queue callback execution (preemptive mode; see
    /// [`Self::enqueue_display_callback`]), one run per entry carrying its callback-data slot.
    /// Serial - one callback at a time, in submission order - matching the real display
    /// queue's single internal thread. Its idle waiters are the threads parked in
    /// `sceGxmDisplayQueueFinish` (see [`Self::display_queue_finish`]).
    display_callbacks: CallbackQueue<u32>,
    /// NGS callback execution (preemptive mode; see [`Self::enqueue_ngs_callback`]): the same
    /// machine as the display queue's, each run carrying `(entry, info slot)`. Serial, in the
    /// order the events happened.
    ngs_callbacks: CallbackQueue<(u32, u32)>,
    /// Per thread: the NGS callbacks its `sceAudioOutOutput` is running on it, and their info
    /// ring - see [`NgsSyncStash`] and [`VitaState::ngs_sync_stash`].
    ngs_sync: NgsSyncStash,
    // Registered service-state callbacks (preemptive mode). A title registers
    // these at boot and pumps them every frame (`sce*CheckCallback`), waiting for
    // the one-time state notification (offline: signed-out / disconnected) before
    // it leaves its boot screen. See [`OneShotCallback`] and [`Self::pump_np_callback`].
    np_service_cb: OneShotCallback,
    net_inet_cb: OneShotCallback,
    /// (waiter thread id, target thread id, stat pointer) for threads parked in a
    /// join. `stat` is the guest `int *` the joiner passed to `sceKernelWaitThreadEnd`
    /// (0 = NULL); the target's exit code is written there when the join completes.
    join_waiters: Vec<(i32, i32, u32)>,
    /// Kernel callbacks (`sceKernelCreateCallback`) and their delivery threads - see
    /// [`KernelCallbacks`], which owns them.
    kcallbacks: KernelCallbacks,
    /// `sceKernelRegisterThreadEventHandler` registrations and which threads' EXIT handlers
    /// have run - see [`ThreadEvents`], which owns them.
    thread_events: ThreadEvents,
    pending_spawns: Vec<Reentry>,
    /// Fiber threads and the REAL thread that first ran them: `(fiber thid, runner)`. A fiber
    /// is not a thread on the hardware - it runs on its caller's core at its caller's priority -
    /// so a multi-worker scheduler places the thread backing it beside that runner (see
    /// [`Self::spawn_near`]).
    spawn_near: Vec<(i32, i32)>,
    pending_wakes: Vec<i32>,
    /// PARKED threads the runtime has ended, for the scheduler to drop without resuming -
    /// see [`Self::take_terminations`] and `retire_fiber_thread`.
    pending_terminations: Vec<i32>,
    /// Threads released by a TIME passing (a timed wait expiring, a sleep ending) since the
    /// scheduler last asked - see [`Self::take_timeout_wakes`].
    timeout_wakes: Vec<i32>,
    /// `(woken, waker thid, source file:line)` per wake while [`Self::trace_wakes`] is on - who
    /// released whom, for the SMP timeline (a wake is drained by whichever worker gets there first).
    wake_causes: Vec<(i32, i32, &'static str)>,
    pub trace_wakes: bool,
    /// Guest memory writes to apply when a blocked joiner is woken: `(stat_ptr,
    /// exit_code)`. `sceKernelWaitThreadEnd`'s handler cannot re-run at wake time, so
    /// the exit code it owes the joiner's `stat` out-parameter is queued here and
    /// applied by the scheduler (which has memory access) before the joiner resumes.
    pending_stat_writes: Vec<(u32, u32)>,
    /// The `r0` value to hand a parked thread when it resumes, keyed by thread id:
    /// `(thid, code)`. A blocking wait sets its return value (`ctx.ret(0)`) *before* it
    /// parks, so a wake that must return something other than 0 - a timed wait that
    /// expired, returning `SCE_KERNEL_ERROR_WAIT_TIMEOUT` - cannot write it then.
    /// Instead the timeout is queued here and the engine applies it to the woken
    /// thread's `r0` at the point it resumes (see the native/browser schedulers).
    /// Absent (the common case, a signal wake) leaves the pre-park `r0` (0) in place.
    pending_resume_codes: Vec<(i32, u32)>,
    // Virtual filesystem backing SceIoFilemgr (open/read/write/lseek/close).
    fs: FileTable,
    /// Installed add-on directory names (see [`VitaState::add_addcont`]).
    addcont: Vec<String>,
    // In-memory savedata slot metadata (SceAppUtil), so a created slot round-trips on get.
    savedata: SaveDataStore,
    /// SceAppUtil safe memory: [`SAFE_MEMORY_SIZE`] bytes once the title has written any
    /// (empty until then, which reads as zeroes), and whether it changed since the last flush.
    safe_memory: Vec<u8>,
    safe_memory_dirty: bool,
    /// SceNpTrophy: the sets read out of the title's own TRP plus this run's unlock
    /// ledger. Public to the NID handlers in `vita/services.rs`.
    pub(crate) trophies: crate::trophy::TrophyStore,
    /// `sceClibMspace*`: general allocators over blocks of the title's own memory.
    pub(crate) mspaces: crate::mspace::MspaceStore,
    /// ScePvf font engine: open lib/font handles, size config, glyph cache. Public so
    /// the ScePvf NID handlers in `vita/pvf.rs` can drive it.
    pub fonts: crate::font::FontLibrary,
    pub capture: Capture,
    // `Send` so a `VitaEnv` can be the data of a wasmtime async Store (the
    // cooperative scheduler runs the guest on a fiber, which wasmtime may resume
    // on any thread). Everything stays single-threaded in practice.
    pub world: Box<dyn World + Send>,
    /// The audio-output sink every `sceAudioOut` port feeds. A `NullSink` by default
    /// (silent), replaced by a host that wants real playback (native device / Web
    /// Audio). See [`crate::audio`].
    pub audio: Box<dyn crate::audio::AudioSink + Send>,
    /// Opens video decoders for movie playback. `NoVideo` by default - which is what a
    /// host that never set one gets, and it reports "no decoder" rather than pretending a
    /// movie played. The front-ends install a real one (`H264Factory`), which reaches
    /// Media Foundation / Video Toolbox / VA-API / WebCodecs depending on where this runs.
    ///
    /// The engine links no decoder itself: this is a port, exactly like `audio` and
    /// `world`, so the browser blob does not carry a native decoder and vice versa.
    pub video: Box<dyn vitaslop_platform::video::VideoDecodeFactory>,
    /// Opens AUDIO decoders for a movie's sound track, on the same terms as `video` above:
    /// `NoAudioDecode` by default, a real one installed by the front-ends. A movie whose
    /// pictures decode and whose sound does not is a movie playing silently, and this is
    /// what the engine says so with rather than guessing.
    pub audio_dec: Box<dyn vitaslop_platform::audio_dec::AudioDecodeFactory>,
    /// Opened NGS/audio port and handle bookkeeping (see `vita::ngs` / `vita::audio`).
    pub(crate) audio_state: crate::vita::audio::AudioState,
    /// SceVoice's ports. A console with no microphone and no session: see
    /// [`crate::vita::voice`].
    pub(crate) voice: crate::vita::voice::VoiceState,
    /// SceJpeg's one decode cache, so the colour-space conversion does not have to invert
    /// the planes the decode wrote. See [`crate::vita::jpeg`].
    pub(crate) jpeg: crate::vita::jpeg::JpegState,
    /// The shader patcher's live programs, their reference counts and the vertex programs it
    /// shares by creation parameters - see [`ProgramRefs`], which owns them.
    program_refs: ProgramRefs,
    /// `SceAudiodec` decoders the title created for a movie's sound - see
    /// [`crate::vita::audiodec`].
    pub(crate) audiodec: crate::vita::audiodec::AudiodecState,
    /// SceLibLocation handles and permission-dialog state (see `vita::location`). The
    /// position itself is never stored here - it is read from [`World`] at each call, so
    /// a fix cannot go stale behind the guest's back.
    pub(crate) location: crate::vita::location::LocationState,
    /// Bring-up aid: halt the run when the guest calls sceGxmTerminate. The cube
    /// entry is `_start`, which spins forever after `main` returns (there is no OS
    /// to exit to yet), so terminate is the clean stopping point after teardown.
    pub halt_on_terminate: bool,
    /// Guest address of the main module's `SceProcessParam`, returned verbatim by
    /// `sceKernelGetProcessParam`. libc's crt reads the `SceLibcParam` it points to
    /// for the heap configuration, so this must be a real address (0 would fault).
    process_param: u32,
    /// What the title's `SceProcessParam` asks the MAIN thread to run at (see
    /// `LinkedProgram::main_thread_request`): its priority and its CPU affinity mask (0 =
    /// every user core). The main thread has no create record, so these stand in for one.
    main_priority: i32,
    main_affinity: i32,
    /// The modules in the linked image (from [`crate::link::LinkedProgram`]), which
    /// is what the kernel's module queries answer from. Set once before the run.
    modules: Vec<crate::link::LoadedModule>,
    /// The title's function imports with no dispatch arm (from
    /// [`crate::link::LinkedProgram::unhandled_imports`]), so the report of the one that
    /// stopped the run can name every other one the title could stop on.
    unhandled_imports: Vec<(u32, u32)>,
    /// Each thread's `__thread` block and `sceKernelGetTLSAddr` slot array - see
    /// [`ThreadLocalStorage`], which owns them.
    tls: ThreadLocalStorage,
    /// Fragment textures currently bound by `sceGxmSetFragmentTexture`, keyed by
    /// sampler unit -> guest `SceGxmTexture*`. Bindings persist across draws until
    /// rebound (GXM state is sticky), so this is read - not cleared - at each draw.
    bound_textures: Vec<TextureBinding>,
    /// Textures bound to VERTEX-stage sampler units, sorted by unit and separate from
    /// [`Self::bound_textures`] because the two stages number their units independently. A
    /// parse of the context block's vertex sampler array, refreshed per draw by
    /// [`Self::sync_vertex_textures`] - the array, not this list, is the state.
    bound_vertex_textures: Vec<TextureBinding>,
    /// The vertex sampler array's bytes as [`Self::bound_vertex_textures`] was last parsed
    /// from - what [`Self::sync_vertex_textures`] compares against.
    vertex_texture_span: [u8; crate::vita::gxmctx::TEXTURE_ARRAY_BYTES as usize],
    /// Counts `begin_scene`s - the "same scene" test of [`DeferredTextures::span`].
    tex_scene_seq: u64,
    /// The previous deferred draw's sampler block, fragment program and scene - see
    /// [`Self::defer_draw_textures`].
    deferred_tex_last: Option<(Box<[u8]>, u32, u64)>,
    /// The vertex bindings last handed to a deferred draw, by generation, so a run of draws
    /// shares one `Arc`.
    deferred_vert_last: Option<(u64, Arc<[TextureBinding]>)>,
    /// How many times [`Self::sync_vertex_textures`] has seen the vertex sampler array CHANGE.
    ///
    /// The byte compare happens once per draw, there; everything downstream asks this one
    /// counter instead - and it needs asking, because a title that binds a vertex sampler on
    /// every one of ~626 draws a frame almost always binds the same one.
    vertex_texture_gen: u64,
    /// The previous draw's parsed fixed-function render state, with the two context-block
    /// spans it was parsed from.
    ///
    /// The state is ~50 fields read one word at a time out of the snapshot and ~240 bytes of
    /// the `Draw` it lands in, and a run of draws inside one pass carries the identical state.
    /// Two `memcmp`s (132 + 24 bytes, the two contiguous regions the state occupies) answer
    /// "the same as last time?", and a hit is an `Arc` clone instead of a parse and a copy.
    render_state_memo: Option<(
        Box<[u8]>,
        Box<[u8]>,
        std::sync::Arc<crate::capture::RenderState>,
    )>,
    /// Scratch for decoding a draw's vertex uniform bank into floats.
    ///
    /// The bank is read as BYTES (that is what the recompiled shader wants) and the reflection
    /// below wants floats, so every draw converted the whole thing into a fresh `Vec<f32>` -
    /// an allocation and a free per draw, ~626 a frame, for a buffer whose only surviving
    /// consumer on the recompiled path is the first sixteen lanes. Reused here and handed back
    /// at the end of the draw.
    uniform_float_scratch: Vec<f32>,
    /// Each guest texture's recorded format and palette, and the nearby-texture memo - see
    /// [`TextureRegistry`], which owns them.
    textures: TextureRegistry,
    /// Scratch for [`Self::snapshot_bound_textures`]'s per-unit pass, kept so the hottest
    /// function in the capture does not allocate a `Vec` on every draw.
    ///
    /// It exists at all because the per-unit control state must be read through a shared borrow
    /// of `self` while the snapshot cache is borrowed mutably; `mem::take` on a field is what
    /// lets that be one reused buffer rather than a fresh allocation per draw. On this title's
    /// race that is ~644 draws a frame, and in the browser an allocation is a good deal dearer
    /// than it is here.
    texture_unit_scratch: Vec<(TextureBinding, Option<u32>, Option<(i64, u32)>, u32)>,
    /// Per-scene texture-byte snapshots, keyed by (guest data address, byte length), so a
    /// texture bound by hundreds of draws is read from guest memory once and shared. Cleared
    /// at `beginScene` - see the note in `decode_texture` for why that is the right scope.
    texture_snapshots: Arc<std::sync::Mutex<TextureSnapshots>>,
    /// Every guest stack handed out - see [`GuestStacks`]. Shared with `texture_snapshots`.
    guest_stacks: GuestStacks,
    /// `sceGxmSetYuvProfile`: the `SceGxmYuvProfile` of each of the two YUV conversion slots
    /// (`CSC0`/`CSC1`, which a YUV texture's swizzle selects). Both start at 0,
    /// `SCE_GXM_YUV_PROFILE_BT601_STANDARD`, the conversion used before any call.
    pub(crate) yuv_profiles: [u32; 2],
    /// What the host keeps about precomputed states and draws that guest memory cannot vouch
    /// for - see [`PrecomputedRecords`], which owns it.
    precomputed: PrecomputedRecords,
    /// Frame boundaries owed to [`Self::texture_snapshots`] because it was held when the flip
    /// came - applied by the next `snaps!`.
    ///
    /// >>> A FLIP MUST NOT WAIT FOR THE RESOLVER. The async flip resolve reads the frame's
    /// geometry on its own worker WITH this cache locked, starting the moment the flip queues
    /// it; the flip then reached `on_frame_boundary` - holding the SMP state lock AND the host
    /// lock - and blocked on this mutex for the whole resolve. MEASURED on the phone at a baseball title's
    /// at-bat: 7-11 ms per frame during which no guest thread ran, no idle clock step could be
    /// taken and the run worker could not even read the scheduler state (~9 of ~58 ms a frame).
    snaps_frames_pending: std::sync::atomic::AtomicU32,
    /// See [`AsyncResolve`].
    async_resolve: Arc<AsyncResolve>,
    // The precomputed states themselves live in GUEST memory (`vita::gxmstate`) - the struct
    // plus an arrays block this engine allocates from the guest heap - which is what lets the
    // per-draw binds be inlined and lets a state the guest `memcpy`s keep working (the same
    // by-value fix the precomputed-DRAW family got, see [`pdraw`]).
    // Two host-side texture diagnostics used to live here - "did the current bindings come
    // from a precomputed state" and "which handles were live when they were bound". Both are
    // now properties of the BINDING, in the context block: `from_precomputed` is per unit
    // (a global one mislabels whichever path did not happen last), and "live at bind" is
    // exactly "the copied control words are not all zero", because the copy happens AT the
    // bind. Keeping host mirrors of them would have been worse than redundant once the bind
    // went inline: the handler stops running, so they would have read as false for every
    // binding in the run while looking like measurements.
    /// Set once, when a single scene has asked for more default-uniform bytes than the
    /// ring holds, so the wrap is reported exactly once instead of every frame.
    ///
    /// The RING ITSELF is not here: it lives in the guest's context block
    /// ([`crate::vita::gxmctx::off::UNIFORM_RING_BASE`]), which is where the hardware keeps
    /// it and what makes `sceGxmReserve*DefaultUniformBuffer` inlinable. Only this
    /// once-per-run report is host state, because "have I already said this" is not a fact
    /// about the guest.
    uniform_ring_wrapped: bool,
    /// Said once, if a reserve ever arrives with no context block to record into.
    reported_reserve_without_context: bool,
    /// How many times each `(stage, bound-for program, drawing program)` triple has been
    /// seen by [`Self::stale_uniforms`]. A COUNT, not a set: the warning fires on powers of
    /// ten, so a run says whether this happened to four draws or to four hundred thousand
    /// without printing hundreds of lines a frame. "Once per pair" hides the scale, and the
    /// scale is the whole question when the suspicion is that a stage is being starved of its
    /// uniforms across a title.
    reported_stale_uniforms: FxHashMap<(&'static str, u32, u32), u64>,
    /// The (program, buffer) pairs whose memory window [`Self::report_mem_window`] has tested.
    mem_windows_zero_tested: FxHashSet<(u32, u32)>,
    /// The raw 64-bit colour targets this frame's ended scenes rendered, as the renderer's
    /// window fill copies from them - see `capture_mem_windows`. Cleared at every flip, as the
    /// renderer clears the set it fills from.
    frame_raw_targets: Vec<vitaslop_platform::gpu::RawFillTarget>,
    /// The GPU notification region: a guest buffer of `SCE_GXM_NOTIFICATION_COUNT`
    /// u32 slots handed out by `sceGxmGetNotificationRegion`, lazily allocated on
    /// first use (0 = not yet allocated). Scenes complete synchronously here, so a
    /// notification the guest waits on is treated as already signalled.
    notification_region: u32,
    /// Occlusion queries - the visibility buffer, the scene's counts, each context's
    /// back-face test - see [`VisibilityQueries`] and [`Self::accumulate_visibility`].
    visibility: VisibilityQueries,
    /// GPU memory mappings from `sceGxmMapMemory(base, size, attr)`, as base -> size.
    /// Needed only so `sceGxmUnmapMemory(base)` - which is given no size - can drop the
    /// texture snapshots taken from that range: after an unmap the guest is free to
    /// reuse the pages, and a snapshot cached against those addresses would then be
    /// sampled as though it were still the texture that lived there.
    gxm_mappings: std::collections::BTreeMap<u32, u32>,
    /// What each render target was created with - its driver memory block and its extent,
    /// the authoritative size of every scene begun on it - see [`RenderTargets`] and
    /// [`Self::render_target_extent`].
    render_targets: RenderTargets,
    /// How many images `scePhotoExportFromData` has written, so each export gets its
    /// own path instead of overwriting the last.
    photo_exports: u32,
    /// The offline network - SceNet, SceHttp, SceNetAdhocMatching - which [`NetState`] owns.
    net: NetState,
    /// SceSystemGesture's process-wide PRIMITIVE layer: the touches derived from the samples
    /// the title feeds it, which every recognizer advances against - see `vita::gesture`.
    /// (Recognizer state itself lives in the guest's own recognizer blocks.)
    pub(crate) gesture: crate::vita::gesture::PrimitiveLayer,
    /// SceMotion's held settings - see [`crate::vita::services::MotionState`], which owns them.
    pub(crate) motion: crate::vita::services::MotionState,
    /// ScePower's held settings - see [`crate::vita::services::PowerState`], which owns them.
    pub(crate) power: crate::vita::services::PowerState,
    /// The shell's shared framebuffer, once a system-mode homebrew opens it:
    /// `(uid, base of two 960x544 RGBA8 buffers)`.
    pub shared_fb: Option<(i32, u32, u8)>,
    /// The `sce::Xml` DOM: the node arena every parsed document lives in, plus the
    /// interned guest copies of the strings handed back. See `vita::sce_xml`.
    pub xml: crate::vita::sce_xml::XmlState,
    /// NpCommon/NpBasic/NpCommerce2/Lookup/WordFilter/Near state - see `vita::np`.
    pub np: crate::vita::np::NpState,
    /// What SceNet / SceHttp / SceSsl were initialised with, and SceLibRudp's objects - see
    /// `vita::net::NetLibs`.
    pub net_libs: crate::vita::net::NetLibs,
    /// The on-screen keyboard a game has open - see `vita::ime`.
    pub ime: crate::vita::ime::ImeState,
    /// Every `SceFiber` the guest has initialised and the state of the chains - see
    /// [`FiberTable`], which owns them.
    fibers: FiberTable,
    /// Threads sleeping until a virtual-clock deadline, as `(thread id, deadline_us)`.
    /// Unlike a lwcond waiter these are woken purely by time - `sceKernelDelayThread`
    /// and, crucially, `sceAudioOutOutput` pacing: an audio mixer thread hands one
    /// grain per call and on hardware blocks until it drains, which paces it to real
    /// time. Parking it here means the audio thread stops busy-spinning and, because
    /// it now yields, the scheduler goes idle and can advance the clock - so a title's
    /// time-based loading wait actually progresses instead of being starved.
    sleep_waiters: Vec<(i32, u64)>,
    /// Threads parked until a HOST WALL-CLOCK time (microseconds, [`host_wall_us`]) - see
    /// [`VitaState::wall_park`].
    wall_waiters: Vec<(i32, u64)>,
    /// Whether a zero-length yield has anyone to give the CPU to, and the mirror word that
    /// lets the inlined yield answer it - see [`YieldElision`] and
    /// [`VitaState::yield_would_repick_this_thread`].
    yields: YieldElision,

    /// The virtual monotonic clock in microseconds, read by
    /// `sceKernelGetProcessTimeWide`/`GetSystemTimeWide`. The scheduler advances it
    /// (jumping to the earliest pending deadline when every thread is parked), so a
    /// timed wait costs one round instead of millions of busy-poll iterations.
    virtual_us: u64,
    /// Which caller is currently advancing the clock, for the [`clock_sources`] split.
    ///
    /// `Idle` is the DEFAULT rather than an `Option`, so an advance made from a path
    /// nobody tagged is attributed to the scheduler's idle jump instead of disappearing.
    /// A bucket that silently drops time would make the split add up to less than the
    /// clock and quietly invite the wrong conclusion.
    clock_source: ClockSource,
    clock_from_quanta_us: u64,
    clock_from_topup_us: u64,
    clock_from_idle_us: u64,
    /// Game clock gained from the wall-clock floor - see [`VitaState::wall_floor_tick`].
    clock_from_wall_us: u64,
    /// `(wall ms at the last tick, floor us)` once the wall-clock floor has started.
    wall_floor: Option<(f64, u64)>,
    /// WHO the idle jumps were made for: microseconds and jump count, per
    /// `(wait kind, thread id)`. See [`VitaState::idle_attribution`].
    idle_by_owner: Vec<(IdleOwner, u64, u64)>,
    /// The current display-frame index, updated each flip by the scheduler
    /// (`on_frame_boundary`). Frame-tags egress-ledger events so a check can establish
    /// roughly when a milestone occurred, not only that it did.
    cur_frame: u64,
    /// SceCtrl's state between calls - see [`crate::vita::ctrl::CtrlState`], which owns it.
    pub(crate) ctrl: crate::vita::ctrl::CtrlState,
}

/// The ceiling on a default uniform buffer, in 4-byte registers. A program header or parameter
/// record that asks for more than this is one we are misreading, and the clamp keeps a bad read
/// from turning into an allocation.
pub(crate) const MAX_DEFAULT_UNIFORM_REGS: u32 = 4096;

/// Byte offset of the first float in the fallback SA bank. The word before it is the
/// high-water float count.
///
/// The bank lives in GUEST memory, and that is what makes `sceGxmSetUniformDataF` -
/// **1,106 calls a frame on one title, 58% of everything it still calls** - an inlinable
/// pair of `memory.copy`s instead of a boundary crossing. It is engine scratch rather than
/// device state (hardware has no such bank: a title writes into the buffer it names, full
/// stop), but the rule that decides where it goes is the same one `gxmctx` rests on - a
/// fact a hot call needs must live where guest code can reach it, or the call cannot be
/// inlined at all. [[vitaslop-guest-state-is-what-makes-a-call-inlinable]]
pub(crate) const SA_BANK_DATA: u32 = 4;

/// Total bytes the bank occupies: the high-water word plus a float per register the bank
/// admits ([`MAX_DEFAULT_UNIFORM_REGS`], which is also the ceiling `set_uniforms` refuses
/// past).
const SA_BANK_BYTES: u32 = SA_BANK_DATA + MAX_DEFAULT_UNIFORM_REGS * 4;

/// Which of the two shader stages a program handle belongs to. The stages differ in one
/// thing only - which host map resolves a handle to its `SceGxmProgram *` - so the code
/// that reserves a default uniform buffer is written once and told which map to ask.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProgramStage {
    Vertex,
    Fragment,
}

/// Queue `thid` to be made runnable (`pending_wakes`); with `trace_wakes` on, note who woke it
/// and from which source line. A macro over the FIELDS, not a method, so it works inside a
/// closure that already borrows another field of the state.
macro_rules! push_wake {
    ($s:ident, $thid:expr) => {{
        let thid: i32 = $thid;
        if $s.trace_wakes {
            $s.wake_causes.push((thid, $s.current, *crate::site!()));
        }
        $s.pending_wakes.push(thid);
    }};
}

/// The [`Sched`] a [`SyncObjects`] operation runs against, borrowed from `$s`'s own fields. A
/// macro over the FIELDS, like `push_wake!`, so the borrow is of those fields alone and
/// `$s.sync` stays free to call into beside it.
macro_rules! sched {
    ($s:ident) => {
        Sched {
            now: $s.virtual_us,
            current: $s.current,
            wakes: &mut $s.pending_wakes,
            causes: &mut $s.wake_causes,
            trace: $s.trace_wakes,
            resume_codes: &mut $s.pending_resume_codes,
            stat_writes: &mut $s.pending_stat_writes,
        }
    };
}

/// Report - once per (fragment, bound vertex, patched-against vertex) - a draw that binds a
/// fragment program with a vertex program OTHER than the one the fragment was created against.
/// The hardware feeds such a fragment by output-lane POSITION from the bound vertex, laid out
/// as the creation-time vertex placed them; see `capture::Draw::fprog_patched_vprog`.
fn report_patched_against_other_vertex(fheader: u32, vheader: u32, patched: u32) {
    use std::collections::HashSet;
    use std::sync::Mutex;
    // Per such draw: a triple this thread has seen returns before the global lock.
    thread_local! {
        static DONE: std::cell::RefCell<crate::fasthash::FxHashSet<(u32, u32, u32)>> = std::cell::RefCell::new(Default::default());
    }
    if !DONE.with(|d| d.borrow_mut().insert((fheader, vheader, patched))) {
        return;
    }
    static SEEN: Mutex<Option<HashSet<(u32, u32, u32)>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !g
        .get_or_insert_with(HashSet::new)
        .insert((fheader, vheader, patched))
    {
        return;
    }
    tracing::info!(
        target: "vitaslop::gxm",
        "draw binds fragment program {fheader:#010x} with vertex program {vheader:#010x}, but the          fragment was PATCHED against vertex program {patched:#010x} - its varyings are fed by          output-lane position as that program laid them out"
    );
}

/// >>> TIMED WAITS THAT EXPIRED, with what produced the guest time they waited through.
///
/// A timeout the console would never reach is a timeout this engine's CLOCK reached: the guest
/// time between the wait and its deadline came from somewhere - threads running (quanta), the
/// wall-clock floor pushing the clock to real time, or idle jumps to the next deadline - and on
/// a slow device the answer is what decides whether a title's handshake survives. MEASURED why
/// (runner jobs 061/062): a movie's sound thread timed out on a semaphore on a phone, quit, and
/// the movie never advanced; the desktop never timed out. Always on - an expiry is rare and one
/// line; capped at [`TIMED_OUT_CAP`], newest kept first-come.
static TIMED_OUT: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
pub const TIMED_OUT_CAP: usize = 200;
static TIMED_OUT_DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn note_timed_out_wait(line: String) {
    if let Ok(mut v) = TIMED_OUT.lock() {
        if v.len() < TIMED_OUT_CAP {
            v.push(line);
        } else {
            TIMED_OUT_DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// The expired timed waits so far (see [`note_timed_out_wait`]), and how many past the cap.
pub fn timed_out_waits() -> (Vec<String>, u64) {
    (
        TIMED_OUT.lock().map(|v| v.clone()).unwrap_or_default(),
        TIMED_OUT_DROPPED.load(std::sync::atomic::Ordering::Relaxed),
    )
}

// The rest of the host lives in `host/`, one concern per file. Every child starts with
// `use super::*;`, so they are declared below the two macros above, which are textual.
mod arena;
mod callbacks;
mod color_surfaces;
mod display;
mod env;
mod fibers;
mod files;
mod fios_overlays;
mod guest_calls;
mod kcallbacks;
mod memory;
mod net;
mod precompile;
mod precomputed_records;
mod program_memos;
mod program_refs;
mod reflect;
mod resolve;
mod shader_patcher;
mod snapshot_cache;
mod snapshots;
mod storage_clock;
mod sync;
mod texture_decode;
mod texture_registry;
mod thread_events;
mod timers;
mod tls;
mod visibility;
mod vertex_layout;
mod yield_elision;
pub use arena::*;
use callbacks::*;
use color_surfaces::*;
use display::*;
pub use env::*;
use fibers::*;
pub use files::*;
pub use fios_overlays::*;
use guest_calls::*;
use kcallbacks::*;
pub use memory::*;
pub use net::*;
use precompile::*;
use precomputed_records::*;
use program_memos::*;
use program_refs::*;
pub(crate) use reflect::*;
pub use resolve::*;
use shader_patcher::*;
pub use snapshot_cache::*;
use storage_clock::*;
pub use sync::*;
pub use texture_decode::*;
use texture_registry::*;
use thread_events::*;
use timers::*;
use tls::*;
use visibility::*;
pub(crate) use vertex_layout::*;
use yield_elision::*;

// `impl VitaState`, split by subsystem.
mod state_clock;
mod state_core;
mod state_debug;
mod state_display;
mod state_draw;
mod state_gxm_bind;
mod state_gxm_programs;
mod state_kobjects;
mod state_mem_windows;
mod state_store_windows;
mod state_net;
mod state_precomputed;
mod state_resolve;
mod state_shaders;
mod state_storage;
mod state_sync;
mod state_threads;
mod state_uniforms;

#[cfg(test)]
mod dirty_runs_tests;
#[cfg(test)]
mod fiber_tests;
#[cfg(test)]
mod filesystem_tests;
#[cfg(test)]
mod game_data_tests;
#[cfg(test)]
mod hostcall_tests;
#[cfg(test)]
mod preemptive_tests;
#[cfg(test)]
mod program_reflection_tests;
#[cfg(test)]
mod register_window_tests;
#[cfg(test)]
mod savedata_tests;
#[cfg(test)]
mod strided_texture_tests;
#[cfg(test)]
mod subsystem_tests;
/// The guest-store stamp mechanism, exercised against a memory that implements the dirty map
/// the way both real engines do.
///
/// # Why this is worth its length
/// The stamps are the difference between proving a texture unchanged for the cost of one byte
/// per 4 KB page and proving it by comparing every byte of it twice. MEASURED on a live race:
/// the compare is **44% of the whole frame and 105.8 MB moved per frame**. And the mechanism
/// fails SILENTLY - when the stamps stop working the picture is still perfectly correct, just
/// half the speed, so nothing in a capture points at it. It was in fact broken for its entire
/// life on the only engine that runs it, and the symptom was "the phone is slow".
///
/// The native engine does not stamp by default (wasmtime bills every operator, so the marks
/// would speed the guest clock up), which is exactly why these are unit tests over a fake
/// backing rather than something the desktop's own runs would have caught.
#[cfg(test)]
mod texture_snapshot_stamp_tests;
#[cfg(test)]
mod tls_slot_tests;
#[cfg(test)]
mod vertex_stream_column_tests;
