//! The preemptive cooperative scheduler: many guest threads, each its own
//! wasmtime instance on its own fiber, all sharing one guest address space, run
//! one-at-a-time on a single OS thread and switched at their blocking points.
//!
//! # Why one instance per thread
//! The transpiler keeps the ARM register file in wasm **globals** (see
//! [`abi`](vitaslop_transpiler::abi)); a global is per-instance, so two guest
//! threads sharing one instance would clobber each other's registers on every
//! switch. Giving each thread its own instance makes each register file naturally
//! private. What the threads must share - the guest's flat memory - is provided by
//! importing one wasm **shared** linear memory into every instance (the transpiler
//! emits this when [`import_memory`](vitaslop_transpiler::Program::import_memory)
//! is set). One address space, private registers: exactly the thread model.
//!
//! wasmtime forbids two in-flight calls on one `Store`, and a fiber belongs to the
//! call that spawned it, so N suspended guest stacks require N stores anyway. The
//! shared memory is the one object that legally crosses stores (the wasm threads
//! proposal), so it is what we lean on.
//!
//! # Why cooperative and single-threaded
//! Only one guest thread runs at any instant; a thread yields control only at a
//! host call (a blocking primitive, or a fuel-quantum preemption). Because no two
//! guest threads ever touch memory truly concurrently, the shared memory needs no
//! atomics for correctness here, and scheduling stays deterministic - the same
//! inputs drive the same interleaving. Real SMP (several guests running at once on
//! several OS threads) is a later step; this establishes the faithful blocking
//! semantics single-worker run-to-completion could not express.
//!
//! # The switch points
//! Each host call returns an [`SvcOutcome`]. [`Continue`](SvcOutcome::Continue)
//! keeps the thread running; [`Block`](SvcOutcome::Block) parks it until another
//! thread wakes it; [`Flip`](SvcOutcome::Flip) is a display frame boundary (and the
//! only thing that counts one - [`Reschedule`](SvcOutcome::Reschedule) is the
//! voluntary yield that does not);
//! [`ThreadExit`](SvcOutcome::ThreadExit) ends just this thread;
//! [`Halt`](SvcOutcome::Halt) ends the whole process. Thread creation and wakeups
//! are side channels the host and scheduler agree on through
//! [`ImportDispatch::take_spawns`] / [`take_wakes`](ImportDispatch::take_wakes).

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use vitaslop_runtime::sched::{
    FiberEnd, GuestEngine, GuestThread, Scheduler, Stop, ThreadHandle, ThreadStep,
};
use vitaslop_runtime::{ImportDispatch, Reentry, SvcOutcome, VFP_ARG_COUNT};
use vitaslop_transpiler::abi;
use vitaslop_transpiler::{self as transpiler};
use wasmtime::{Caller, Config, Engine, Instance, Linker, Module, SharedMemory, Store, Val};

use crate::RunError;

/// The scheduler verdict is shared policy; re-export it so `vitaslop_native`'s public
/// path (`threaded::RunReport`) is unchanged.
pub use vitaslop_runtime::sched::RunReport;

/// Fuel a thread may retire before the fiber yields back to the scheduler even
/// without a host call. A deterministic (retired-instruction) quantum so the
/// interleaving is reproducible; large enough that a normal run of guest code
/// between host calls completes in one slice.
///
/// It is the runtime's [`QUANTUM_FUEL`](vitaslop_runtime::host::QUANTUM_FUEL) rather than a
/// number of its own, because the same constant is the denominator the game clock prices
/// guest work against. Two engines that preempt at different intervals may still keep the
/// same clock - the charge is per unit of fuel - but the clock's calibration is stated
/// against this one, so it must not drift from it silently.
const DEFAULT_QUANTUM_FUEL: u64 = vitaslop_runtime::host::QUANTUM_FUEL;

/// Headroom reserved above the main thread's initial stack pointer, below the top
/// of the guest region. ELF/crt startup and some libc routines address memory at
/// and just above the initial SP (the argc/argv/env/auxv block a kernel would have
/// placed there, plus large formatting scratch buffers), so putting SP exactly at
/// the region end (`base + mem_bytes`) makes those touches run one past the last
/// valid byte and trap. This guard gives them valid slack. Spawned threads get
/// their own allocated stacks and do not need it.
const MAIN_STACK_HEADROOM: u32 = 0x0010_0000; // 1 MiB

/// The main thread's initial stack pointer: the top of the guest region minus the
/// startup headroom, 16-byte aligned (AAPCS at a public entry).
fn main_stack_top(base: u32, mem_bytes: u32) -> u32 {
    base.wrapping_add(mem_bytes).wrapping_sub(MAIN_STACK_HEADROOM) & !0xF
}

/// The one-word channel from a thread's host-call closure (running inside its
/// fiber, hence inside its store) out to the [`WasmtimeThread::resume`] that polled
/// it, which cannot otherwise see into the borrowed store. Only the [`Stop`] reason
/// needs to cross; exit codes and halt/exit kinds ride the fiber's own return value.
struct Signal {
    stop: Stop,
    /// Engine fuel this thread has burned in total, as of the last moment anything could
    /// observe it. The game clock is charged for the DIFFERENCE across a resume, so this
    /// is what makes a suspend cost what the guest actually executed.
    ///
    /// Two paths reach it, and both are exact. A host call can read `Caller::get_fuel`
    /// directly, so any suspend that goes through the closure below samples the true
    /// value. wasmtime's OWN periodic yield does not run any of our code - but it fires
    /// every [`quantum_fuel`](WasmtimeEngine::quantum_fuel) units by construction, so a
    /// resume that suspended without passing through the closure burned exactly that much,
    /// and [`WasmtimeThread::resume`] adds it.
    ///
    /// Getting this second path right is not an edge case: a title's vblank spin reads the
    /// clock through an INLINED mirror load and makes no host call at all, so it suspends
    /// only on the periodic yield. Charging it nothing is a stopped clock, which is the
    /// exact livelock the CPU charge exists to prevent.
    fuel: u64,
    /// GUEST ARM INSTRUCTIONS this thread has retired in total, from the emitted per-block
    /// counter. This - not [`Self::fuel`] - is what the game clock is billed in, because an
    /// ARM instruction is the same amount of guest work whatever the codegen turns it into.
    /// See `sched::ThreadHandle::arm_retired`.
    arm: u64,
    /// How many times the host-call closure has suspended this thread. `resume` compares
    /// it across a poll to tell a suspend it can see (a host call, which sampled `fuel`)
    /// from one it cannot (wasmtime's periodic fuel yield).
    host_suspends: u64,
}

/// The scheduler's per-thread store data: the shared host, this thread's id, the
/// shared memory (so a host call can view guest memory), the image base, and the
/// flags the host-call closure raises for the fiber's return value.
struct ThreadData<H: ImportDispatch + Send + 'static> {
    host: Arc<Mutex<H>>,
    thid: i32,
    shared_mem: SharedMemory,
    base: u32,
    /// Linear-memory offset of the guest-store dirty block, when this build was
    /// transpiled with one (`VITASLOP_DIRTY_PAGES`). `None` in an ordinary NATIVE
    /// build, and deliberately so: wasmtime bills every operator it executes, so the
    /// stamps would burn fuel and speed the game clock up. It exists here to let the
    /// mechanism be tested against the desktop's cheap bit-exact oracles before it is
    /// trusted in the browser, which is the engine that uses it.
    dirty_off: Option<u64>,
    signal: Arc<Mutex<Signal>>,
    /// Set by a host call that returned `Halt`, read by the fiber's async block.
    process_halt: bool,
    /// Set by a host call that returned `ThreadExit`.
    thread_exit: bool,
    /// Set by a host call that returned `Fatal`: the run must stop with this message
    /// (surfaced as `FiberEnd::Error` -> `RunReport::Error`). Read by the fiber's
    /// async block after the guest call unwinds.
    fatal: Option<String>,
    /// This thread's guest register/VFP globals, resolved once at instantiation.
    /// `None` only in the window before the instance exists. See [`GuestGlobals`].
    globals: Option<GuestGlobals>,
    /// This thread's SOFTWARE fuel counter (`abi::FUEL_EXPORT`), present only when the
    /// build emitted software fuel (`VITASLOP_FUEL`). Native does not need it - wasmtime
    /// meters the store - so it exists here purely to be COMPARED against wasmtime's own
    /// reading. See [`software_fuel_report`].
    sw_fuel: Option<wasmtime::Global>,
    /// The software counter's value at the last reading. It counts DOWN and reloads to a
    /// full interval after each yield, so only differences accumulate to guest work -
    /// exactly the arithmetic the browser host does, deliberately duplicated so a bug in
    /// it shows up HERE, next to the ground truth.
    sw_last: i64,
    /// wasmtime's own cumulative reading at the last software-fuel sample, so the two
    /// counters are differenced over exactly the same intervals.
    sw_wasmtime_last: u64,
    /// The EMITTED work counter's yield interval in wasm operators, or 0 when this build
    /// carries no counter. Taken from the engine rather than from
    /// `transpiler::fuel_interval()`, which is a THREAD-LOCAL set at transpile time: what
    /// the module contains is a property of the module, and a runtime reading of it must
    /// not depend on which thread asks.
    fuel_interval: u32,
}

/// The wasm globals holding the guest register file, resolved once per thread.
///
/// Every host call marshals the whole ARM register file plus the VFP argument
/// registers across the boundary in both directions - 64 global accesses per call on
/// this ABI. Resolving each one by its export NAME (`format!("r{i}")` into
/// `Caller::get_export`) makes that 64 string allocations and 64 export-table lookups
/// per host call, on a path a title takes millions of times. The handles are constant
/// for the life of the instance, so they are resolved once here and the closure just
/// indexes them.
///
/// Diagnostic paths (the function tracer, register dumps) still resolve by name: they
/// run at most thousands of times and the name is what they are reporting.
struct GuestGlobals {
    regs: [wasmtime::Global; abi::REG_COUNT],
    vfp: [wasmtime::Global; VFP_ARG_COUNT],
}

impl GuestGlobals {
    /// Resolve every register and VFP-argument global of `instance`. The transpiler
    /// always emits the full register file and `s0..s31`, so a missing export is a
    /// broken module rather than a title that happens not to use the register - it
    /// panics here, at instantiation, instead of on the first host call.
    fn resolve<T>(store: &mut Store<T>, instance: &Instance) -> Self {
        let regs = std::array::from_fn(|i| {
            instance
                .get_global(&mut *store, &abi::reg_export(i))
                .expect("module exports registers")
        });
        let vfp = std::array::from_fn(|i| {
            instance
                .get_global(&mut *store, &abi::vfp_s_export(i as u8))
                .expect("module exports vfp registers")
        });
        GuestGlobals { regs, vfp }
    }
}

/// One suspendable guest thread on the wasmtime engine: an async fiber (the in-flight
/// `call_async`) that owns its store and suspends at each switch point. This is the
/// wasmtime implementation of the engine-agnostic [`GuestThread`]; the shared
/// [`Scheduler`] drives it.
pub struct WasmtimeThread {
    thid: i32,
    /// The in-flight guest call; owns its store, suspends at each switch point.
    future: Pin<Box<dyn Future<Output = FiberEnd> + Send>>,
    signal: Arc<Mutex<Signal>>,
    /// SceKernel priority (lower number = higher priority). The scheduler always
    /// runs the highest-priority runnable thread, matching the real kernel.
    priority: i32,
    /// This thread's preemption interval, so [`resume`](GuestThread::resume) can price a
    /// suspend that wasmtime took on its own (see [`Signal::fuel`]).
    quantum_fuel: u64,
    /// The EMITTED work counter's interval, or 0 on a build without one. Non-zero means
    /// [`Signal::arm`] is sampled at every switch point and the game clock is billed in
    /// guest instructions - see [`ThreadHandle::arm_retired`].
    fuel_interval: u32,
}

impl ThreadHandle for WasmtimeThread {
    fn thid(&self) -> i32 {
        self.thid
    }
    fn priority(&self) -> i32 {
        self.priority
    }

    fn fuel_used(&mut self) -> Option<u64> {
        Some(self.signal.lock().unwrap().fuel)
    }

    /// Guest ARM instructions this thread has retired, from the EMITTED per-block counter
    /// - the unit the game clock is billed in on both engines.
    ///
    /// # Why this needs native's preemption to be ours rather than wasmtime's
    /// The counter is a wasm GLOBAL, and reading a global needs the thread's `Store`,
    /// which only exists inside the fiber. So it is sampled from the host-call closure
    /// ([`note_suspend`]) and sees only suspends that pass through it. While native
    /// preempted with `fuel_async_yield_interval`, wasmtime's own periodic yield ran none
    /// of our code, so a thread that spun WITHOUT making a host call was never sampled at
    /// all - and that thread is ordinary, since a title's vblank spin reads the clock
    /// through an INLINED mirror load. Billing an unsampled counter charged it nothing,
    /// the clock stopped, and the wait it was spinning on could never complete. MEASURED
    /// at the time: a retail boot ran in 6 s and then fast-forwarded for over ten minutes
    /// without reaching frame 300.
    ///
    /// The retail path now emits the transpiler's own fuel check
    /// (`from_linked` -> `set_fuel_interval`), so a preemption IS an `env.import` call and
    /// every switch point passes through the closure - the mechanism the browser already
    /// used. A spin with no host call yields on its loop back edge instead, which is
    /// exactly the case that used to livelock.
    ///
    /// And the high half is CUMULATIVE (see `abi::WORK_GLOBAL`), never cleared, so even a
    /// stretch that somehow suspends without being sampled - wasmtime's backstop yield,
    /// see [`GuestThread::resume`] - only DEFERS the charge to the next sample. It cannot
    /// lose it. That is what makes a stopped clock unreachable by this route.
    ///
    /// `None` on a build with no counter (the raw-image path: the ARM corpus and unit
    /// tests, whose mock hosts have no clock), which selects the fuel-derived fallback in
    /// `on_guest_work`.
    fn arm_retired(&mut self) -> Option<u64> {
        (self.fuel_interval != 0).then(|| self.signal.lock().unwrap().arm)
    }
}

impl GuestThread for WasmtimeThread {
    fn resume(&mut self) -> ThreadStep {
        // Poll the fiber once: it runs until its next `.await` (a switch point) or
        // returns. A no-op waker is right because the scheduler, not a reactor,
        // decides when to poll again. Reset the reason first; the host-call closure
        // overwrites it only if it suspends.
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let before = {
            let mut s = self.signal.lock().unwrap();
            s.stop = Stop::Quantum;
            s.host_suspends
        };
        match self.future.as_mut().poll(&mut cx) {
            Poll::Ready(end) => ThreadStep::Finished(end),
            Poll::Pending => {
                let mut s = self.signal.lock().unwrap();
                if s.host_suspends == before {
                    // Nothing of ours ran, so this is wasmtime's own periodic yield and
                    // the thread burned exactly one interval getting here. Without this the
                    // only threads that ever cost game time would be the ones that make
                    // host calls, and a guest spin that reads an inlined mirror makes none.
                    s.fuel = s.fuel.saturating_add(self.quantum_fuel);
                    // A build with the emitted work check has wasmtime's periodic yield
                    // switched OFF (see `instantiate_thread_seq`), so there is nothing left
                    // that can suspend a fiber without running our code. Reaching here is
                    // therefore an ENGINE surprise rather than a preemption, and it is
                    // silent otherwise: the thread just resumes.
                    if self.fuel_interval != 0 {
                        note_unattributed_suspend();
                    }
                }
                ThreadStep::Suspended(s.stop)
            }
        }
    }
}

/// The wasmtime execution engine: the transpiled module, the one shared memory every
/// thread instance imports, the shared host, and the knobs to stand up a new thread.
/// This is the wasmtime implementation of the engine-agnostic [`GuestEngine`].
pub struct WasmtimeEngine<H: ImportDispatch + Send + 'static> {
    engine: Engine,
    module: Module,
    shared_mem: SharedMemory,
    host: Arc<Mutex<H>>,
    base: u32,
    quantum_fuel: u64,
    /// The EMITTED work counter's yield interval, in wasm operators, or 0 when this
    /// build carries no counter (see [`ThreadData::fuel_interval`]).
    fuel_interval: u32,
    /// Linear-memory offset of the "diagnostics armed" word, when this build was
    /// transpiled with `VITASLOP_ARM_AT_FRAME` (see
    /// [`vitaslop_transpiler::arm_at_frame`]). `None` in an ordinary build.
    arm_word_off: Option<u64>,
    /// Linear-memory offset of the host-mirror block, when this build inlined any read
    /// of it (see `vitaslop_transpiler::InlineOp::LoadMirror`). The scheduler refreshes
    /// it before every resume.
    mirror_off: Option<u64>,
    /// Linear-memory offset of the guest-store dirty block - see [`ThreadData`], which
    /// carries it to the host-call view.
    dirty_off: Option<u64>,
}

impl<H: ImportDispatch + Send + 'static> GuestEngine for WasmtimeEngine<H> {
    type Thread = WasmtimeThread;

    fn spawn(&mut self, reentry: &Reentry) -> Result<WasmtimeThread, ()> {
        self.instantiate_thread(
            reentry.thid,
            reentry.entry,
            reentry.arg_len,
            reentry.arg_ptr,
            reentry.r2,
            reentry.stack_top,
            reentry.priority,
        )
        .map_err(|_| ())
    }

    fn write_mem(&mut self, addr: u32, bytes: &[u8]) {
        let off = addr.wrapping_sub(self.base) as usize;
        if off + bytes.len() <= self.shared_mem.data().len() {
            write_shared(&self.shared_mem, off, bytes);
            // A scheduler-side write is a host write like any other - see
            // `SharedView::stamp_written`.
            SharedView::new(&self.shared_mem, self.dirty_off).stamp_written(off, bytes.len());
        }
    }

    fn read_mem(&self, addr: u32, out: &mut [u8]) -> bool {
        let off = addr.wrapping_sub(self.base) as usize;
        let data = self.shared_mem.data();
        if off.checked_add(out.len()).is_none_or(|end| end > data.len()) {
            return false;
        }
        for (i, b) in out.iter_mut().enumerate() {
            // SAFETY: the same condition `write_shared` relies on - the scheduler calls
            // this between fiber steps, with one fiber at a time and none running, so
            // there is no concurrent guest access to this shared memory.
            *b = unsafe { *data[off + i].get() };
        }
        true
    }

    /// Arm the frame-gated diagnostics the instant the run reaches the requested
    /// frame. One word in shared linear memory covers every guest thread at once,
    /// which is the whole reason the gate is not a wasm global.
    fn on_frame(&mut self, frames: u64) {
        // Stamp the frame first, unconditionally: every host-side diagnostic that
        // prints a line reads it, and a trace line with no frame on it cannot be
        // placed against a crash frame or a recipe cue at all.
        CURRENT_FRAME.store(frames, std::sync::atomic::Ordering::Relaxed);
        let (Some(off), Some(at)) = (self.arm_word_off, transpiler::arm_at_frame()) else {
            return;
        };
        if frames != at {
            return;
        }
        write_shared(&self.shared_mem, off as usize, &1u32.to_le_bytes());
        DIAG_ARMED.store(true, std::sync::atomic::Ordering::Relaxed);
        eprintln!("[diag] armed at frame {frames} (VITASLOP_ARM_AT_FRAME)");
    }

    fn mirror_base(&self) -> Option<u32> {
        // The block sits above the guest region, so its guest address is the rebase
        // origin plus the offset - the same convention `write_mem` undoes.
        self.mirror_off.map(|off| self.base.wrapping_add(off as u32))
    }
}

/// Set once the run reaches `VITASLOP_ARM_AT_FRAME`, for the HOST-side diagnostics
/// (the qemu-diff snapshot and register trace) that live in this file rather than in
/// emitted code. True from the start when no frame gate was requested, so an
/// ungated run behaves exactly as before.
static DIAG_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The frame the run has reached, stamped by [`WasmtimeEngine::on_frame`]. Read by the
/// host-side diagnostics so every line they print carries a frame number: a block trace
/// that says only "this happened" cannot be lined up with a crash frame, a recipe cue or
/// another instrument's output, which is most of what such a trace is for.
static CURRENT_FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The frame the run has reached (0 before the first flip).
pub fn current_frame() -> u64 {
    CURRENT_FRAME.load(std::sync::atomic::Ordering::Relaxed)
}

/// Are the frame-gated diagnostics live yet?
fn diag_armed() -> bool {
    transpiler::arm_at_frame().is_none()
        || DIAG_ARMED.load(std::sync::atomic::Ordering::Relaxed)
}

/// A preemptive multi-thread guest run. A thin wasmtime front for the shared
/// [`Scheduler`] policy: it stands up the wasmtime [`WasmtimeEngine`] plus the main
/// thread, then hands both to the policy, which owns the thread table and the
/// scheduling loop. All the discipline (priority, deadlock/timed-wait, frame
/// counting) is in `vitaslop_runtime::sched`, shared with the browser scheduler.
pub struct ThreadedScheduler<H: ImportDispatch + Send + 'static> {
    inner: Scheduler<WasmtimeEngine<H>, H>,
}

impl<H: ImportDispatch + Send + 'static> ThreadedScheduler<H> {
    /// Transpile `code` (loaded at `base`) with a shared imported memory, seed the
    /// image into that memory, and stand up the main thread ready to run from
    /// `entry`. `host` is the shared host every thread dispatches its NID calls to
    /// (its [`ImportDispatch`] provides the spawn/wake/exit signals); `externs`
    /// wires the guest's import stubs to dense host indices.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        code: &[u8],
        base: u32,
        thumb: bool,
        entries: &[u32],
        externs: &[transpiler::Extern],
        mem_bytes: u32,
        host: H,
    ) -> Result<ThreadedScheduler<H>, RunError> {
        Self::with_quantum(code, base, thumb, entries, externs, mem_bytes, host, DEFAULT_QUANTUM_FUEL)
    }

    /// Like [`new`](Self::new) with an explicit fuel quantum (the preemption
    /// granularity, and the knob tests use to force fine interleaving).
    ///
    /// `entries[0]` is the main thread's entry. Any further entries are extra
    /// functions to transpile up front; real Vita modules leave these out and let
    /// code-pointer discovery find thread entries, but a caller that already knows
    /// the entry addresses (a test, or a module with an export table) can name them.
    #[allow(clippy::too_many_arguments)]
    pub fn with_quantum(
        code: &[u8],
        base: u32,
        thumb: bool,
        entries: &[u32],
        externs: &[transpiler::Extern],
        mem_bytes: u32,
        host: H,
        quantum_fuel: u64,
    ) -> Result<ThreadedScheduler<H>, RunError> {
        let entry = entries[0] & !1;
        let entries: Vec<u32> = entries.iter().map(|e| e & !1).collect();
        let artifact = transpiler::transpile(&transpiler::Program {
            code,
            base,
            thumb,
            entries: &entries,
            arm_entries: &[],
            externs,
            redirects: &[],
            // Raw-image entry point (the ARM corpus and tests): no NID import table,
            // so nothing here is known to be inlinable. The retail path goes through
            // `from_linked`, which passes the linker's list.
            inline_imports: &[],
            noreturn_svc: &[],
            mem_bytes,
            discover_code_pointers: true,
            import_memory: true,
        })?;
        wasmparser::validate(&artifact.wasm)
            .map_err(|e| RunError::Wasm(format!("invalid module: {e}")))?;

        let mut cfg = Config::new();
        cfg.wasm_threads(true);
        cfg.shared_memory(true);
        cfg.consume_fuel(true);
        let engine = Engine::new(&cfg).map_err(|e| RunError::Wasm(e.to_string()))?;
        let module = Module::from_binary(&engine, &artifact.wasm)?;

        // One shared memory, sized exactly as the transpiler declared (the guest
        // region plus the appended indirect-dispatch address table), imported into
        // every thread instance. Seed the image into it once.
        let pages = artifact.mem_pages;
        let mem_ty = wasmtime::MemoryType::shared(pages, pages);
        let shared_mem =
            SharedMemory::new(&engine, mem_ty).map_err(|e| RunError::Wasm(e.to_string()))?;
        write_shared(&shared_mem, 0, code);

        let host = Arc::new(Mutex::new(host));
        let engine = WasmtimeEngine {
            engine,
            module,
            shared_mem,
            host: host.clone(),
            base,
            quantum_fuel,
            // The raw-image path (the ARM corpus, unit tests) keeps wasmtime's own
            // periodic preemption: its hosts are mock `ImportDispatch`es with no
            // scheduler and no game clock, so there is nothing for a work counter to
            // bill. See `from_linked`, which is the retail path.
            fuel_interval: 0,
            arm_word_off: artifact.arm_word_off,
            mirror_off: artifact.mirror_off,
            dirty_off: artifact.dirty_off,
        };

        // The main thread: sp near the top of the region (with startup headroom), no
        // entry args, its thid is whatever the host reports for the main thread (0 by
        // convention here; the host maps it as it likes).
        let main = engine.instantiate_thread(
            0,
            entry & !1,
            0,
            0,
            0,
            main_stack_top(base, mem_bytes),
            vitaslop_runtime::host::DEFAULT_THREAD_PRIORITY,
        )?;
        Ok(ThreadedScheduler { inner: Scheduler::new(engine, host, main) })
    }

    /// Stand up a preemptive run of a multi-module linked title
    /// ([`vitaslop_runtime::link::LinkedProgram`]) - the faithful counterpart to the
    /// single-thread [`Vm::from_linked`](crate::Vm::from_linked). Transpiles
    /// *leniently* (still-unlifted functions become trapping stubs) with an imported
    /// shared memory and the inter-module redirects, seeds the combined image, and
    /// makes the main thread run every module's `module_start` in load order (shared
    /// libraries first, the executable last) before returning control to the
    /// scheduler loop. Returns the scheduler and the addresses that became stubs (as
    /// `(guest addr, wasm function index)`).
    pub fn from_linked(
        linked: &vitaslop_runtime::link::LinkedProgram,
        host: H,
        quantum_fuel: u64,
    ) -> Result<(ThreadedScheduler<H>, Vec<(u32, u32)>), RunError> {
        // >>> THE RETAIL PATH EMITS THE WORK COUNTER, AND PREEMPTS ON IT. This is what
        // lets native bill its game clock in GUEST INSTRUCTIONS like the browser does,
        // rather than in wasm operators - see `WasmtimeThread::arm_retired` for the
        // livelock that kept it on operators until now, and `host::QUANTUM_ARM` for what
        // the unit buys. The interval is the same quantum wasmtime's periodic yield used,
        // so the preemption GRANULARITY is unchanged; what changes is that every
        // preemption is now a call through our own import, where the counter can be read.
        transpiler::set_fuel_interval(u32::try_from(quantum_fuel).unwrap_or(u32::MAX));
        let built = transpiler::transpile_lenient(&linked.shared_program());
        // Leave the thread as we found it: the emitted module carries its own interval
        // and every runtime reader takes it from `ThreadData`, so nothing after this
        // point should depend on a thread-local that another transpile could inherit.
        transpiler::set_fuel_interval(u32::MAX);
        let fuel_interval = u32::try_from(quantum_fuel).unwrap_or(u32::MAX);
        // >>> THE CODE EXPANSION FACTOR, reported unconditionally on the engine that takes
        // every calibration measurement. The game clock is charged per unit of fuel and a
        // unit of fuel is one executed wasm operator, so the emulated Vita's CPU speed is
        // `fuel rate / this number`. Codegen work moves it, and until it was printed
        // nothing in a run said so - the constant it calibrates
        // (`vitaslop_runtime::host::QUANTUM_CPU_US`) derives from a figure that was only
        // ever estimated in a comment.
        {
            let x = built.artifact.expansion;
            // >>> MODULE SIZE, which is a cost no operator count and no wall clock reports.
            // The engine this ships on is a phone browser, where the module has to be
            // fetched, parsed and compiled before a frame is drawn and where the project
            // has repeatedly found itself memory-bound. A codegen change that emits fewer
            // operators emits fewer BYTES too, and that is worth something on its own.
            tracing::info!(
                target: "vitaslop::perf",
                "module: {:.2} MB of wasm ({} bytes) for {} guest instructions - {:.1} bytes \
                 each",
                built.artifact.wasm.len() as f64 / (1024.0 * 1024.0),
                built.artifact.wasm.len(),
                x.arm_instructions,
                built.artifact.wasm.len() as f64 / x.arm_instructions.max(1) as f64,
            );
            tracing::info!(
                target: "vitaslop::perf",
                "code expansion: {:.2} wasm operators per guest instruction \
                 ({} instructions -> {} operators), of which {:.1}% ({}) are moves of the \
                 ARM registers and flags to and from the instance globals. The game clock \
                 is billed in guest INSTRUCTIONS where the emitted work counter exists \
                 (the browser), and that counter rides the fuel commit at no extra cost, \
                 so the expansion no longer divides the emulated CPU's speed there",
                x.per_instruction(),
                x.arm_instructions,
                x.emitted_ops,
                x.core_state_share(),
                x.core_state_ops,
            );
            // >>> AND THE SPLIT OF THAT BOOKKEEPING, because the total names no fix. Three
            // unrelated mechanisms share it: the work counter (commits + fuel checks), the
            // guest-store dirty map, and the promotion cache. The commits-per-guest-
            // instruction figure is the one that says whether the flush POLICY costs more
            // than the commit does.
            {
                let cache = x
                    .unbilled_ops
                    .saturating_sub(x.unbilled_work_ops)
                    .saturating_sub(x.unbilled_dirty_ops);
                let pct = |n: u64| 100.0 * n as f64 / x.unbilled_ops.max(1) as f64;
                tracing::info!(
                    target: "vitaslop::perf",
                    "bookkeeping split: work counter {} ops ({:.1}%) over {} commits \
                     ({:.2} commits per guest instruction), dirty map {} ops ({:.1}%), \
                     promotion cache {} ops ({:.1}%)",
                    x.unbilled_work_ops,
                    pct(x.unbilled_work_ops),
                    x.work_flushes,
                    x.work_flushes as f64 / x.arm_instructions.max(1) as f64,
                    x.unbilled_dirty_ops,
                    pct(x.unbilled_dirty_ops),
                    cache,
                    pct(cache),
                );
            }
            // >>> AND WHICH LOWERING SPENDS THEM. The average is a corpus statistic; a fix
            // has to land on a particular lowering, and the two are not the same question.
            // Ranked by TOTAL operators (what a change is worth) with the per-statement
            // cost alongside (how bad the lowering is), because those two rankings
            // disagree: a 40-operator lowering used twice is not the target a 6-operator
            // one used a million times is.
            {
                let mut rows: Vec<(&'static str, u64, u64)> =
                    vitaslop_transpiler::StmtKind::ALL
                        .iter()
                        .enumerate()
                        .map(|(i, k)| (k.label(), x.by_stmt[i].0, x.by_stmt[i].1))
                        .filter(|r| r.2 != 0)
                        .collect();
                rows.sort_by(|a, b| b.1.cmp(&a.1));
                let total: u64 = rows.iter().map(|r| r.1).sum();
                let line: Vec<String> = rows
                    .iter()
                    .map(|(label, ops, n)| {
                        format!(
                            "{label} {:.1}% ({ops} ops over {n}, {:.1} each)",
                            100.0 * *ops as f64 / total.max(1) as f64,
                            *ops as f64 / *n as f64,
                        )
                    })
                    .collect();
                tracing::info!(
                    target: "vitaslop::perf",
                    "operators by lowering: {}",
                    line.join("; "),
                );
            }
            // >>> AND INSIDE THE LARGEST LINE, the shape that dominates it. `flags-add` is
            // already gated on flag liveness, so its cost is decided by WHICH flags stay
            // live - and C alone is nine of its operators.
            {
                let total: u64 = x.flags_add_live.iter().sum();
                let mut rows: Vec<(String, u64)> = (0..16usize)
                    .filter(|m| x.flags_add_live[*m] != 0)
                    .map(|m| {
                        let name: String = [(0, 'N'), (1, 'Z'), (2, 'C'), (3, 'V')]
                            .iter()
                            .filter(|(b, _)| m & (1 << b) != 0)
                            .map(|(_, c)| *c)
                            .collect();
                        (if name.is_empty() { "-".into() } else { name }, x.flags_add_live[m])
                    })
                    .collect();
                rows.sort_by(|a, b| b.1.cmp(&a.1));
                let line: Vec<String> = rows
                    .iter()
                    .map(|(n, c)| format!("{n} {c} ({:.1}%)", 100.0 * *c as f64 / total.max(1) as f64))
                    .collect();
                tracing::info!(
                    target: "vitaslop::perf",
                    "flags-add by live mask: {} - and {} of {total} ({:.1}%) have a \
                     compile-time carry-in",
                    line.join(", "),
                    x.flags_add_const_cin,
                    100.0 * x.flags_add_const_cin as f64 / total.max(1) as f64,
                );
            }
            // >>> THE CONTROL-FLOW SHAPE. Not an operator count and not priced by one: a
            // dispatch re-entry is an INDIRECT BRANCH through the function's `br_table`,
            // which costs a prediction rather than an operator.
            tracing::info!(
                target: "vitaslop::perf",
                "control flow: {} blocks ({:.1} guest instructions each), {:.1}% end in a \
                 FALLTHROUGH (straight-line, free), {} dispatch re-entries - one indirect \
                 branch per {:.1} guest instructions",
                x.blocks,
                x.arm_instructions as f64 / x.blocks.max(1) as f64,
                100.0 * x.fallthrough_blocks as f64 / x.blocks.max(1) as f64,
                x.dispatch_reentries,
                x.arm_instructions as f64 / x.dispatch_reentries.max(1) as f64,
            );
            // What the counter does NOT bill, so the standing cross-check against
            // wasmtime's own metering (`software_fuel_report`) has a predicted value
            // instead of an unexplained one.
            tracing::info!(
                target: "vitaslop::perf",
                "work counter: {} operators of its own bookkeeping are NOT billed \
                 ({:.1}% of what the module executes), so a real engine's metering of the \
                 same code reads that much HIGHER than ours",
                x.unbilled_ops,
                x.unbilled_share(),
            );
            // >>> AND WHAT PROMOTING THOSE MOVES INTO WASM LOCALS WOULD ACTUALLY BE
            // WORTH. The share above is a CEILING and reads far higher than the truth:
            // most core-state accesses sit in straight-line runs too short to pay a
            // promotion back. See `transpiler::promote` for the policy and for the
            // measured price of a converted access.
            let p = x.promotion;
            tracing::info!(
                target: "vitaslop::perf",
                "register promotion model: {} straight-line runs, {} accesses would \
                 become LOCAL ({:.1}% of all operators), {} would stay on their globals, \
                 costing {} added operators ({:+.1}%); longest run {} accesses",
                p.runs,
                p.converted,
                p.converted_share(x.emitted_ops),
                p.left,
                p.overhead,
                p.overhead_share(x.emitted_ops),
                p.longest_run,
            );
            // WHAT ENDED THE RUNS the unpromotable accesses were in. A call is not
            // negotiable - the callee reaches the same globals - but a scope boundary is
            // an `if`/`end` from ARM predication, which a smarter policy could carry a
            // cache across. This split is what says whether writing that policy pays.
            tracing::info!(
                target: "vitaslop::perf",
                "register promotion, accesses left behind by cause: {} call, {} scope \
                 (if/else/end), {} branch, {} return/trap",
                p.lost_to[transpiler::promote::Ender::Call as usize],
                p.lost_to[transpiler::promote::Ender::Scope as usize],
                p.lost_to[transpiler::promote::Ender::Branch as usize],
                p.lost_to[transpiler::promote::Ender::Exit as usize],
            );
        }
        // Record the wasm-index -> guest-address table before anything can trap, so a
        // backtrace names guest code instead of listing module indices.
        record_function_addresses(built.artifact.funcs.iter().map(|f| f.addr).collect());
        wasmparser::validate(&built.artifact.wasm)
            .map_err(|e| RunError::Wasm(format!("invalid module: {e}")))?;

        let mut cfg = Config::new();
        cfg.wasm_threads(true);
        cfg.shared_memory(true);
        cfg.consume_fuel(true);
        let engine = Engine::new(&cfg).map_err(|e| RunError::Wasm(e.to_string()))?;
        let module = Module::from_binary(&engine, &built.artifact.wasm)?;

        let pages = built.artifact.mem_pages;
        let mem_ty = wasmtime::MemoryType::shared(pages, pages);
        let shared_mem =
            SharedMemory::new(&engine, mem_ty).map_err(|e| RunError::Wasm(e.to_string()))?;
        write_shared(&shared_mem, 0, &linked.image);

        // The main module's `module_start(SceSize args, void *argp)` is handed the launch
        // arguments the way a Vita loader passes them: `argp` points to a run of
        // NUL-terminated strings (argv), `args` is their total byte length. The executable's
        // `main` parses this argv and derives its data paths from `argv[0]` (the self path);
        // handed an empty block it walks off an uninitialized path and faults deep in startup.
        // Seed a one-argument block (the app's own eboot path) in the stack headroom - the
        // region the crt already expects the kernel to have populated.
        let arg_block: &[u8] = b"app0:/eboot.bin\0";
        // High in the 1 MiB headroom (well above the initial SP and its crt scratch, below the
        // region end), so nothing the stack does can clobber it.
        let arg_ptr = linked.base.wrapping_add(linked.mem_bytes).wrapping_sub(0x1000);
        write_shared(&shared_mem, (arg_ptr - linked.base) as usize, arg_block);
        let arg_len = arg_block.len() as u32;

        let host = Arc::new(Mutex::new(host));
        let engine = WasmtimeEngine {
            engine,
            module,
            shared_mem,
            host: host.clone(),
            base: linked.base,
            quantum_fuel,
            fuel_interval,
            arm_word_off: built.artifact.arm_word_off,
            mirror_off: built.artifact.mirror_off,
            dirty_off: built.artifact.dirty_off,
        };

        // The main thread runs every module_start in load order, then (as the last
        // entry) the eboot's - which is where a render loop lives.
        let sp = main_stack_top(linked.base, linked.mem_bytes);
        let main = engine.instantiate_thread_seq(
            0,
            linked.module_inits.clone(),
            arg_len,
            arg_ptr,
            0,
            sp,
            vitaslop_runtime::host::DEFAULT_THREAD_PRIORITY,
        )?;

        let stubs = built
            .stubbed
            .iter()
            .copied()
            .zip(built.stub_wasm_indices.iter().copied())
            .collect();
        Ok((ThreadedScheduler { inner: Scheduler::new(engine, host, main) }, stubs))
    }

    /// Borrow the shared host (e.g. to read captured output after the run).
    pub fn host(&self) -> std::sync::MutexGuard<'_, H> {
        self.inner.engine().host.lock().unwrap()
    }

    /// Run `f` against the host WITH guest memory in hand - the accessor for anything that
    /// reaches guest-resident state from outside a host call. See
    /// [`vitaslop_runtime::sched::SchedCore::with_host_words`].
    pub fn with_host_words<R>(
        &mut self,
        f: impl FnOnce(&mut H, &mut dyn vitaslop_runtime::host::GuestWords) -> R,
    ) -> R {
        self.inner.core_mut().with_host_words(f)
    }

    /// Read `len` bytes of guest memory at guest address `addr` (diagnostic; the
    /// shared image outlives any trap, so a probe can inspect object state after a
    /// fault). Returns an empty vec if the range is out of bounds.
    pub fn read_guest(&self, addr: u32, len: usize) -> Vec<u8> {
        self.inner.engine().read_guest(addr, len)
    }

    /// Diagnostic: overwrite guest memory at `addr` (used by the probe's POKE knob).
    pub fn write_guest(&self, addr: u32, bytes: &[u8]) {
        self.inner.engine().write_guest(addr, bytes)
    }

    /// Bulk-read `buf.len()` bytes of guest memory at `addr` into `buf`. Returns
    /// false (leaving `buf` untouched) if the range is out of bounds. This is the
    /// whole-region read a memory scanner does repeatedly, so it is one block copy
    /// rather than [`read_guest`](Self::read_guest)'s allocate-and-loop.
    pub fn read_guest_into(&self, addr: u32, buf: &mut [u8]) -> bool {
        self.inner.engine().read_guest_into(addr, buf)
    }

    /// The guest address range backed by linear memory, as `(base, len)`. A memory
    /// scanner needs it to know what there is to search.
    pub fn guest_region(&self) -> (u32, usize) {
        let e = self.inner.engine();
        (e.base, e.shared_mem.data().len())
    }

    /// Display frame boundaries (flips) observed so far. A live windowed front-end
    /// steps one frame per redraw via `run_frames(frames() + 1, ..)`.
    pub fn frames(&self) -> u64 {
        self.inner.frames()
    }

    /// Thread resumes so far - the scheduler's own activity, for a profiler that has
    /// to separate the guest's work from the cost of switching between guest threads.
    pub fn rounds_total(&self) -> u64 {
        self.inner.rounds_total()
    }

    /// `(live, finished)` guest threads - see
    /// [`vitaslop_runtime::sched::SchedCore::thread_census`].
    pub fn thread_census(&self) -> (usize, usize) {
        self.inner.thread_census()
    }

    /// Who actually got the CPU - see
    /// [`vitaslop_runtime::sched::SchedCore::cpu_share_report`].
    pub fn cpu_share_report(&self) -> String {
        self.inner.cpu_share_report()
    }

    /// How much of the device's parallelism the run used - see
    /// [`vitaslop_runtime::sched::SchedCore::runnable_report`].
    pub fn runnable_report(&self) -> String {
        self.inner.runnable_report(vitaslop_runtime::host::guest_cores())
    }

    /// `(total fuel burned, samples, largest single burn)` - see
    /// [`vitaslop_runtime::sched::SchedCore::fuel_report`].
    pub fn fuel_report(&self) -> (u64, u64, u64) {
        self.inner.fuel_report()
    }

    /// Cumulative retired guest ARM instructions - see
    /// [`vitaslop_runtime::sched::SchedCore::arm_report`].
    pub fn arm_report(&self) -> u64 {
        self.inner.arm_report()
    }

    /// Run cooperatively until the process halts, every thread finishes, or the run
    /// deadlocks / errors. Delegates to the shared scheduler policy.
    pub fn run(&mut self) -> RunReport {
        self.inner.run()
    }

    /// Like [`run`](Self::run) but stop after `max_frames` frame boundaries; `max_rounds`
    /// caps thread resumes so a busy-waiting guest cannot run unbounded.
    pub fn run_frames(&mut self, max_frames: u64, max_rounds: u64) -> RunReport {
        self.inner.run_frames(max_frames, max_rounds)
    }
}

impl<H: ImportDispatch + Send + 'static> WasmtimeEngine<H> {
    /// Read `len` bytes of guest memory at guest address `addr`. Returns an empty vec
    /// if the range is out of bounds.
    fn read_guest(&self, addr: u32, len: usize) -> Vec<u8> {
        let off = addr.wrapping_sub(self.base) as usize;
        let data = self.shared_mem.data();
        if off + len > data.len() {
            return Vec::new();
        }
        // SAFETY: bounds checked above; no fiber runs concurrently with the probe.
        let mut out = vec![0u8; len];
        for (i, b) in out.iter_mut().enumerate() {
            unsafe {
                *b = *data[off + i].get();
            }
        }
        out
    }

    /// Bulk-read guest memory into `buf`; false if the range is out of bounds.
    fn read_guest_into(&self, addr: u32, buf: &mut [u8]) -> bool {
        let off = addr.wrapping_sub(self.base) as usize;
        let data = self.shared_mem.data();
        let Some(end) = off.checked_add(buf.len()) else { return false };
        if end > data.len() {
            return false;
        }
        // SAFETY: bounds checked above, and the scheduler holds the baton - no fiber
        // runs concurrently with a host-side read. `UnsafeCell<u8>` has the same
        // layout as `u8`, so the region is one contiguous byte block.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data[off].get() as *const u8,
                buf.as_mut_ptr(),
                buf.len(),
            );
        }
        true
    }

    /// Diagnostic write into guest memory. No-op if out of bounds.
    fn write_guest(&self, addr: u32, bytes: &[u8]) {
        let off = addr.wrapping_sub(self.base) as usize;
        let data = self.shared_mem.data();
        if off + bytes.len() > data.len() {
            return;
        }
        // SAFETY: bounds checked; no fiber runs concurrently with the probe.
        for (i, b) in bytes.iter().enumerate() {
            unsafe {
                *data[off + i].get() = *b;
            }
        }
    }

    /// Build one thread from a single entry (a spawned worker).
    fn instantiate_thread(
        &self,
        thid: i32,
        entry: u32,
        r0: u32,
        r1: u32,
        r2: u32,
        sp: u32,
        priority: i32,
    ) -> Result<WasmtimeThread, RunError> {
        self.instantiate_thread_seq(thid, vec![entry], r0, r1, r2, sp, priority)
    }

    /// Build one thread that runs `entries` in sequence on a single fiber, resetting
    /// the stack pointer before each so they nest cleanly. This is how the main
    /// thread runs a linked title's shared-library constructors (in load order)
    /// before the executable's own entry: each `module_start` runs to completion,
    /// and the last entry (the eboot's) is the one that may spin a render loop. Only
    /// the first entry receives `(r0, r1)`; the rest are called with no arguments,
    /// matching how a loader invokes `module_start(0, NULL)`.
    fn instantiate_thread_seq(
        &self,
        thid: i32,
        entries: Vec<u32>,
        r0: u32,
        r1: u32,
        r2: u32,
        sp: u32,
        priority: i32,
    ) -> Result<WasmtimeThread, RunError> {
        let signal =
            Arc::new(Mutex::new(Signal { stop: Stop::Quantum, fuel: 0, arm: 0, host_suspends: 0 }));
        let data = ThreadData {
            host: self.host.clone(),
            thid,
            shared_mem: self.shared_mem.clone(),
            base: self.base,
            dirty_off: self.dirty_off,
            signal: signal.clone(),
            process_halt: false,
            thread_exit: false,
            fatal: None,
            globals: None,
            sw_fuel: None,
            sw_last: 0,
            sw_wasmtime_last: 0,
            fuel_interval: self.fuel_interval,
        };
        let mut store = Store::new(&self.engine, data);
        store.set_fuel(u64::MAX).map_err(|e| RunError::Wasm(e.to_string()))?;
        // >>> ON A BUILD WITH THE EMITTED WORK CHECK, THAT CHECK IS THE ONLY THING THAT
        // PREEMPTS. wasmtime's periodic yield is switched OFF rather than kept as a
        // backstop, and the reason is that it is not one: its interval is free-running on
        // ITS own fuel and our yields do not reset it, so it fires on a fixed period no
        // matter how often the emitted check has already preempted. MEASURED with it left
        // on at eight times the interval: it fired anyway, and because that path credits a
        // blind quantum to a reading that is otherwise absolute, the per-suspend burn read
        // up to 9.1 M against a 5 M interval.
        //
        // Switching it off also makes native preempt at EXACTLY the points the browser
        // does - the same emitted check in the same module - which is what lets a desktop
        // run stand in for a browser one. The runaway risk it used to cover is the risk
        // the browser has always carried: a guest loop with no back-edge check would spin
        // there too, and the product does not.
        //
        // Fuel consumption stays ON either way: `note_suspend` reads it as the second
        // opinion the software counter is checked against.
        let yield_at = match self.fuel_interval {
            0 => Some(self.quantum_fuel),
            _ => None,
        };
        store.fuel_async_yield_interval(yield_at).map_err(|e| RunError::Wasm(e.to_string()))?;

        let mut linker = Linker::new(&self.engine);
        bind_svc(&mut linker)?;
        bind_import(&mut linker)?;
        bind_dispatch_miss(&mut linker)?;
        linker
            .define(&store, abi::IMPORT_MODULE, abi::MEMORY_EXPORT, self.shared_mem.clone())
            .map_err(|e| RunError::Wasm(e.to_string()))?;

        // No start section, so instantiation completes without suspending.
        let instance = pollster::block_on(linker.instantiate_async(&mut store, &self.module))?;
        // Resolve the register-file globals once, now, so no host call ever looks one
        // up by name (see `GuestGlobals`).
        let globals = GuestGlobals::resolve(&mut store, &instance);
        store.data_mut().globals = Some(globals);
        // Absent unless this build emitted software fuel, which native never needs and
        // only a comparison run turns on.
        let sw_fuel = instance.get_global(&mut store, abi::FUEL_EXPORT);
        store.data_mut().sw_fuel = sw_fuel;

        // This thread's thread-local-storage: a private block whose base becomes the
        // thread pointer (TPIDRURO). Copy the template's initialized `.tdata` head into
        // it (its `.tbss` tail is already zero); the guest reaches `__thread` variables
        // at `tp + offset`. No fiber is running here, so the shared-memory copy is safe.
        let (tp, tls_src, tls_len) = self.host.lock().unwrap().thread_tls_base(thid);
        if tls_len != 0 && tp != 0 {
            let src_off = tls_src.wrapping_sub(self.base) as usize;
            let dst_off = tp.wrapping_sub(self.base) as usize;
            copy_shared(&self.shared_mem, src_off, dst_off, tls_len as usize);
        }

        let future = Box::pin(async move {
            // The thread pointer is a per-thread constant: set it once before running
            // any entry (a `MRC p15,0,Rt,c13,c0,3` reads it via the `tp` global).
            set_tp_store(&mut store, &instance, tp);
            let mut last_r0 = 0u32;
            let last = entries.len().saturating_sub(1);
            for (i, &entry) in entries.iter().enumerate() {
                // Each entry starts with a fresh stack. The executable's `module_start` (the
                // LAST entry - shared libraries are constructed first and take no arguments)
                // is the one a loader hands the launch argv to; give the args to it.
                let carries_args = i == last;
                set_reg_store(&mut store, &instance, abi::SP, sp);
                set_reg_store(&mut store, &instance, 0, if carries_args { r0 } else { 0 });
                set_reg_store(&mut store, &instance, 1, if carries_args { r1 } else { 0 });
                set_reg_store(&mut store, &instance, 2, if carries_args { r2 } else { 0 });
                let func = match instance
                    .get_typed_func::<(), ()>(&mut store, &abi::func_export(entry))
                {
                    Ok(f) => f,
                    // The entry was not a transpiled function; skip it.
                    Err(_) => continue,
                };
                let call_res = func.call_async(&mut store, ()).await;
                last_r0 = get_reg_store(&mut store, &instance, 0);
                if let Err(e) = call_res {
                    let d = store.data_mut();
                    if let Some(msg) = d.fatal.take() {
                        return FiberEnd::Error(msg);
                    }
                    if d.process_halt {
                        return FiberEnd::ProcessHalt(last_r0);
                    }
                    if d.thread_exit {
                        // A module_start can end its (initial) thread with
                        // sceKernelExitThread instead of returning. When that entry
                        // is not the last, it just ends that init - clear the flag
                        // and run the next one, so an early library constructor
                        // exiting cannot tear down the whole main thread before the
                        // executable's own entry has run.
                        if i != last {
                            d.thread_exit = false;
                            continue;
                        }
                        return FiberEnd::ThreadExit(last_r0);
                    }
                    let regs = reg_dump(&mut store, &instance);
                    return FiberEnd::Error(format!("{}\n{}", trap_detail(&e), regs));
                }
            }
            FiberEnd::Returned(last_r0)
        });

        Ok(WasmtimeThread {
            thid,
            future,
            signal,
            priority,
            quantum_fuel: self.quantum_fuel,
            fuel_interval: self.fuel_interval,
        })
    }
}

/// Bind `env.svc`. Unused by the Vita NID path (which never traps a real `svc`), but
/// the diagnostic function tracer (`VITASLOP_TRACE_FUNCS`) routes guest-function-entry
/// announcements through it: a selector with the top bit set is a traced function's own
/// address (guest addresses are always >= 0x81000000; real `svc` immediates are 24-bit),
/// and we log the entry with its incoming argument registers. A real (small) selector is
/// a genuine syscall and stays a no-op on this path.
fn bind_svc<H: ImportDispatch + Send + 'static>(
    linker: &mut Linker<ThreadData<H>>,
) -> Result<(), RunError> {
    linker
        .func_wrap_async(
            abi::IMPORT_MODULE,
            abi::SVC_NAME,
            |mut caller: Caller<'_, ThreadData<H>>, (selector,): (i32,)| {
                Box::new(async move {
                    use std::sync::atomic::Ordering::Relaxed;
                    let sel = selector as u32;
                    if sel & 0x8000_0000 != 0 {
                        // Block-visit histogram: count entries per block PC (and record the
                        // exact entry sequence for the first few thousand) instead of printing.
                        // Reveals a hot loop's structure without flooding stderr or perturbing
                        // the schedule the way a per-block eprintln does.
                        if block_hist_enabled() {
                            block_hist_record(sel);
                        }
                        // The verbose per-block eprintln is the human trace; suppress it when
                        // a machine register trace is being captured (the file is the record,
                        // and a wide qdiff hook range would otherwise flood stderr), or when
                        // only the histogram is wanted.
                        if qdiff_regtrace().is_none()
                            && !block_hist_enabled()
                            && trace_frame_window_open(current_frame())
                        {
                            let thid = caller.data().thid;
                            let r: [u32; 13] = std::array::from_fn(|i| get_reg(&mut caller, i));
                            let frame = current_frame();
                            // The watched words (VITASLOP_REGTRACE_WATCH) belong on this line
                            // too, not only in the machine regtrace file. A control-flow trace
                            // says which branch was taken but never WHICH STATE decided it, and
                            // for a guard that another thread or a host call writes, the
                            // deciding word is not in any register the block reads.
                            let watched = watch_words(&mut caller);
                            eprintln!(
                                "[trace] frame={frame} t{thid} f_{sel:x}  r0={:#010x} r1={:#010x} r2={:#010x} \
                                 r3={:#010x} r4={:#010x} r5={:#010x} r6={:#010x} r7={:#010x} \
                                 r8={:#010x} r9={:#010x} r10={:#010x} r11={:#010x} r12={:#010x} lr={:#010x}{watched}",
                                r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
                                r[8], r[9], r[10], r[11], r[12], get_reg(&mut caller, 14),
                            );
                        }
                        // qemu-diff capture (opt-in; see the qdiff_* helpers below). The
                        // snapshot fires on the (skip+1)-th entry to its block, so a specific
                        // invocation of a repeatedly-called function can be targeted.
                        if let Some((snap_pc, path)) = qdiff_snapshot().as_ref().filter(|_| diag_armed()) {
                            if *snap_pc == sel && !QDIFF_SNAP_FIRED.load(Relaxed) {
                                let seen = QDIFF_SNAP_SEEN.fetch_add(1, Relaxed);
                                if seen >= qdiff_snapshot_skip() {
                                    QDIFF_SNAP_FIRED.store(true, Relaxed);
                                    qdiff_dump_snapshot(&mut caller, sel, path);
                                }
                            }
                        }
                        if let Some((lo, hi, path)) = qdiff_regtrace() {
                            let armed = diag_armed()
                                && (qdiff_snapshot().is_none() || QDIFF_SNAP_FIRED.load(Relaxed));
                            if armed && sel >= *lo && sel <= *hi {
                                qdiff_log_regtrace(&mut caller, sel, path);
                            }
                        }
                    }
                    Ok(())
                })
            },
        )
        .map_err(|e| RunError::Wasm(e.to_string()))?;
    Ok(())
}

/// `VITASLOP_TRACE_FRAMES=<from>-<to>` (decimal display frames, inclusive) - print the
/// per-block `[trace]` lines only inside that frame window. Unset traces the whole run.
///
/// `VITASLOP_TRACE_BLOCKS` selects WHICH blocks to trace but not WHEN, and the two are
/// not interchangeable: a function that a title calls every frame emits its whole boot
/// history before reaching the frame under investigation. On this codebase's own titles
/// that is a quarter of a million lines before the first flip, which buries the twenty
/// lines the question is about and slows the run enough to move the schedule being
/// measured. The window is applied HOST-side, so it needs no re-transpile and can be
/// changed between runs of the same build.
///
/// This is the block tracer's counterpart to [`transpiler::arm_at_frame`], which already
/// gates the qdiff snapshot and register trace the same way.
fn trace_frame_window() -> &'static Option<(u64, u64)> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<(u64, u64)>> = OnceLock::new();
    CELL.get_or_init(|| {
        let s = std::env::var("VITASLOP_TRACE_FRAMES").ok()?;
        let (from, to) = s.split_once('-').unwrap_or_else(|| {
            panic!("VITASLOP_TRACE_FRAMES: {s:?} is not <from>-<to> (decimal frames)")
        });
        match (from.trim().parse::<u64>(), to.trim().parse::<u64>()) {
            (Ok(from), Ok(to)) if from <= to => Some((from, to)),
            _ => panic!("VITASLOP_TRACE_FRAMES: {s:?} is not a valid frame window <from>-<to>"),
        }
    })
}

/// The `VITASLOP_REGTRACE_WATCH` words, formatted as ` mADDR=VALUE` fields ready to append
/// to a trace line. Empty when the knob is unset, so an unwatched trace is unchanged.
///
/// Shared by the human block trace and the machine register trace so the two never disagree
/// about what a word held at the same block entry.
fn watch_words<H: ImportDispatch + Send + 'static>(caller: &mut Caller<'_, ThreadData<H>>) -> String {
    let watch = qdiff_regtrace_watch();
    if watch.is_empty() {
        return String::new();
    }
    let base = caller.data().base;
    let shared = caller.data().shared_mem.clone();
    let data = shared.data();
    // SAFETY: as in `qdiff_dump_snapshot` - `UnsafeCell<u8>` is repr(transparent) over `u8`,
    // and no other fiber runs while this svc handler executes.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
    let mut out = String::new();
    for &addr in watch {
        let off = addr.wrapping_sub(base) as usize;
        match bytes.get(off..off + 4) {
            Some(w) => out.push_str(&format!(
                " m{addr:08x}={:08x}",
                u32::from_le_bytes([w[0], w[1], w[2], w[3]])
            )),
            None => out.push_str(&format!(" m{addr:08x}=oob")),
        }
    }
    out
}

/// Is `frame` inside the [`trace_frame_window`]? True everywhere when no window is set.
fn trace_frame_window_open(frame: u64) -> bool {
    match trace_frame_window() {
        Some((from, to)) => frame >= *from && frame <= *to,
        None => true,
    }
}

// --- qemu-diff capture: full-state snapshot + per-block register trace ------------
//
// A REUSABLE differential-oracle capture (not game-specific): piggyback the per-block
// `svc` hook (emitted by VITASLOP_TRACE_BLOCKS) to (a) dump the whole guest state once
// at a chosen block PC and (b) log the register+flag file at every block entry from that
// point on. An external reference ARMv7 CPU (qemu-arm, driven by the qdiff host tool)
// then replays the SAME instruction stream from the SAME state, and a per-block diff
// pinpoints the first block whose entry state departs from the reference - i.e. the
// mis-lifted op lives in the preceding block. To use, transpile with VITASLOP_TRACE_BLOCKS
// covering the window (so the hooks are emitted) plus these two knobs.

/// `VITASLOP_SNAPSHOT=<hexpc>:<path>` - dump full state on first entry to block `hexpc`.
fn qdiff_snapshot() -> &'static Option<(u32, String)> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<(u32, String)>> = OnceLock::new();
    CELL.get_or_init(|| {
        let s = std::env::var("VITASLOP_SNAPSHOT").ok()?;
        // First ':' splits the (colon-free) hex PC from the path (which may be a
        // Windows path containing a drive-letter colon).
        let (pc, path) = s.split_once(':')?;
        let pc = u32::from_str_radix(pc.trim().trim_start_matches("0x"), 16).ok()?;
        Some((pc, path.trim().to_string()))
    })
}

/// `VITASLOP_REGTRACE=<lo>-<hi>:<path>` - append the reg+flag file per block entry in
/// `[lo,hi]` (starting once the snapshot has fired, or immediately if none is configured).
fn qdiff_regtrace() -> &'static Option<(u32, u32, String)> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<(u32, u32, String)>> = OnceLock::new();
    CELL.get_or_init(|| {
        let s = std::env::var("VITASLOP_REGTRACE").ok()?;
        // The range "lo-hi" is colon-free, so the first ':' delimits range from path.
        let (range, path) = s.split_once(':')?;
        let (lo, hi) = range.split_once('-')?;
        let lo = u32::from_str_radix(lo.trim().trim_start_matches("0x"), 16).ok()?;
        let hi = u32::from_str_radix(hi.trim().trim_start_matches("0x"), 16).ok()?;
        Some((lo, hi, path.trim().to_string()))
    })
}

/// `VITASLOP_REGTRACE_WATCH=<hex guest addr>[,<hex guest addr>...]` - append the WORD
/// AT each address to every register-trace line, as extra `mNNNNNNNN=VVVVVVVV` fields.
///
/// The register trace alone answers "which block did the register go bad after"; this
/// answers the same question for a memory location, which is the one a corrupted stack
/// slot needs. It is deliberately not a watchpoint: `VITASLOP_WATCH_STORE` traps on the
/// first (or Nth) GUEST store to an address, which is useless for a heavily-reused
/// stack slot and blind to host-side writes - a host call writes guest memory straight
/// from Rust and never goes through a lifted store at all. Sampling the word at every
/// block entry catches the change whoever made it, and names the block it happened
/// under.
fn qdiff_regtrace_watch() -> &'static Vec<u32> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Vec<u32>> = OnceLock::new();
    CELL.get_or_init(|| {
        std::env::var("VITASLOP_REGTRACE_WATCH")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|a| u32::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok())
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// Set once the snapshot block has been dumped, so the register trace records only the
/// window from the snapshot forward (not earlier passes through the same blocks).
static QDIFF_SNAP_FIRED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// How many matching block entries have been seen (to honor the skip count).
static QDIFF_SNAP_SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// `VITASLOP_SNAPSHOT_SKIP=<n>` - skip the first `n` entries to the snapshot block before
/// firing (default 0 = first entry). Targets a specific call of a re-entered function
/// (e.g. the 4th `find` invocation) without the colon-ambiguity of packing it into the path.
fn qdiff_snapshot_skip() -> u32 {
    use std::sync::OnceLock;
    static CELL: OnceLock<u32> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_SNAPSHOT_SKIP")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(0)
    })
}

/// Lazily-opened register-trace file (plain `File`: each line is written straight
/// through, so no flush is needed at process exit where the static is leaked).
fn qdiff_regtrace_writer() -> &'static Mutex<Option<std::fs::File>> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(None))
}

/// Dump the full guest state (all non-zero pages + r0..r15 + NZCV) to `path`, in the
/// simple `VSNP` container the qdiff host tool reads. Fired once, at block `pc`.
fn qdiff_dump_snapshot<H: ImportDispatch + Send + 'static>(
    caller: &mut Caller<'_, ThreadData<H>>,
    pc: u32,
    path: &str,
) {
    let mut regs = [0u32; 16];
    for (i, r) in regs.iter_mut().enumerate() {
        *r = get_reg(caller, i);
    }
    let flags = [
        get_flag(caller, abi::Flag::N),
        get_flag(caller, abi::Flag::Z),
        get_flag(caller, abi::Flag::C),
        get_flag(caller, abi::Flag::V),
    ];
    let base = caller.data().base;
    let shared = caller.data().shared_mem.clone();
    let data = shared.data();
    // SAFETY: `UnsafeCell<u8>` is repr(transparent) over `u8`; under the cooperative
    // scheduler no other fiber runs while this svc handler executes on the live thread.
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
    // Coalesce contiguous non-zero 4 KiB pages into regions (sparse: skips the huge
    // zero gaps between image, heap, and stack).
    const PAGE: usize = 4096;
    // `VITASLOP_SNAPSHOT_DENSE=lo-hi` (hex guest addrs) forces every page in that range
    // into the snapshot even when it reads as all-zero. The default sparse dump omits
    // zero pages to stay small, but a loop that WRITES to fresh (currently-zero) pages
    // ahead of a marching pointer - a table build, memset, hash fill - would leave qemu
    // (which maps only the snapshot's pages) faulting on the first such store while our
    // engine, holding the whole linear memory, does not. Including those pages as zeros
    // lets the reference CPU follow the same stores. Multiple ranges are comma-separated.
    let dense_ranges: Vec<(u32, u32)> = std::env::var("VITASLOP_SNAPSHOT_DENSE")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|item| {
                    let (lo, hi) = item.split_once('-')?;
                    Some((
                        u32::from_str_radix(lo.trim().trim_start_matches("0x"), 16).ok()?,
                        u32::from_str_radix(hi.trim().trim_start_matches("0x"), 16).ok()?,
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    let page_dense = |page_addr: u32| dense_ranges.iter().any(|&(lo, hi)| page_addr < hi && page_addr + PAGE as u32 > lo);
    let mut regions: Vec<(u32, usize, usize)> = Vec::new(); // (guest_addr, linear_off, len)
    let mut i = 0usize;
    while i < bytes.len() {
        let end = (i + PAGE).min(bytes.len());
        if bytes[i..end].iter().any(|&b| b != 0) || page_dense(base + i as u32) {
            if let Some(last) = regions.last_mut() {
                if last.1 + last.2 == i {
                    last.2 += end - i;
                    i = end;
                    continue;
                }
            }
            regions.push((base + i as u32, i, end - i));
        }
        i = end;
    }
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"VSNP");
    buf.extend_from_slice(&1u32.to_le_bytes()); // version
    buf.extend_from_slice(&pc.to_le_bytes());
    for r in regs {
        buf.extend_from_slice(&r.to_le_bytes());
    }
    for f in flags {
        buf.extend_from_slice(&f.to_le_bytes());
    }
    buf.extend_from_slice(&(regions.len() as u32).to_le_bytes());
    for (addr, off, len) in &regions {
        buf.extend_from_slice(&addr.to_le_bytes());
        buf.extend_from_slice(&(*len as u32).to_le_bytes());
        buf.extend_from_slice(&bytes[*off..*off + *len]);
    }
    match std::fs::write(path, &buf) {
        Ok(()) => eprintln!(
            "[qdiff] snapshot at {pc:#010x}: {} region(s), {} bytes -> {path}",
            regions.len(),
            buf.len()
        ),
        Err(e) => eprintln!("[qdiff] snapshot write to {path} failed: {e}"),
    }
}

/// `VITASLOP_REGTRACE_MAX=<n>` caps the register trace at `n` lines (0 = unbounded).
/// Bounds the qdiff replay: a host-call-free loop emits millions of in-range blocks, but
/// a few hundred (a dozen loop iterations) already decide escape-vs-loop, and qdiff single
/// -continues qemu once per record, so an uncapped trace makes it crawl.
fn qdiff_regtrace_max() -> u64 {
    use std::sync::OnceLock;
    static CELL: OnceLock<u64> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_REGTRACE_MAX")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(0)
    })
}
static QDIFF_REGTRACE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Append one `pc r0..r15 n z c v` line (all hex, flags 0/1) to the register trace.
fn qdiff_log_regtrace<H: ImportDispatch + Send + 'static>(
    caller: &mut Caller<'_, ThreadData<H>>,
    pc: u32,
    path: &str,
) {
    let cap = qdiff_regtrace_max();
    if cap != 0 && QDIFF_REGTRACE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= cap {
        return;
    }
    let mut line = format!("{pc:08x}");
    for i in 0..16 {
        line.push_str(&format!(" {:08x}", get_reg(caller, i)));
    }
    for f in [abi::Flag::N, abi::Flag::Z, abi::Flag::C, abi::Flag::V] {
        line.push_str(&format!(" {}", get_flag(caller, f)));
    }
    // Optional watched memory words (VITASLOP_REGTRACE_WATCH). Appended after the
    // fixed reg+flag columns so the qdiff host tool's parser, which reads the leading
    // 21 fields, is unaffected.
    line.push_str(&watch_words(caller));
    line.push('\n');
    let mut guard = qdiff_regtrace_writer().lock().unwrap();
    if guard.is_none() {
        match std::fs::File::create(path) {
            Ok(f) => *guard = Some(f),
            Err(e) => {
                eprintln!("[qdiff] regtrace create {path} failed: {e}");
                return;
            }
        }
    }
    use std::io::Write;
    if let Some(w) = guard.as_mut() {
        let _ = w.write_all(line.as_bytes());
    }
}

// --- block-visit histogram (VITASLOP_BLOCK_HIST) --------------------------------
//
// A cheap, non-flooding companion to VITASLOP_TRACE_BLOCKS. For a hot loop, printing
// every block entry both floods stderr and (because the print costs guest fuel) shifts
// the preemptive schedule so successive runs cannot be cross-correlated. Instead this
// counts entries per block PC and records the exact PC sequence for the first few
// thousand entries. The result maps a loop's structure empirically: the hottest blocks
// are the loop body, the relative counts give the nesting/trip counts, and the recorded
// prefix shows the exact repeating cycle - whose head is the loop head (the block to
// snapshot for qemu-diff). Enable by transpiling with VITASLOP_TRACE_BLOCKS=lo-hi (so
// the per-block svc hooks are emitted) plus VITASLOP_BLOCK_HIST=1; dumped at run end by
// `dump_block_hist`. Zero cost when unset.
struct BlockHist {
    counts: std::collections::HashMap<u32, u64>,
    /// The most recent `SEQ_CAP` block PCs in execution order (a ring buffer, so it
    /// holds the STEADY-STATE cycle at run end, not the warmup prefix).
    seq: std::collections::VecDeque<u32>,
    total: u64,
}
const BLOCK_HIST_SEQ_CAP: usize = 8192;
static BLOCK_HIST: Mutex<Option<BlockHist>> = Mutex::new(None);

fn block_hist_enabled() -> bool {
    use std::sync::OnceLock;
    static CELL: OnceLock<bool> = OnceLock::new();
    *CELL.get_or_init(|| std::env::var("VITASLOP_BLOCK_HIST").is_ok())
}

fn block_hist_record(pc: u32) {
    let mut g = BLOCK_HIST.lock().unwrap();
    let h = g.get_or_insert_with(|| BlockHist {
        counts: std::collections::HashMap::new(),
        seq: std::collections::VecDeque::new(),
        total: 0,
    });
    *h.counts.entry(pc).or_insert(0) += 1;
    if h.seq.len() >= BLOCK_HIST_SEQ_CAP {
        h.seq.pop_front();
    }
    h.seq.push_back(pc);
    h.total += 1;
}

/// Print the block-visit histogram gathered under `VITASLOP_BLOCK_HIST`: the `top`
/// most-entered block PCs with their counts, the recorded entry-sequence prefix, and
/// the shortest exact repeating period detected in that prefix (the loop's cycle
/// length, if it settled into one). Call once after the run.
pub fn dump_block_hist(top: usize) {
    let g = BLOCK_HIST.lock().unwrap();
    let Some(h) = g.as_ref() else {
        return;
    };
    // If VITASLOP_BLOCK_HIST_SEQ=<path> is set, write the full recorded steady-state
    // entry sequence (one hex PC per line) there, so the macro-structure (outer-loop
    // period, pass boundaries) can be analyzed offline when it is not a short cycle.
    if let Ok(path) = std::env::var("VITASLOP_BLOCK_HIST_SEQ") {
        let mut s = String::with_capacity(h.seq.len() * 11);
        for pc in &h.seq {
            s.push_str(&format!("{pc:08x}\n"));
        }
        match std::fs::write(&path, &s) {
            Ok(()) => eprintln!("wrote {} block-seq entries to {path}", h.seq.len()),
            Err(e) => eprintln!("block-seq write to {path} failed: {e}"),
        }
    }
    eprintln!(
        "--- block histogram: {} distinct blocks, {} total entries (seq prefix {} of them) ---",
        h.counts.len(),
        h.total,
        h.seq.len()
    );
    let mut pairs: Vec<(u32, u64)> = h.counts.iter().map(|(&a, &c)| (a, c)).collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (addr, count) in pairs.iter().take(top) {
        eprintln!("  {addr:#010x}: {count}");
    }
    if pairs.len() > top {
        eprintln!("  ...({} more blocks)", pairs.len() - top);
    }
    // Detect the shortest period p such that the tail of the recorded sequence repeats
    // with period p (loop cycle length). Check the last 4*p entries agree. Searches up
    // to a long period so a big nested body (many blocks per outer iteration) is still
    // caught. The ring buffer holds the steady-state tail, so this is the real cycle.
    let seq: Vec<u32> = h.seq.iter().copied().collect();
    if seq.len() >= 8 {
        let mut period = None;
        for p in 1..=(seq.len() / 4) {
            let window = 4 * p;
            let start = seq.len() - window;
            if (start..seq.len() - p).all(|i| seq[i] == seq[i + p]) {
                period = Some(p);
                break;
            }
        }
        match period {
            Some(p) => {
                let cycle: Vec<String> =
                    seq[seq.len() - p..].iter().map(|a| format!("{a:#x}")).collect();
                eprintln!("  detected loop cycle length {p}; blocks in one cycle (order):");
                // Wrap the (possibly long) cycle over several lines for readability.
                for chunk in cycle.chunks(12) {
                    eprintln!("    {}", chunk.join(" -> "));
                }
            }
            None => eprintln!("  no exact short period in the recorded tail (nested/irregular)"),
        }
    }
}

/// Bind `env.dispatch_miss`: an indirect call that resolves to no translated
/// function traps here with the faulting `(target, caller)` addresses, so an
/// unmapped `blx`/`bx` target surfaces as a clear report (with the trapping thread's
/// register dump, added by `instantiate_thread_seq`) instead of an opaque
/// `unreachable`.
fn bind_dispatch_miss<H: ImportDispatch + Send + 'static>(
    linker: &mut Linker<ThreadData<H>>,
) -> Result<(), RunError> {
    linker
        .func_wrap_async(
            abi::IMPORT_MODULE,
            abi::DISPATCH_MISS_NAME,
            |_caller: Caller<'_, ThreadData<H>>, (target, caller): (i32, i32)| {
                Box::new(async move {
                    Err::<(), wasmtime::Error>(wasmtime::Error::msg(format!(
                        "indirect dispatch to unknown target {:#010x} from f_{:x}",
                        target as u32, caller as u32
                    )))
                })
            },
        )
        .map_err(|e| RunError::Wasm(e.to_string()))?;
    Ok(())
}

/// Diagnostic memory watchpoint state (see the poll in `bind_import`).
static POLL_LAST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static POLL_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Host calls between periodic stall dumps, matching the browser's window so the two
/// engines' dumps line up call-for-call.
const STALL_DUMP_EVERY: u64 = 250_000;
static STALL_DUMP_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Guest address to sample after each host call, from `VITASLOP_POLL_ADDR` (hex).
fn poll_addr() -> Option<u32> {
    use std::sync::OnceLock;
    static CELL: OnceLock<Option<u32>> = OnceLock::new();
    *CELL.get_or_init(|| {
        std::env::var("VITASLOP_POLL_ADDR")
            .ok()
            .and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok())
    })
}

/// Bind the `env.import` NID trap: read guest state, dispatch to the shared host
/// as the current thread, write state back, and act on the outcome - suspend for
/// a block or a frame flip, trap for a thread exit or a process halt.
fn bind_import<H: ImportDispatch + Send + 'static>(
    linker: &mut Linker<ThreadData<H>>,
) -> Result<(), RunError> {
    linker
        .func_wrap_async(
            abi::IMPORT_MODULE,
            abi::IMPORT_NAME,
            |mut caller: Caller<'_, ThreadData<H>>, (selector,): (i32,)| {
                Box::new(async move {
                    let perf = crate::perf::enabled();
                    let call_start = perf.then(std::time::Instant::now);

                    let mut regs = [0u32; abi::REG_COUNT];
                    let mut vfp = [0u32; VFP_ARG_COUNT];
                    read_guest_regs(&mut caller, &mut regs, &mut vfp);
                    // The state as the guest left it, so write-back can skip every
                    // register the handler did not touch.
                    let before = (regs, vfp);

                    let dispatch_start = perf.then(std::time::Instant::now);
                    let outcome = {
                        let data = caller.data();
                        let thid = data.thid;
                        let base = data.base;
                        let shared = data.shared_mem.clone();
                        let mut view = SharedView::new(&shared, data.dirty_off);
                        let mut host = data.host.lock().unwrap();
                        host.set_current_thread(thid);
                        host.dispatch(selector as u32, &mut regs, &mut vfp, &mut view, base)
                    };
                    let dispatch_ns = dispatch_start.map_or(0, |t| t.elapsed().as_nanos() as u64);

                    write_guest_regs(&mut caller, &before, &regs, &vfp);

                    // Charged before the outcome is acted on: a `Block` outcome parks
                    // the fiber here, and the time the thread spends descheduled is the
                    // scheduler's, not this call's.
                    if let Some(t) = call_start {
                        crate::perf::note_import(
                            selector as u32,
                            t.elapsed().as_nanos() as u64,
                            dispatch_ns,
                        );
                    }

                    // Periodic stall dump (`vitaslop::sched=debug`), on the same schedule
                    // and in the same format as the browser's, so a run that stalls on
                    // one engine and not the other can be diffed line for line rather
                    // than argued about. Off unless the target is enabled, and the
                    // counter is a relaxed add.
                    {
                        let n = STALL_DUMP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if n % STALL_DUMP_EVERY == 0 && tracing::enabled!(target: "vitaslop::sched", tracing::Level::DEBUG) {
                            let host = caller.data().host.lock().unwrap();
                            tracing::debug!(
                                target: "vitaslop::sched",
                                "stall dump after {n} host calls:\n{}",
                                host.sync_dump()
                            );
                        }
                    }

                    // Diagnostic memory watchpoint (VITASLOP_POLL_ADDR=<hex guest
                    // addr>): sample the word after every host call and report each
                    // transition. Path- and thread-agnostic - catches writes my
                    // transpiler-side store watchpoint misses (NEON/multi stores) or
                    // that happen on another thread's instance. Brackets the writer
                    // to a [prev host call, this host call] window by NID.
                    if let Some(addr) = poll_addr() {
                        let data = caller.data();
                        let off = addr.wrapping_sub(data.base) as usize;
                        let shared = data.shared_mem.data();
                        if off + 4 <= shared.len() {
                            let mut b = [0u8; 4];
                            for (i, bb) in b.iter_mut().enumerate() {
                                // SAFETY: bounds checked; cooperative scheduling.
                                unsafe {
                                    *bb = *shared[off + i].get();
                                }
                            }
                            let now = u32::from_le_bytes(b);
                            let prev = POLL_LAST.swap(now, std::sync::atomic::Ordering::Relaxed);
                            let n = POLL_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if now != prev {
                                eprintln!(
                                    "[poll] {addr:#010x}: {prev:#010x} -> {now:#010x} \
                                     after host call #{n} sel={:#x} thid={:#x}",
                                    selector, data.thid
                                );
                            }
                        }
                    }

                    match outcome {
                        SvcOutcome::Continue => {}
                        SvcOutcome::Reschedule => {
                            // Stay runnable but suspend so the scheduler re-picks by
                            // priority now (a higher-priority thread just became
                            // runnable and must preempt us).
                            note_suspend(&mut caller, Stop::Quantum, selector as u32);
                            YieldNow(false).await;
                        }
                        SvcOutcome::Block => {
                            note_suspend(&mut caller, Stop::Blocked, selector as u32);
                            YieldNow(false).await;
                            // Resumed. A timed wait that expired owes this thread a
                            // return code other than the 0 it parked with (a
                            // WAIT_TIMEOUT); apply it to r0 before returning to the
                            // guest. A signal wake has no code and keeps r0 = 0.
                            let code = {
                                let data = caller.data();
                                let thid = data.thid;
                                data.host.lock().unwrap().take_resume_code(thid)
                            };
                            if let Some(code) = code {
                                set_reg_caller(&mut caller, 0, code); // r0 = return value
                            }
                        }
                        SvcOutcome::Flip => {
                            note_suspend(&mut caller, Stop::Flip, selector as u32);
                            YieldNow(false).await;
                        }
                        SvcOutcome::ThreadExit => {
                            caller.data_mut().thread_exit = true;
                            return Err(wasmtime::Error::msg("thread exit"));
                        }
                        SvcOutcome::Halt => {
                            caller.data_mut().process_halt = true;
                            return Err(wasmtime::Error::msg("process halt"));
                        }
                        SvcOutcome::Fatal(msg) => {
                            // Unfaithful call (e.g. unimplemented NID): stop the run
                            // loudly. Stash the message and unwind; the fiber surfaces
                            // it as FiberEnd::Error -> RunReport::Error.
                            caller.data_mut().fatal = Some(msg.clone());
                            return Err(wasmtime::Error::msg(msg));
                        }
                    }
                    Ok(())
                })
            },
        )
        .map_err(|e| RunError::Wasm(e.to_string()))?;
    Ok(())
}

/// Record that this thread is about to suspend at `stop`, and sample the fuel it has
/// burned so far.
///
/// The fuel reading is the point: `Caller::get_fuel` is the exact remaining fuel of this
/// thread's store, and the store started at `u64::MAX`, so the difference is everything the
/// thread has executed. Sampling it HERE - at the switch point, before the scheduler can
/// look - is what lets the game clock be charged for guest work rather than for the number
/// of times a thread happened to stop. See [`Signal::fuel`].
fn note_suspend<H: ImportDispatch + Send + 'static>(
    caller: &mut Caller<'_, ThreadData<H>>,
    stop: Stop,
    selector: u32,
) {
    // `get_fuel` fails only on a store without fuel consumption enabled, which this engine
    // always enables; leave the previous reading rather than invent one if it ever does.
    let burned = caller.get_fuel().ok().map(|left| u64::MAX - left);
    sample_software_fuel(caller, burned, selector);
    let retired = sample_arm_instructions(caller);
    let mut s = caller.data().signal.lock().unwrap();
    s.stop = stop;
    s.host_suspends = s.host_suspends.wrapping_add(1);
    if let Some(burned) = burned {
        s.fuel = burned;
    }
    if let Some(retired) = retired {
        s.arm = retired;
    }
}

/// Read this thread's cumulative GUEST ARM INSTRUCTION count out of the emitted work
/// counter's high half (see `abi::WORK_GLOBAL`). `None` on a build without the counter.
///
/// This is the reading the game clock is billed from - see
/// [`ThreadHandle::arm_retired`], which is where the unit is argued for.
fn sample_arm_instructions<H: ImportDispatch + Send + 'static>(
    caller: &mut Caller<'_, ThreadData<H>>,
) -> Option<u64> {
    if caller.data().fuel_interval == 0 {
        return None;
    }
    let g = caller.data().sw_fuel.clone()?;
    let packed = g.get(&mut *caller).i64()? as u64;
    Some(packed >> abi::WORK_INSTR_SHIFT)
}

/// Times a fiber suspended without any of our code running, on a build where the emitted
/// work check is the only preemption there is. See [`GuestThread::resume`].
static UNATTRIBUTED_SUSPENDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Report a suspend nothing of ours produced, the FIRST time it happens. Silence here
/// would read as the ordinary path, which is exactly what it is not.
fn note_unattributed_suspend() {
    use std::sync::atomic::Ordering::Relaxed;
    if UNATTRIBUTED_SUSPENDS.fetch_add(1, Relaxed) == 0 {
        tracing::warn!(
            target: "vitaslop::sched",
            "a guest fiber suspended without passing through the host-call closure, on a \
             build whose only preemption is the emitted work check. Nothing in this engine \
             should be able to do that (wasmtime's periodic yield is off), so the thread's \
             burn for that slice is priced from a stale reading",
        );
    }
}

/// Suspends nothing of ours produced, over the whole run. Zero is the expected reading;
/// see [`note_unattributed_suspend`].
pub fn unattributed_suspends() -> u64 {
    UNATTRIBUTED_SUSPENDS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Report the one reading that says the software work counter is being read wrong: it
/// went DOWN between two samples, which can only mean a clear this closure never saw.
fn note_software_fuel_went_backwards(last: i64, now: i64) {
    use std::sync::atomic::Ordering::Relaxed;
    static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if SEEN.fetch_add(1, Relaxed) == 0 {
        tracing::warn!(
            target: "vitaslop::sched",
            "the software work counter went BACKWARDS ({last} -> {now}): the operator half \
             was cleared by a fuel yield this host never saw, so its comparison against \
             wasmtime undercounts by whole intervals",
        );
    }
}

/// The comparison totals: `(software fuel burned, wasmtime fuel burned over the same
/// intervals, samples)`. A process-global because it is a whole-run diagnostic and the
/// threads it is summed over are released as they finish.
static SW_FUEL: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// `(software fuel, wasmtime fuel, samples)` over the whole run, or `None` when this
/// build emitted no software fuel.
///
/// # It runs on every retail run now, and it is a STANDING cross-check
/// The retail path emits the work counter unconditionally (it is what preempts and what
/// the game clock is billed from), so this comparison is no longer an opt-in experiment.
/// It costs one global read per switch point and it is the only thing in the project that
/// can say the counter still agrees with a real engine's accounting after a codegen
/// change. A ratio that walks away from one is a codegen change the counter does not
/// model.
///
/// # What it is for
/// The browser has no engine fuel, so the transpiler emits a counter that is supposed to
/// reproduce wasmtime's own accounting - its operator cost table and its flush points -
/// and the browser's game clock is driven by it. That claim was measured against the
/// desktop's and disagreed by a factor of ten, which is either a wrong counter or a
/// browser that really does ten times the work; nothing in a browser run can tell those
/// apart, because it has no second opinion.
///
/// Running the SAME module on wasmtime with the counter compiled in gives it one. Both
/// numbers here are sampled at the same instants (every host-call switch point) and cover
/// the same guest execution, so their ratio is the counter's error and nothing else.
///
/// Note the wasmtime side is measured on an INSTRUMENTED module, so it legitimately reads
/// a little high - it bills the counter's own `global.get`/`i32.sub`/`global.set`, which
/// the module native normally runs does not contain. That biases the ratio TOWARDS one,
/// so a ratio far from one is not an artefact of the experiment.
pub fn software_fuel_report() -> Option<(u64, u64, u64)> {
    use std::sync::atomic::Ordering::Relaxed;
    let samples = SW_FUEL[2].load(Relaxed);
    if samples == 0 {
        return None;
    }
    Some((SW_FUEL[0].load(Relaxed), SW_FUEL[1].load(Relaxed), samples))
}

/// Read this thread's software fuel counter, difference it the way the browser host does,
/// and accumulate it next to wasmtime's reading for the same interval. A no-op on a build
/// without software fuel, which is every ordinary native run.
fn sample_software_fuel<H: ImportDispatch + Send + 'static>(
    caller: &mut Caller<'_, ThreadData<H>>,
    wasmtime_total: Option<u64>,
    selector: u32,
) {
    // The global is exported unconditionally; it only COUNTS when this build emitted
    // fuel. Without this the report would fire on every ordinary run and read a flat
    // zero as "the counter says the guest did no work".
    let interval = i64::from(caller.data().fuel_interval);
    if interval == 0 {
        return;
    }
    let Some(g) = caller.data().sw_fuel.clone() else { return };
    // The counter is the PACKED i64 work global: operators in the low half, guest
    // instructions in the high half (see `abi::WORK_GLOBAL`). Only the operator half is
    // wasmtime's unit, so only that half is comparable with wasmtime's own reading.
    let Some(now) = g.get(&mut *caller).i64().map(|v| v & abi::WORK_OPS_MASK) else { return };
    let last = caller.data().sw_last;
    // >>> THE CLEAR IS TRACKED, NOT INFERRED FROM THE READING, AND THAT IS THE WHOLE
    // ACCURACY OF THIS INSTRUMENT. The operator half counts UP and the emitted code zeroes
    // it immediately after each fuel yield, so a reading below the last one means a yield
    // happened in between - but "how MANY" is not recoverable from two readings, and
    // guessing one undercounts by a whole interval for every extra yield. MEASURED with
    // the guess in place: the counter read 0.41x wasmtime over a 3000-frame run, which
    // looks exactly like a counter that bills 41% of what it should.
    //
    // Every clear is preceded by a `FUEL_SELECTOR` call through this very closure, so the
    // clears can be COUNTED instead: after one, the next reading starts from zero.
    let burned = (now - last).max(0);
    caller.data_mut().sw_last =
        if selector == vitaslop_transpiler::abi::FUEL_SELECTOR { 0 } else { now };
    // A reading below the last one is now impossible - it would mean a clear this closure
    // never saw, i.e. an operator charge going unbilled somewhere.
    if now < last {
        note_software_fuel_went_backwards(last, now);
    }

    use std::sync::atomic::Ordering::Relaxed;
    SW_FUEL[0].fetch_add(burned.max(0) as u64, Relaxed);
    // wasmtime's side is a cumulative total, so difference it per thread the same way.
    if let Some(total) = wasmtime_total {
        let prev = caller.data().sw_wasmtime_last;
        caller.data_mut().sw_wasmtime_last = total;
        SW_FUEL[1].fetch_add(total.saturating_sub(prev), Relaxed);
    }
    SW_FUEL[2].fetch_add(1, Relaxed);
}

/// A future that suspends the fiber exactly once (the switch point), then
/// completes. The scheduler decides when the fiber is polled again - immediately
/// for a fuel/frame yield, or only after a wake for a block.
struct YieldNow(bool);
impl Future for YieldNow {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            Poll::Pending
        }
    }
}

/// A [`GuestMemory`](vitaslop_runtime::GuestMemory) view over the shared linear
/// memory. Cooperative single-OS-thread scheduling means only one guest thread's
/// host call touches this at a time, so the unsynchronized access is sound.
struct SharedView {
    ptr: *mut u8,
    len: usize,
    /// Offset of the guest-store dirty block, when this build was transpiled with one.
    /// See [`ThreadData::dirty_off`] for why an ordinary native build has none.
    dirty_off: Option<u64>,
}

impl SharedView {
    fn new(mem: &SharedMemory, dirty_off: Option<u64>) -> SharedView {
        let data = mem.data();
        SharedView { ptr: data.as_ptr() as *mut u8, len: data.len(), dirty_off }
    }

    /// The dirty block's `[epoch byte][page map]`, as one mutable slice.
    ///
    /// SAFETY of the caller's use: the block lies above the guest region inside the
    /// same linear memory, and scheduling is cooperative, so no fiber runs while a host
    /// call holds this.
    unsafe fn dirty_block(&self) -> Option<&mut [u8]> {
        let off = self.dirty_off? as usize;
        let end = self.len.min(off + vitaslop_transpiler::DIRTY_MAP_OFF as usize + self.pages() + 1);
        Some(std::slice::from_raw_parts_mut(self.ptr.add(off), end - off))
    }

    fn pages(&self) -> usize {
        self.len >> vitaslop_transpiler::DIRTY_SHIFT
    }

    /// Stamp every page `[off, off + len)` touches with the current epoch.
    ///
    /// # This is not an extra - it is the other half of the map
    /// The transpiler stamps what the GUEST stores, and a host call writes guest memory
    /// too: a file read, a `memcpy` NID, a GXM transfer. Those writes are invisible to
    /// translated code [[vitaslop-host-write-watch]], so a map that only the guest
    /// wrote would report a texture the host had just overwritten as untouched - a
    /// silent stale texture, the exact bug the compare exists to prevent. Every backing
    /// with a map must do this in its `write`.
    fn stamp_written(&self, off: usize, len: usize) {
        if len == 0 {
            return;
        }
        // SAFETY: see `dirty_block`.
        let Some(block) = (unsafe { self.dirty_block() }) else { return };
        let epoch = block[vitaslop_transpiler::DIRTY_EPOCH_OFF as usize];
        let map = &mut block[vitaslop_transpiler::DIRTY_MAP_OFF as usize..];
        let shift = vitaslop_transpiler::DIRTY_SHIFT;
        let first = off >> shift;
        let last = ((off + len - 1) >> shift).min(map.len().saturating_sub(1));
        map[first..=last].fill(epoch);
    }
}

impl vitaslop_runtime::GuestMemory for SharedView {
    fn len(&self) -> usize {
        self.len
    }
    fn read(&self, off: usize, buf: &mut [u8]) {
        // SAFETY: `off + buf.len() <= len` is guaranteed by GuestCtx bounds checks,
        // and no other thread runs concurrently (cooperative scheduling).
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.add(off), buf.as_mut_ptr(), buf.len());
        }
    }
    fn write(&mut self, off: usize, bytes: &[u8]) {
        // SAFETY: see `read`.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(off), bytes.len());
        }
        self.stamp_written(off, bytes.len());
    }
    fn borrow(&self, off: usize, len: usize) -> Option<&[u8]> {
        if off.checked_add(len)? > self.len {
            return None;
        }
        // SAFETY: bounds checked above, and the scheduler is cooperative - no fiber
        // runs while a host call holds this borrow, so the bytes cannot change under
        // it. The lifetime is tied to `&self`, which lives only for the host call.
        Some(unsafe { std::slice::from_raw_parts(self.ptr.add(off), len) })
    }

    fn dirty_since(&self, off: usize, len: usize, stamp: u8) -> Option<bool> {
        if len == 0 {
            return Some(false);
        }
        // SAFETY: see `dirty_block`.
        let block = unsafe { self.dirty_block()? };
        let map = &block[vitaslop_transpiler::DIRTY_MAP_OFF as usize..];
        let shift = vitaslop_transpiler::DIRTY_SHIFT;
        // One page BELOW the range too - a store is stamped against the page it STARTS
        // in. See `GuestMemory::dirty_since`.
        let first = (off >> shift).saturating_sub(1);
        // `max(first)` so a clamped range is a single page rather than a reversed slice.
        let last = ((off + len - 1) >> shift).min(map.len().saturating_sub(1)).max(first);
        Some(map[first..=last].iter().any(|&s| s >= stamp))
    }

    fn dirty_epoch(&self) -> Option<u8> {
        // SAFETY: see `dirty_block`.
        let block = unsafe { self.dirty_block()? };
        Some(block[vitaslop_transpiler::DIRTY_EPOCH_OFF as usize])
    }

    /// See `GuestMemory::rebase_dirty_epoch`. Mirrors the browser's, over the slice.
    fn rebase_dirty_epoch(&self, floor: u8) -> Option<u8> {
        // SAFETY: see `dirty_block`.
        let block = unsafe { self.dirty_block()? };
        for p in block[vitaslop_transpiler::DIRTY_MAP_OFF as usize..].iter_mut() {
            *p = if *p >= floor { *p - floor + 1 } else { 0 };
        }
        let at = vitaslop_transpiler::DIRTY_EPOCH_OFF as usize;
        let next = if block[at] >= floor { block[at] - floor + 1 } else { 1 };
        block[at] = next;
        Some(next)
    }

    fn bump_dirty_epoch(&self) -> Option<(u8, bool)> {
        // SAFETY: see `dirty_block`.
        let block = unsafe { self.dirty_block()? };
        let next = block[vitaslop_transpiler::DIRTY_EPOCH_OFF as usize].wrapping_add(1);
        // A one-byte epoch compared with `>=` may not wrap silently - see the browser's
        // impl, which this mirrors exactly, for why the map is zeroed instead.
        if next == 0 || next == u8::MAX {
            block[vitaslop_transpiler::DIRTY_MAP_OFF as usize..].fill(0);
            block[vitaslop_transpiler::DIRTY_EPOCH_OFF as usize] = 1;
            return Some((1, true));
        }
        block[vitaslop_transpiler::DIRTY_EPOCH_OFF as usize] = next;
        Some((next, false))
    }
}

/// Write `bytes` into the shared memory at `off` (host-side seeding).
fn write_shared(mem: &SharedMemory, off: usize, bytes: &[u8]) {
    let data = mem.data();
    for (i, &b) in bytes.iter().enumerate() {
        // SAFETY: called only at points where no fiber is running - initial image
        // seeding (before any fiber starts) and a scheduler drain (between fiber
        // steps, one fiber at a time) - so there is no concurrent guest access.
        unsafe {
            *data[off + i].get() = b;
        }
    }
}

/// A concise trap description (kind + message), matching the sync `Vm`'s detail.
/// Snapshot the guest core register file (r0..r15) at the point of a trap. Because
/// every lifted register read/write goes straight to its global (no caching in wasm
/// locals across statements), the globals hold the exact ARM state at the faulting
/// instruction - so on a MemoryOutOfBounds this shows the garbage pointer that was
/// dereferenced (and `this` in r0..r3 for a C++ vtable dispatch). Diagnostic only.
///
/// >>> THAT "no caching in wasm locals" IS A REAL PRECONDITION, AND ONE BUILD BREAKS IT.
/// `VITASLOP_PROMOTE_REGS` holds the register file in wasm LOCALS along each straight-line
/// run and writes back only at calls, branches and returns (see `transpiler::promote`). A
/// trap in the middle of such a run therefore leaves the globals holding the values from
/// the last write-back, not the faulting instruction - so this dump goes STALE rather than
/// wrong-looking, which is the worse failure for a diagnostic. The knob is off by default;
/// if it is ever made the default, this dump has to spill the promoted locals first or say
/// that it cannot.
fn reg_dump<T>(store: &mut Store<T>, instance: &Instance) -> String {
    let mut s = String::from("regs at trap:");
    for i in 0..abi::REG_COUNT {
        let name = match i {
            abi::SP => "sp".to_string(),
            abi::LR => "lr".to_string(),
            abi::PC => "pc".to_string(),
            n => format!("r{n}"),
        };
        if i % 4 == 0 {
            s.push_str("\n  ");
        }
        s.push_str(&format!("{name}={:#010x} ", get_reg_store(store, instance, i)));
    }
    // Diagnostic guest-PC tracker (nonzero only when the module was emitted with
    // VITASLOP_TRACK_PC): the address of the basic block executing at the trap.
    if let Some(g) = instance.get_global(&mut *store, abi::GUEST_PC_EXPORT) {
        let pc = g.get(&mut *store).i32().unwrap_or(0) as u32;
        if pc != 0 {
            s.push_str(&format!("\n  guest_block={pc:#010x}"));
        }
    }
    s
}

/// Guest address of each transpiled function, indexed by wasm function index minus
/// [`abi::IMPORT_FUNC_COUNT`]. Recorded once, when the module is built.
static FUNC_ADDRS: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();

/// Record the emitted module's function table so a trap backtrace can name guest code.
///
/// # Why a process-wide latch rather than plumbing
/// A trap surfaces deep inside a fiber closure that has no reference to the artifact, and
/// the module is built exactly once per run. Threading the table down to that point would
/// touch every layer between for a diagnostic; a `OnceLock` set at build time reaches it
/// from anywhere and cannot be set twice.
pub fn record_function_addresses(addrs: Vec<u32>) {
    let _ = FUNC_ADDRS.set(addrs);
}

/// Rewrite `<wasm function N>` in a trap backtrace to name the GUEST function it is.
///
/// # Why this is not optional
/// A wasm backtrace is a list of module indices. On its own it says a fault happened
/// nineteen frames deep and nothing about where, and the indices are useless to anyone
/// without the module in front of them - a device capture pasted into a chat window is
/// exactly that situation. The mapping is `funcs[N - IMPORT_FUNC_COUNT]` (the imports occupy
/// the low indices), which is the same arithmetic the retail probe's `VITASLOP_WASM_INDICES`
/// does by hand, applied where the trap is actually printed.
///
/// MEASURED on the crash this was written for: the raw backtrace is
/// `6676 -> 12889 -> 6808` repeating nineteen frames, which reads as noise. Named, the
/// repeat is a guest routine recursing through the indirect-call dispatcher, which is a
/// description of the bug.
fn name_guest_frames(s: &str) -> String {
    let Some(addrs) = FUNC_ADDRS.get() else { return s.to_string() };
    const MARK: &str = "<wasm function ";
    let mut out = String::with_capacity(s.len() + 32);
    let mut rest = s;
    while let Some(at) = rest.find(MARK) {
        let (head, tail) = rest.split_at(at + MARK.len());
        out.push_str(head);
        let end = tail.find('>').unwrap_or(tail.len());
        let (num, after) = tail.split_at(end);
        out.push_str(num);
        if let Ok(widx) = num.trim().parse::<usize>() {
            match widx.checked_sub(abi::IMPORT_FUNC_COUNT as usize).and_then(|i| addrs.get(i)) {
                Some(a) => out.push_str(&format!(" = guest {a:#010x}")),
                // Above every guest function sit the dispatcher and `reset`, emitted last.
                // Saying so is as useful as an address: a backtrace that ALTERNATES with the
                // dispatcher is an indirect-call chain, which is a fact about the bug.
                None => out.push_str(" = dispatcher/reset (not guest code)"),
            }
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

fn trap_detail(e: &wasmtime::Error) -> String {
    let mut s = match e.downcast_ref::<wasmtime::Trap>() {
        Some(t) => format!("{t:?}: {e}"),
        None => e.to_string(),
    };
    // Walk the CAUSE CHAIN. Without this the report is the wrapper alone - "error while
    // executing at wasm backtrace: ..." - which names the frames and not the reason, and
    // the reason is the whole point. A host import that fails, or a panic caught at the
    // boundary, carries its explanation one or more levels down.
    let mut src = std::error::Error::source(e.as_ref() as &dyn std::error::Error);
    let mut depth = 0;
    while let Some(c) = src {
        // A wasm backtrace repeated at every level would bury the causes.
        let text = c.to_string();
        if !s.contains(&text) {
            s.push_str(&format!("\ncaused by: {text}"));
        }
        src = c.source();
        depth += 1;
        if depth > 8 {
            break;
        }
    }
    // Name the guest functions in the backtrace. Done here, at the one place a trap is
    // turned into text, so every caller gets it and none has to remember.
    name_guest_frames(&s)
}

// --- register/vfp accessors (Caller during a call, Store during setup) --------

/// Marshal the guest register file and VFP argument registers OUT of the wasm
/// globals into host arrays, through this thread's cached handles ([`GuestGlobals`]).
/// This and its write-back counterpart are the whole host-call marshalling cost, so
/// they take the register file in one call rather than per register.
fn read_guest_regs<H: ImportDispatch + Send + 'static>(
    caller: &mut Caller<'_, ThreadData<H>>,
    regs: &mut [u32; abi::REG_COUNT],
    vfp: &mut [u32; VFP_ARG_COUNT],
) {
    let g = caller.data().globals.as_ref().expect("globals resolved at instantiation");
    // `Global` is a Copy handle; copy them out so the store can be borrowed mutably.
    let (rg, vg) = (g.regs, g.vfp);
    for (i, r) in regs.iter_mut().enumerate() {
        *r = rg[i].get(&mut *caller).i32().expect("register global is i32") as u32;
    }
    for (i, s) in vfp.iter_mut().enumerate() {
        *s = vg[i].get(&mut *caller).i32().expect("vfp global is i32") as u32;
    }
}

/// Marshal the register file back INTO the wasm globals after a host call.
///
/// Any register a handler CHANGED is written, not just the ABI's return registers: a
/// handler may rewrite the whole file (a context switch does). But a register it left
/// alone is written back to the identical value it was read from, so comparing against
/// `before` and skipping those is exactly equivalent and much cheaper - a compare is a
/// couple of instructions where a `Global::set` is a store lookup, a type check and a
/// `Val` round trip. The typical call returns in r0 and touches nothing else, so this
/// turns 32 global writes into one.
fn write_guest_regs<H: ImportDispatch + Send + 'static>(
    caller: &mut Caller<'_, ThreadData<H>>,
    before: &([u32; abi::REG_COUNT], [u32; VFP_ARG_COUNT]),
    regs: &[u32; abi::REG_COUNT],
    vfp: &[u32; VFP_ARG_COUNT],
) {
    let g = caller.data().globals.as_ref().expect("globals resolved at instantiation");
    let (rg, vg) = (g.regs, g.vfp);
    for (i, &v) in regs.iter().enumerate() {
        if v != before.0[i] {
            rg[i].set(&mut *caller, Val::I32(v as i32)).expect("register global is mutable i32");
        }
    }
    for (i, &v) in vfp.iter().enumerate() {
        if v != before.1[i] {
            vg[i].set(&mut *caller, Val::I32(v as i32)).expect("vfp global is mutable i32");
        }
    }
}

fn get_reg<T>(caller: &mut Caller<'_, T>, i: usize) -> u32 {
    caller
        .get_export(&abi::reg_export(i))
        .and_then(|e| e.into_global())
        .expect("module exports registers")
        .get(&mut *caller)
        .i32()
        .expect("register global is i32") as u32
}

/// Read condition flag `f` (0 or 1) from its wasm global. Absent global (a title that
/// never uses the flag) reads as 0.
fn get_flag<T>(caller: &mut Caller<'_, T>, f: abi::Flag) -> u32 {
    caller
        .get_export(abi::flag_export(f))
        .and_then(|e| e.into_global())
        .map(|g| g.get(&mut *caller).i32().unwrap_or(0) as u32)
        .unwrap_or(0)
}

fn set_reg_caller<T>(caller: &mut Caller<'_, T>, i: usize, v: u32) {
    caller
        .get_export(&abi::reg_export(i))
        .and_then(|e| e.into_global())
        .expect("module exports registers")
        .set(&mut *caller, Val::I32(v as i32))
        .expect("register global is mutable i32");
}

fn set_reg_store<T>(store: &mut Store<T>, instance: &Instance, i: usize, v: u32) {
    instance
        .get_global(&mut *store, &abi::reg_export(i))
        .expect("module exports registers")
        .set(&mut *store, Val::I32(v as i32))
        .expect("register global is mutable i32");
}

/// Seed the per-thread pointer global (`tp`, ARM `TPIDRURO`) for this thread's
/// instance. A title with no TLS leaves the export absent and every read of the
/// thread pointer is dead code, so a missing global is not an error.
fn set_tp_store<T>(store: &mut Store<T>, instance: &Instance, tp: u32) {
    if let Some(g) = instance.get_global(&mut *store, abi::TP_EXPORT) {
        let _ = g.set(&mut *store, Val::I32(tp as i32));
    }
}

/// Copy `len` bytes within the shared guest memory (`src_off` -> `dst_off`, both
/// linear offsets). Used to lay a TLS template's initialized data into a thread's
/// private block. Called only when no fiber is running (thread instantiation), so
/// there is no concurrent guest access.
fn copy_shared(mem: &SharedMemory, src_off: usize, dst_off: usize, len: usize) {
    let data = mem.data();
    for i in 0..len {
        // SAFETY: as write_shared - no fiber runs during instantiation. The TLS
        // block is freshly allocated above the image, so it never overlaps the source.
        unsafe {
            *data[dst_off + i].get() = *data[src_off + i].get();
        }
    }
}

fn get_reg_store<T>(store: &mut Store<T>, instance: &Instance, i: usize) -> u32 {
    instance
        .get_global(&mut *store, &abi::reg_export(i))
        .expect("module exports registers")
        .get(&mut *store)
        .i32()
        .expect("register global is i32") as u32
}

/// The scheduler wants `Reentry` (shared with the synchronous re-entry path) as
/// its spawn descriptor; re-export the type so hosts naming it need only this crate.
pub use vitaslop_runtime::Reentry as ThreadSpawn;
