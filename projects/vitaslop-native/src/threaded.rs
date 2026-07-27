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
const DEFAULT_QUANTUM_FUEL: u64 = 1_000_000;

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
}

/// The scheduler's per-thread store data: the shared host, this thread's id, the
/// shared memory (so a host call can view guest memory), the image base, and the
/// flags the host-call closure raises for the fiber's return value.
struct ThreadData<H: ImportDispatch + Send + 'static> {
    host: Arc<Mutex<H>>,
    thid: i32,
    shared_mem: SharedMemory,
    base: u32,
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
}

impl ThreadHandle for WasmtimeThread {
    fn thid(&self) -> i32 {
        self.thid
    }
    fn priority(&self) -> i32 {
        self.priority
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
        self.signal.lock().unwrap().stop = Stop::Quantum;
        match self.future.as_mut().poll(&mut cx) {
            Poll::Ready(end) => ThreadStep::Finished(end),
            Poll::Pending => ThreadStep::Suspended(self.signal.lock().unwrap().stop),
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
    /// Linear-memory offset of the "diagnostics armed" word, when this build was
    /// transpiled with `VITASLOP_ARM_AT_FRAME` (see
    /// [`vitaslop_transpiler::arm_at_frame`]). `None` in an ordinary build.
    arm_word_off: Option<u64>,
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
        }
    }

    /// Arm the frame-gated diagnostics the instant the run reaches the requested
    /// frame. One word in shared linear memory covers every guest thread at once,
    /// which is the whole reason the gate is not a wasm global.
    fn on_frame(&mut self, frames: u64) {
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
}

/// Set once the run reaches `VITASLOP_ARM_AT_FRAME`, for the HOST-side diagnostics
/// (the qemu-diff snapshot and register trace) that live in this file rather than in
/// emitted code. True from the start when no frame gate was requested, so an
/// ungated run behaves exactly as before.
static DIAG_ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

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
            arm_word_off: artifact.arm_word_off,
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
        let built = transpiler::transpile_lenient(&linked.shared_program());
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
            arm_word_off: built.artifact.arm_word_off,
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
        let signal = Arc::new(Mutex::new(Signal { stop: Stop::Quantum }));
        let data = ThreadData {
            host: self.host.clone(),
            thid,
            shared_mem: self.shared_mem.clone(),
            base: self.base,
            signal: signal.clone(),
            process_halt: false,
            thread_exit: false,
            fatal: None,
            globals: None,
        };
        let mut store = Store::new(&self.engine, data);
        store.set_fuel(u64::MAX).map_err(|e| RunError::Wasm(e.to_string()))?;
        store
            .fuel_async_yield_interval(Some(self.quantum_fuel))
            .map_err(|e| RunError::Wasm(e.to_string()))?;

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

        Ok(WasmtimeThread { thid, future, signal, priority })
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
                        if qdiff_regtrace().is_none() && !block_hist_enabled() {
                            let thid = caller.data().thid;
                            let r: [u32; 13] = std::array::from_fn(|i| get_reg(&mut caller, i));
                            eprintln!(
                                "[trace] t{thid} f_{sel:x}  r0={:#010x} r1={:#010x} r2={:#010x} \
                                 r3={:#010x} r4={:#010x} r5={:#010x} r6={:#010x} r7={:#010x} \
                                 r8={:#010x} r9={:#010x} r10={:#010x} r11={:#010x} r12={:#010x} lr={:#010x}",
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
    let watch = qdiff_regtrace_watch();
    if !watch.is_empty() {
        let base = caller.data().base;
        let shared = caller.data().shared_mem.clone();
        let data = shared.data();
        // SAFETY: as in `qdiff_dump_snapshot` - `UnsafeCell<u8>` is repr(transparent)
        // over `u8`, and no other fiber runs while this svc handler executes.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
        for &addr in watch {
            let off = addr.wrapping_sub(base) as usize;
            match bytes.get(off..off + 4) {
                Some(w) => line.push_str(&format!(
                    " m{addr:08x}={:08x}",
                    u32::from_le_bytes([w[0], w[1], w[2], w[3]])
                )),
                None => line.push_str(&format!(" m{addr:08x}=oob")),
            }
        }
    }
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
                        let mut view = SharedView::new(&shared);
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
                            caller.data().signal.lock().unwrap().stop = Stop::Quantum;
                            YieldNow(false).await;
                        }
                        SvcOutcome::Block => {
                            caller.data().signal.lock().unwrap().stop = Stop::Blocked;
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
                            caller.data().signal.lock().unwrap().stop = Stop::Flip;
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
}

impl SharedView {
    fn new(mem: &SharedMemory) -> SharedView {
        let data = mem.data();
        SharedView { ptr: data.as_ptr() as *mut u8, len: data.len() }
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
    s
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
