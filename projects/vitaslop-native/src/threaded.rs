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
//! thread wakes it; [`Yield`](SvcOutcome::Yield) is a frame boundary;
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
}

impl<H: ImportDispatch + Send + 'static> GuestEngine for WasmtimeEngine<H> {
    type Thread = WasmtimeThread;

    fn spawn(&mut self, reentry: &Reentry) -> Result<WasmtimeThread, ()> {
        self.instantiate_thread(
            reentry.thid,
            reentry.entry,
            reentry.arg_len,
            reentry.arg_ptr,
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
        let engine = WasmtimeEngine { engine, module, shared_mem, host: host.clone(), base, quantum_fuel };

        // The main thread: sp near the top of the region (with startup headroom), no
        // entry args, its thid is whatever the host reports for the main thread (0 by
        // convention here; the host maps it as it likes).
        let main = engine.instantiate_thread(
            0,
            entry & !1,
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

        let host = Arc::new(Mutex::new(host));
        let engine = WasmtimeEngine {
            engine,
            module,
            shared_mem,
            host: host.clone(),
            base: linked.base,
            quantum_fuel,
        };

        // The main thread runs every module_start in load order, then (as the last
        // entry) the eboot's - which is where a render loop lives.
        let sp = main_stack_top(linked.base, linked.mem_bytes);
        let main = engine.instantiate_thread_seq(
            0,
            linked.module_inits.clone(),
            0,
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

    /// Display frame boundaries (flips) observed so far. A live windowed front-end
    /// steps one frame per redraw via `run_frames(frames() + 1, ..)`.
    pub fn frames(&self) -> u64 {
        self.inner.frames()
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

    /// Build one thread from a single entry (a spawned worker).
    fn instantiate_thread(
        &self,
        thid: i32,
        entry: u32,
        r0: u32,
        r1: u32,
        sp: u32,
        priority: i32,
    ) -> Result<WasmtimeThread, RunError> {
        self.instantiate_thread_seq(thid, vec![entry], r0, r1, sp, priority)
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

        let future = Box::pin(async move {
            let mut last_r0 = 0u32;
            let last = entries.len().saturating_sub(1);
            for (i, &entry) in entries.iter().enumerate() {
                // Each entry starts with a fresh stack; only the first carries args.
                set_reg_store(&mut store, &instance, abi::SP, sp);
                set_reg_store(&mut store, &instance, 0, if i == 0 { r0 } else { 0 });
                set_reg_store(&mut store, &instance, 1, if i == 0 { r1 } else { 0 });
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

/// Bind `env.svc` (unused on the Vita NID path, but the module may declare it).
fn bind_svc<H: ImportDispatch + Send + 'static>(
    linker: &mut Linker<ThreadData<H>>,
) -> Result<(), RunError> {
    linker
        .func_wrap_async(
            abi::IMPORT_MODULE,
            abi::SVC_NAME,
            |_caller: Caller<'_, ThreadData<H>>, (_selector,): (i32,)| Box::new(async { Ok(()) }),
        )
        .map_err(|e| RunError::Wasm(e.to_string()))?;
    Ok(())
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
                    let mut regs = [0u32; abi::REG_COUNT];
                    for (i, r) in regs.iter_mut().enumerate() {
                        *r = get_reg(&mut caller, i);
                    }
                    let mut vfp = [0u32; VFP_ARG_COUNT];
                    for (i, s) in vfp.iter_mut().enumerate() {
                        *s = get_vfp(&mut caller, i);
                    }

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

                    for (i, &v) in regs.iter().enumerate() {
                        set_reg_caller(&mut caller, i, v);
                    }
                    for (i, &v) in vfp.iter().enumerate() {
                        set_vfp_caller(&mut caller, i, v);
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
                        }
                        SvcOutcome::Yield => {
                            caller.data().signal.lock().unwrap().stop = Stop::Yielded;
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
    match e.downcast_ref::<wasmtime::Trap>() {
        Some(t) => format!("{t:?}: {e}"),
        None => e.to_string(),
    }
}

// --- register/vfp accessors (Caller during a call, Store during setup) --------

fn get_reg<T>(caller: &mut Caller<'_, T>, i: usize) -> u32 {
    caller
        .get_export(&abi::reg_export(i))
        .and_then(|e| e.into_global())
        .expect("module exports registers")
        .get(&mut *caller)
        .i32()
        .expect("register global is i32") as u32
}

fn set_reg_caller<T>(caller: &mut Caller<'_, T>, i: usize, v: u32) {
    caller
        .get_export(&abi::reg_export(i))
        .and_then(|e| e.into_global())
        .expect("module exports registers")
        .set(&mut *caller, Val::I32(v as i32))
        .expect("register global is mutable i32");
}

fn get_vfp<T>(caller: &mut Caller<'_, T>, i: usize) -> u32 {
    caller
        .get_export(&abi::vfp_s_export(i as u8))
        .and_then(|e| e.into_global())
        .expect("module exports vfp registers")
        .get(&mut *caller)
        .i32()
        .expect("vfp global is i32") as u32
}

fn set_vfp_caller<T>(caller: &mut Caller<'_, T>, i: usize, v: u32) {
    caller
        .get_export(&abi::vfp_s_export(i as u8))
        .and_then(|e| e.into_global())
        .expect("module exports vfp registers")
        .set(&mut *caller, Val::I32(v as i32))
        .expect("vfp global is mutable i32");
}

fn set_reg_store<T>(store: &mut Store<T>, instance: &Instance, i: usize, v: u32) {
    instance
        .get_global(&mut *store, &abi::reg_export(i))
        .expect("module exports registers")
        .set(&mut *store, Val::I32(v as i32))
        .expect("register global is mutable i32");
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
