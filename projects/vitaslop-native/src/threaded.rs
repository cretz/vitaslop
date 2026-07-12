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

use vitaslop_runtime::{ImportDispatch, SvcOutcome, VFP_ARG_COUNT};
use vitaslop_transpiler::abi;
use vitaslop_transpiler::{self as transpiler};
use wasmtime::{Caller, Config, Engine, Instance, Linker, Module, SharedMemory, Store, Val};

use crate::RunError;

/// Fuel a thread may retire before the fiber yields back to the scheduler even
/// without a host call. A deterministic (retired-instruction) quantum so the
/// interleaving is reproducible; large enough that a normal run of guest code
/// between host calls completes in one slice.
const DEFAULT_QUANTUM_FUEL: u64 = 1_000_000;

/// Backstop on scheduler rounds in [`ThreadedScheduler::run`], so a runaway or
/// live-locking guest cannot spin forever. A round is one fiber poll.
const MAX_ROUNDS: u64 = 100_000_000;

/// Why a fiber poll returned `Pending` (it suspended rather than finishing). Set
/// by the host-call closure just before it awaits, read by the scheduler through
/// the thread's shared signal cell.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stop {
    /// A fuel-quantum preemption: no host call blocked, the thread is still
    /// runnable and simply used up its slice.
    Quantum,
    /// The thread hit a blocking primitive and must be parked until woken.
    Blocked,
    /// The thread reached a frame boundary (display flip).
    Yielded,
}

/// The one-word channel from a thread's host-call closure (running inside its
/// fiber, hence inside its store) out to the scheduler's poll loop, which cannot
/// otherwise see into the borrowed store. Only the `Stop` reason needs to cross;
/// exit codes and halt/exit kinds ride the fiber's own return value.
struct Signal {
    stop: Stop,
}

/// How a thread's fiber finished (the `Future`'s output).
enum FiberEnd {
    /// The guest entry returned normally; the value is its r0 (the thread's exit
    /// code).
    Returned(u32),
    /// The thread called `sceKernelExitThread` (or equivalent): it ends, but the
    /// process continues. The value is its r0.
    ThreadExit(u32),
    /// The thread called `sceKernelExitProcess`: the whole run stops. The value is
    /// the process exit code (r0).
    ProcessHalt(u32),
    /// The guest trapped for a reason that was not a clean halt/exit.
    Error(String),
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

/// One scheduled guest thread.
struct ThreadCtl {
    thid: i32,
    /// The in-flight guest call; owns its store, suspends at each switch point.
    future: Pin<Box<dyn Future<Output = FiberEnd> + Send>>,
    signal: Arc<Mutex<Signal>>,
    state: ThreadState,
}

/// A scheduled thread's lifecycle state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ThreadState {
    /// On the run queue; the scheduler may poll it.
    Runnable,
    /// Parked at a blocking primitive; not polled until a wake makes it runnable.
    Blocked,
    /// Done. The value is its exit code.
    Finished(u32),
}

/// The verdict of a [`ThreadedScheduler::run`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunReport {
    /// Every thread finished cleanly (or the process exited). The value is the
    /// process exit code (the halting thread's r0, or the main thread's).
    Finished(u32),
    /// No thread is runnable but some remain blocked: a deadlock (e.g. a wait with
    /// no possible signaller). The value lists the still-blocked thread ids.
    Deadlock(Vec<i32>),
    /// A thread trapped. The run cannot continue.
    Error(String),
    /// The round budget was exhausted (a live-lock backstop).
    RoundLimit,
}

/// A preemptive multi-thread guest run: the transpiled module, one shared memory,
/// a shared host, and the live thread table.
pub struct ThreadedScheduler<H: ImportDispatch + Send + 'static> {
    engine: Engine,
    module: Module,
    shared_mem: SharedMemory,
    host: Arc<Mutex<H>>,
    base: u32,
    quantum_fuel: u64,
    threads: Vec<ThreadCtl>,
    /// Round-robin cursor: the index after the last thread polled.
    cursor: usize,
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
            externs,
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

        // One shared memory, sized exactly as the transpiler declared, imported
        // into every thread instance. Seed the image into it once.
        let pages = (mem_bytes as u64).div_ceil(abi::PAGE_SIZE as u64).max(1) as u32;
        let mem_ty = wasmtime::MemoryType::shared(pages, pages);
        let shared_mem =
            SharedMemory::new(&engine, mem_ty).map_err(|e| RunError::Wasm(e.to_string()))?;
        write_shared(&shared_mem, 0, code);

        let host = Arc::new(Mutex::new(host));
        let mut sched = ThreadedScheduler {
            engine,
            module,
            shared_mem,
            host,
            base,
            quantum_fuel,
            threads: Vec::new(),
            cursor: 0,
        };

        // The main thread: sp at the top of the region, no entry args, its thid is
        // whatever the host reports for the main thread (0 by convention here; the
        // host maps it as it likes).
        let main = sched.instantiate_thread(0, entry & !1, 0, 0, base.wrapping_add(mem_bytes))?;
        sched.threads.push(main);
        Ok(sched)
    }

    /// Borrow the shared host (e.g. to read captured output after the run).
    pub fn host(&self) -> std::sync::MutexGuard<'_, H> {
        self.host.lock().unwrap()
    }

    /// Run cooperatively until the process halts, every thread finishes, or the
    /// run deadlocks / errors. Returns the verdict.
    pub fn run(&mut self) -> RunReport {
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut rounds = 0u64;
        loop {
            if rounds >= MAX_ROUNDS {
                return RunReport::RoundLimit;
            }
            rounds += 1;

            let Some(idx) = self.next_runnable() else {
                // Nobody can run. Either all finished, or the rest are blocked.
                let blocked: Vec<i32> = self
                    .threads
                    .iter()
                    .filter(|t| t.state == ThreadState::Blocked)
                    .map(|t| t.thid)
                    .collect();
                if blocked.is_empty() {
                    return RunReport::Finished(self.main_exit_code());
                }
                return RunReport::Deadlock(blocked);
            };
            self.cursor = idx + 1;

            // Fresh reason each poll; the closure overwrites it if it awaits.
            self.threads[idx].signal.lock().unwrap().stop = Stop::Quantum;
            match self.threads[idx].future.as_mut().poll(&mut cx) {
                Poll::Ready(end) => {
                    if let Some(report) = self.on_finish(idx, end) {
                        return report;
                    }
                }
                Poll::Pending => {
                    let stop = self.threads[idx].signal.lock().unwrap().stop;
                    match stop {
                        Stop::Blocked => self.threads[idx].state = ThreadState::Blocked,
                        // A frame boundary or a fuel slice: still runnable.
                        Stop::Yielded | Stop::Quantum => {}
                    }
                }
            }

            // A host call in this poll may have asked to start threads or woken
            // parked ones; act on both before the next round.
            self.drain_spawns_and_wakes();
        }
    }

    /// Handle a finished fiber; returns `Some(report)` if the whole run must stop.
    fn on_finish(&mut self, idx: usize, end: FiberEnd) -> Option<RunReport> {
        let thid = self.threads[idx].thid;
        match end {
            FiberEnd::Returned(code) | FiberEnd::ThreadExit(code) => {
                self.threads[idx].state = ThreadState::Finished(code);
                // Tell the host this thread ended, so any sibling waiting on it can
                // be woken (the wake is drained right after, by the caller).
                self.host.lock().unwrap().set_thread_exit(thid, code);
                None
            }
            FiberEnd::ProcessHalt(code) => {
                self.threads[idx].state = ThreadState::Finished(code);
                // The process is over: mark every other live thread finished too.
                for t in self.threads.iter_mut() {
                    if let ThreadState::Runnable | ThreadState::Blocked = t.state {
                        t.state = ThreadState::Finished(code);
                    }
                }
                Some(RunReport::Finished(code))
            }
            FiberEnd::Error(e) => Some(RunReport::Error(format!("thread {thid:#x}: {e}"))),
        }
    }

    /// Start any threads the last host call requested, and wake any it released.
    fn drain_spawns_and_wakes(&mut self) {
        let (spawns, wakes) = {
            let mut host = self.host.lock().unwrap();
            (host.take_spawns(), host.take_wakes())
        };
        for sp in spawns {
            match self.instantiate_thread(sp.thid, sp.entry, sp.arg_len, sp.arg_ptr, sp.stack_top) {
                Ok(ctl) => self.threads.push(ctl),
                // A spawn whose entry was not transpiled: record it as finished
                // with code 0 so a later join does not hang.
                Err(_) => self.host.lock().unwrap().set_thread_exit(sp.thid, 0),
            }
        }
        for thid in wakes {
            if let Some(t) = self
                .threads
                .iter_mut()
                .find(|t| t.thid == thid && t.state == ThreadState::Blocked)
            {
                t.state = ThreadState::Runnable;
            }
        }
    }

    /// The next runnable thread index in round-robin order from the cursor, or
    /// None if none is runnable.
    fn next_runnable(&self) -> Option<usize> {
        let n = self.threads.len();
        (0..n)
            .map(|k| (self.cursor + k) % n)
            .find(|&i| self.threads[i].state == ThreadState::Runnable)
    }

    /// The process exit code: the main thread's finished code if it has one, else 0.
    fn main_exit_code(&self) -> u32 {
        self.threads
            .first()
            .and_then(|t| match t.state {
                ThreadState::Finished(c) => Some(c),
                _ => None,
            })
            .unwrap_or(0)
    }

    /// Build one thread: a fresh store+instance importing the shared memory, seeded
    /// registers (r0/r1 = entry args, sp), and its in-flight `call_async` future.
    fn instantiate_thread(
        &self,
        thid: i32,
        entry: u32,
        r0: u32,
        r1: u32,
        sp: u32,
    ) -> Result<ThreadCtl, RunError> {
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
        linker
            .define(&store, abi::IMPORT_MODULE, abi::MEMORY_EXPORT, self.shared_mem.clone())
            .map_err(|e| RunError::Wasm(e.to_string()))?;

        // No start section, so instantiation completes without suspending.
        let instance = pollster::block_on(linker.instantiate_async(&mut store, &self.module))?;

        // Seed the entry arguments and stack pointer for this thread.
        set_reg_store(&mut store, &instance, 0, r0);
        set_reg_store(&mut store, &instance, 1, r1);
        set_reg_store(&mut store, &instance, abi::SP, sp);

        let entry_name = abi::func_export(entry);
        let future = Box::pin(async move {
            let func = match instance.get_typed_func::<(), ()>(&mut store, &entry_name) {
                Ok(f) => f,
                // The entry was not a transpiled function; nothing to run.
                Err(_) => return FiberEnd::Returned(0),
            };
            let call_res = func.call_async(&mut store, ()).await;
            let r0 = get_reg_store(&mut store, &instance, 0);
            match call_res {
                Ok(()) => FiberEnd::Returned(r0),
                Err(e) => {
                    let d = store.data();
                    if d.process_halt {
                        FiberEnd::ProcessHalt(r0)
                    } else if d.thread_exit {
                        FiberEnd::ThreadExit(r0)
                    } else {
                        FiberEnd::Error(trap_detail(&e))
                    }
                }
            }
        });

        Ok(ThreadCtl { thid, future, signal, state: ThreadState::Runnable })
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

                    match outcome {
                        SvcOutcome::Continue => {}
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
        // SAFETY: called only during setup, before any fiber runs.
        unsafe {
            *data[off + i].get() = b;
        }
    }
}

/// A concise trap description (kind + message), matching the sync `Vm`'s detail.
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
