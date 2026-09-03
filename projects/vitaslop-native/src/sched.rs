//! The cooperative scheduler: run the guest on a wasmtime async fiber so it can
//! *yield* at a blocking primitive and be resumed later, one frame at a time,
//! from a synchronous host loop (the desktop window's redraw). This is the
//! single-worker cooperative model from the runtime README - the guest is a
//! coroutine the host schedules - realized on native via wasmtime's fiber-based
//! async plus fuel-based preemption for the quantum.
//!
//! # Why a fiber and not a second thread
//! The guest's `_start` runs its whole render loop internally; there is no
//! per-frame entry to call. To hand control back to the window each frame without
//! a second OS thread, the guest's per-frame display flip (`SvcOutcome::Flip`)
//! suspends the fiber. wasmtime requires the async Store's data to be `Send` (a
//! fiber may resume on any thread), which is why `World` is `Send`; in practice
//! everything here runs on one thread.
//!
//! # Why the store lives inside the future
//! wasmtime's `call_async` future borrows `&mut Store`, so it cannot be persisted
//! next to the store it borrows. Instead the future *owns* the store, and the one
//! thing that must cross the boundary each frame - the input going in and the
//! finished scene coming out - travels through a small shared cell both the host
//! import handler (inside the store) and the window loop (outside) hold.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use vitaslop_runtime::capture::Scene;
use vitaslop_runtime::{CtrlFrame, ImportDispatch, SliceMemory, SvcOutcome, VitaEnv, World};
use vitaslop_transpiler::abi;
use vitaslop_transpiler::{self as transpiler};
use wasmtime::{Caller, Config, Engine, Instance, Linker, Module, Store, Val};

use crate::RunError;

/// The quantum: fuel units the guest may retire before the fiber yields back to
/// the scheduler even without hitting a blocking call. Deterministic (a retired-
/// instruction count), so single-worker scheduling stays reproducible. Large
/// enough that a normal frame reaches its display flip in one slice; small enough
/// that a runaway guest cannot monopolize the worker (see `MAX_QUANTA_PER_FRAME`).
const QUANTUM_FUEL: u64 = 1_000_000;

/// Cap on quanta the scheduler will resume within a single `run_frame` before
/// giving control back to the host even if the guest never reached a flip. Keeps
/// the window responsive against a runaway guest; a normal frame returns a
/// presentable scene long before this.
const MAX_QUANTA_PER_FRAME: u32 = 4096;

/// The cell shared between the guest (inside the fiber's store) and the host
/// window loop (outside it). Input flows in, the finished scene flows out.
struct Shared {
    /// Latest controller frame from the host, read by the guest's `poll_ctrl`.
    input: CtrlFrame,
    /// The scene captured at the most recent display flip, moved out by the
    /// scheduler when it hands a frame to the window.
    latest_scene: Option<Scene>,
    /// Set by the flip handler to signal "a frame is ready"; cleared by the
    /// scheduler when it consumes it.
    present_pending: bool,
}

/// The store data for the async run: the Vita host environment (owned directly,
/// not behind an `Rc`, so the store is `Send`) plus the shared cell the flip
/// handler writes the finished scene into, and the halt flag.
struct SchedState {
    env: VitaEnv,
    shared: Arc<Mutex<Shared>>,
    halted: bool,
    base: u32,
}

/// A `World` that reads live controller input from the shared cell each poll, and
/// runs a virtual per-frame monotonic clock (so guest timing stays frame-based
/// and reproducible while the *input* is live). All non-determinism still enters
/// through this one seam.
struct LiveWorld {
    shared: Arc<Mutex<Shared>>,
    polls: u32,
}

impl World for LiveWorld {
    // A virtual 60 Hz clock advanced one frame per controller poll (the cube polls
    // once per frame). Same convention as the scripted worlds - poll_ctrl advances
    // the frame, the clocks read it - so a no-input live run is bit-identical to
    // the scripted run-to-completion, keeping guest time a pure function of
    // progress rather than wall-clock.
    fn monotonic_us(&mut self) -> u64 {
        self.polls as u64 * 16_666
    }
    fn wall_us(&mut self) -> u64 {
        1_500_000_000_000_000 + self.polls as u64 * 16_666
    }
    fn poll_ctrl(&mut self, _port: u32) -> CtrlFrame {
        self.polls += 1;
        self.shared.lock().unwrap().input
    }
    fn fill_random(&mut self, buf: &mut [u8]) {
        buf.fill(0);
    }
}

/// What stopped the guest this `run_frame`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameStop {
    /// A frame reached its display flip and is ready to present ([`Scheduler::current_scene`]).
    Present,
    /// The guest ran the whole quantum budget without a flip (a runaway or a very
    /// heavy frame). The host should present the previous scene and call again.
    Preempted,
    /// The guest returned or exited (e.g. it tore down after START). No more frames.
    Finished,
}

/// A guest running cooperatively on a fiber, stepped one frame per `run_frame`.
pub struct Scheduler {
    /// The in-flight guest execution. Owns its store; suspends at each flip.
    future: Pin<Box<dyn Future<Output = Result<(), String>> + Send>>,
    shared: Arc<Mutex<Shared>>,
    /// The scene handed out by the last `Present`, kept so the window can borrow
    /// it without locking the shared cell.
    current: Option<Scene>,
    /// Quanta the scheduler resumes within one `run_frame` before preempting.
    max_quanta: u32,
    finished: bool,
    /// Set if the guest run ended in an error rather than a clean exit.
    error: Option<String>,
}

impl Scheduler {
    /// Transpile and instantiate `code` for cooperative execution, ready to be
    /// stepped from `entry`. Mirrors [`Vm::new`](crate::Vm::new) but on an async
    /// engine: the returned scheduler holds the guest suspended before its first
    /// instruction; the first `run_frame` runs init through the first flip. Uses
    /// the default quantum ([`QUANTUM_FUEL`] / [`MAX_QUANTA_PER_FRAME`]).
    pub fn new(
        code: &[u8],
        base: u32,
        thumb: bool,
        entries: &[u32],
        externs: &[transpiler::Extern],
        mem_bytes: u32,
        imports: Vec<(u32, u32)>,
    ) -> Result<Scheduler, RunError> {
        Scheduler::with_quantum(
            QUANTUM_FUEL,
            MAX_QUANTA_PER_FRAME,
            code,
            base,
            thumb,
            entries,
            externs,
            mem_bytes,
            imports,
        )
    }

    /// Like [`new`](Scheduler::new) but with an explicit quantum: `quantum_fuel`
    /// retired-instruction units per preemptive yield, and `max_quanta` resumes
    /// per `run_frame` before returning [`FrameStop::Preempted`]. Tuning knob (and
    /// the seam tests use to force preemption on the CPU-light cube).
    #[allow(clippy::too_many_arguments)]
    pub fn with_quantum(
        quantum_fuel: u64,
        max_quanta: u32,
        code: &[u8],
        base: u32,
        thumb: bool,
        entries: &[u32],
        externs: &[transpiler::Extern],
        mem_bytes: u32,
        imports: Vec<(u32, u32)>,
    ) -> Result<Scheduler, RunError> {
        let artifact = transpiler::transpile(&transpiler::Program {
            code,
            base,
            thumb,
            entries,
            arm_entries: &[],
            externs,
            redirects: &[],
            // This entry point builds from a raw code image with no NID import table,
            // so nothing here is known to be inlinable.
            inline_imports: &[],
            noreturn_svc: &[],
            mem_bytes,
            // Vita modules take function addresses (thread entries, callbacks).
            discover_code_pointers: true,
            // The single-worker scheduler is one instance with its own memory.
            import_memory: false,
        })?;
        wasmparser::validate(&artifact.wasm)
            .map_err(|e| RunError::Wasm(format!("invalid module: {e}")))?;

        // Async (fibers) is always available in wasmtime 46; we only need fuel
        // for the quantum. The store is driven on a fiber by call_async below.
        let engine = Engine::new(Config::new().consume_fuel(true))
            .map_err(|e| RunError::Wasm(e.to_string()))?;
        let module = Module::from_binary(&engine, &artifact.wasm)?;

        let shared = Arc::new(Mutex::new(Shared {
            input: CtrlFrame::default(),
            latest_scene: None,
            present_pending: false,
        }));
        let world = Box::new(LiveWorld { shared: shared.clone(), polls: 0 });
        let env = VitaEnv::new(imports, base, mem_bytes, world);

        let mut store = Store::new(
            &engine,
            SchedState { env, shared: shared.clone(), halted: false, base },
        );
        // Unlimited fuel, but yield to us every QUANTUM_FUEL units so the guest
        // cannot monopolize the worker between blocking calls (the quantum).
        store.set_fuel(u64::MAX).map_err(|e| RunError::Wasm(e.to_string()))?;
        store
            .fuel_async_yield_interval(Some(quantum_fuel))
            .map_err(|e| RunError::Wasm(e.to_string()))?;

        let mut linker = Linker::new(&engine);
        bind_svc(&mut linker)?;
        bind_import(&mut linker)?;
        bind_dispatch_miss(&mut linker)?;
        // No start section in the guest module, so instantiation completes without
        // suspending; drive it to completion synchronously.
        let instance = pollster::block_on(linker.instantiate_async(&mut store, &module))?;

        // Seed the image and stack pointer, exactly like the sync Vm.
        let memory = instance
            .get_memory(&mut store, abi::MEMORY_EXPORT)
            .expect("module exports memory");
        memory.write(&mut store, 0, code)?;
        set_reg(&mut store, &instance, abi::SP, base.wrapping_add(mem_bytes));

        let entry = entries[0] & !1;
        let entry_name = abi::func_export(entry);
        let future = Box::pin(async move {
            let func = instance
                .get_typed_func::<(), ()>(&mut store, &entry_name)
                .map_err(|e| e.to_string())?;
            match func.call_async(&mut store, ()).await {
                Ok(()) => Ok(()),
                Err(e) => {
                    if store.data().halted {
                        Ok(()) // Clean exit: a host handler unwound the stack.
                    } else {
                        Err(e.to_string())
                    }
                }
            }
        });

        Ok(Scheduler { future, shared, current: None, max_quanta, finished: false, error: None })
    }

    /// Set the controller input the guest will read on its next `poll_ctrl`. Call
    /// before `run_frame` each host frame.
    pub fn set_input(&mut self, ctrl: CtrlFrame) {
        self.shared.lock().unwrap().input = ctrl;
    }

    /// Resume the guest until it reaches its next display flip (a presentable
    /// frame), exits, or exhausts the per-frame quantum budget.
    pub fn run_frame(&mut self) -> FrameStop {
        if self.finished {
            return FrameStop::Finished;
        }
        // A synchronous stepping executor: our only awaits (the flip yield and the
        // fuel quantum) are always resumable on the next poll, so a no-op waker
        // and a re-poll loop drive the fiber to its next stop.
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        for _ in 0..self.max_quanta {
            match self.future.as_mut().poll(&mut cx) {
                Poll::Ready(res) => {
                    self.finished = true;
                    if let Err(e) = res {
                        self.error = Some(e);
                    }
                    return FrameStop::Finished;
                }
                Poll::Pending => {
                    let mut s = self.shared.lock().unwrap();
                    if s.present_pending {
                        s.present_pending = false;
                        self.current = s.latest_scene.take();
                        return FrameStop::Present;
                    }
                    // Otherwise it was a fuel-quantum yield mid-frame: resume.
                }
            }
        }
        FrameStop::Preempted
    }

    /// The scene from the most recent `Present`, if any.
    pub fn current_scene(&self) -> Option<&Scene> {
        self.current.as_ref()
    }

    /// The error that ended the run, if it ended in one rather than a clean exit.
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

/// Bind `env.svc` as an async import. The Vita path never traps `svc` (NID stubs
/// route through `env.import`), but the module may still declare the import, so it
/// must resolve; it is a no-op that keeps running.
fn bind_svc(linker: &mut Linker<SchedState>) -> Result<(), RunError> {
    linker
        .func_wrap_async(
            abi::IMPORT_MODULE,
            abi::SVC_NAME,
            |_caller: Caller<'_, SchedState>, (_selector,): (i32,)| {
                Box::new(async { Ok(()) })
            },
        )
        .map_err(|e| RunError::Wasm(e.to_string()))?;
    Ok(())
}

/// Bind `env.dispatch_miss`: an indirect call resolving to no translated function
/// unwinds the fiber with the faulting `(target, caller)` addresses, so an unmapped
/// target is a clear report rather than an opaque `unreachable` trap.
fn bind_dispatch_miss(linker: &mut Linker<SchedState>) -> Result<(), RunError> {
    linker
        .func_wrap_async(
            abi::IMPORT_MODULE,
            abi::DISPATCH_MISS_NAME,
            |_caller: Caller<'_, SchedState>, (target, caller): (i32, i32)| {
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

/// Bind the `env.import` NID trap as an async import: dispatch to the Vita host,
/// then act on the outcome - halt unwinds the fiber, yield suspends it after
/// recording the finished scene, continue returns normally.
fn bind_import(linker: &mut Linker<SchedState>) -> Result<(), RunError> {
    linker
        .func_wrap_async(
            abi::IMPORT_MODULE,
            abi::IMPORT_NAME,
            |mut caller: Caller<'_, SchedState>, (selector,): (i32,)| {
                Box::new(async move {
                    let mut regs = [0u32; abi::REG_COUNT];
                    for (i, r) in regs.iter_mut().enumerate() {
                        *r = get_reg(&mut caller, i);
                    }
                    let mut vfp = [0u32; vitaslop_runtime::VFP_ARG_COUNT];
                    for (i, s) in vfp.iter_mut().enumerate() {
                        *s = get_vfp(&mut caller, i);
                    }
                    let mem = caller
                        .get_export(abi::MEMORY_EXPORT)
                        .and_then(|e| e.into_memory())
                        .expect("module exports memory");
                    let outcome = {
                        let (bytes, host) = mem.data_and_store_mut(&mut caller);
                        let base = host.base;
                        let mut gm = SliceMemory(bytes);
                        host.env.dispatch(selector as u32, &mut regs, &mut vfp, &mut gm, base)
                    };
                    for (i, &v) in regs.iter().enumerate() {
                        set_reg_caller(&mut caller, i, v);
                    }
                    for (i, &v) in vfp.iter().enumerate() {
                        set_vfp_caller(&mut caller, i, v);
                    }
                    match outcome {
                        // This single-worker scheduler runs one thread of control,
                        // so a worker ending (`ThreadExit`) is the process ending,
                        // same as `Halt`. Preemptive multithreading lives in the
                        // `ThreadedScheduler`.
                        SvcOutcome::Halt | SvcOutcome::ThreadExit => {
                            caller.data_mut().halted = true;
                            return Err(wasmtime::Error::msg("guest halted"));
                        }
                        SvcOutcome::Flip => {
                            // Record the just-finished frame, flag it, and suspend
                            // the fiber so the scheduler can present and refresh
                            // input before we resume into the next frame.
                            let scene = caller.data().env.state.capture.scenes.last().cloned();
                            {
                                let host = caller.data();
                                let mut s = host.shared.lock().unwrap();
                                s.latest_scene = scene;
                                s.present_pending = true;
                            }
                            YieldNow(false).await;
                        }
                        // A single-worker run has no other thread to switch to, so
                        // its uncontended waits never block and a priority reschedule
                        // is a no-op; if one occurred, treat it as Continue rather than
                        // deadlock.
                        SvcOutcome::Continue | SvcOutcome::Block | SvcOutcome::Reschedule => {}
                        // Unfaithful call (e.g. unimplemented NID): unwind WITHOUT
                        // setting `halted`, so the guest-call future surfaces it as a
                        // run error (a loud stop) rather than a clean exit.
                        SvcOutcome::Fatal(msg) => {
                            return Err(wasmtime::Error::msg(msg));
                        }
                    }
                    Ok(())
                })
            },
        )
        .map_err(|e| RunError::Wasm(e.to_string()))?;
    // The non-suspending trap is the same call here - see `abi::IMPORT_FAST_NAME`.
    linker
        .alias(abi::IMPORT_MODULE, abi::IMPORT_NAME, abi::IMPORT_MODULE, abi::IMPORT_FAST_NAME)
        .map_err(|e| RunError::Wasm(e.to_string()))?;
    Ok(())
}

/// A future that yields once (suspends the fiber) then completes. Awaiting it
/// inside a host import returns control to the scheduler's poll loop exactly once.
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

/// Read guest register `i` through a `Caller` (during a host import).
fn get_reg<T>(caller: &mut Caller<'_, T>, i: usize) -> u32 {
    caller
        .get_export(&abi::reg_export(i))
        .and_then(|e| e.into_global())
        .expect("module exports registers")
        .get(&mut *caller)
        .i32()
        .expect("register global is i32") as u32
}

/// Write guest register `i` through a `Caller` (during a host import).
fn set_reg_caller<T>(caller: &mut Caller<'_, T>, i: usize, v: u32) {
    caller
        .get_export(&abi::reg_export(i))
        .and_then(|e| e.into_global())
        .expect("module exports registers")
        .set(&mut *caller, Val::I32(v as i32))
        .expect("register global is mutable i32");
}

/// Read VFP single-precision register s`i` (raw bits) through a `Caller`.
fn get_vfp<T>(caller: &mut Caller<'_, T>, i: usize) -> u32 {
    caller
        .get_export(&abi::vfp_s_export(i as u8))
        .and_then(|e| e.into_global())
        .expect("module exports vfp registers")
        .get(&mut *caller)
        .i32()
        .expect("vfp global is i32") as u32
}

/// Write VFP single-precision register s`i` (raw bits) through a `Caller`.
fn set_vfp_caller<T>(caller: &mut Caller<'_, T>, i: usize, v: u32) {
    caller
        .get_export(&abi::vfp_s_export(i as u8))
        .and_then(|e| e.into_global())
        .expect("module exports vfp registers")
        .set(&mut *caller, Val::I32(v as i32))
        .expect("vfp global is mutable i32");
}

/// Write guest register `i` through the owned store (during setup).
fn set_reg<T>(store: &mut Store<T>, instance: &Instance, i: usize, v: u32) {
    instance
        .get_global(&mut *store, &abi::reg_export(i))
        .expect("module exports registers")
        .set(&mut *store, Val::I32(v as i32))
        .expect("register global is mutable i32");
}
