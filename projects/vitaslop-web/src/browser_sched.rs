//! The browser preemptive scheduler: the JSPI implementation of the engine-agnostic
//! [`SchedCore`](vitaslop_runtime::sched::SchedCore). It is the browser twin of
//! `vitaslop-native`'s wasmtime `ThreadedScheduler`, and shares the exact scheduling
//! policy - only the "resume a guest thread to its next switch point" primitive is
//! reimplemented here on the browser's own WebAssembly engine.
//!
//! # One worker, instance-per-thread, one shared memory
//! Every guest thread is its own `WebAssembly.Instance` (its ARM register file lives
//! in wasm globals, which are per-instance, so each thread's registers are naturally
//! private), and all instances import ONE shared linear memory (the transpiler emits
//! `env.memory` when `import_memory` is set) - one guest address space, private
//! registers, exactly the native model. Everything runs on one thread: the single
//! owner of [`VitaEnv`], the scheduler, and all guest instances. Because the host
//! (`VitaEnv`) lives here too, a guest host call needs no cross-thread hop.
//!
//! # JSPI is how a mid-stack thread suspends
//! A guest thread blocks deep inside its wasm call stack (inside game logic that
//! called a blocking kernel primitive). To switch away we must suspend that stack.
//! The browser has no wasmtime fibers, so we use **JSPI**: `env.import` is a
//! `WebAssembly.Suspending` function and each thread's entry is called through
//! `WebAssembly.promising`. A host call that must block returns a *pending Promise*,
//! which suspends the guest stack and returns control to the async scheduler loop; a
//! host call that continues returns a plain value, which does NOT suspend (so the
//! common case stays cheap). Resuming a suspended thread resolves that Promise.
//!
//! Because a JSPI resume is inherently asynchronous (it unwinds to the event loop),
//! the run loop here is `async` and cannot reuse the synchronous
//! [`Scheduler`](vitaslop_runtime::sched::Scheduler) loop - but it composes the same
//! [`SchedCore`] helpers (priority pick, frame counting, spawn/wake drain,
//! deadlock/timed-wait, verdict), so the discipline is identical to native.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use js_sys::{Array, Function, Object, Promise, Reflect, Uint8Array, WebAssembly};
use vitaslop_runtime::sched::{
    FiberEnd, GuestEngine, IdleStep, RunReport, SchedCore, Stop, ThreadHandle, ThreadStep,
};
use vitaslop_runtime::{GuestMemory, ImportDispatch, Reentry, SvcOutcome, VitaEnv, VFP_ARG_COUNT};
use vitaslop_transpiler::abi;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

/// The shared host: a single `VitaEnv` behind an `Arc<Mutex>` (single-threaded here,
/// so the lock never contends - it just satisfies `SchedCore`'s bound, which mirrors
/// native's `Send` host).
type Host = Arc<Mutex<VitaEnv>>;

/// A `Uint8Array` view over the shared linear memory, rebased so guest address `A` is
/// byte `A - base`. Rebuilt per host call because a `memory.grow` would detach the
/// buffer (the guest never grows the fixed shared memory, but this stays correct).
struct SharedView {
    mem: WebAssembly::Memory,
}

impl GuestMemory for SharedView {
    fn len(&self) -> usize {
        Uint8Array::new(&self.mem.buffer()).length() as usize
    }
    fn read(&self, off: usize, buf: &mut [u8]) {
        Uint8Array::new(&self.mem.buffer())
            .subarray(off as u32, (off + buf.len()) as u32)
            .copy_to(buf);
    }
    fn write(&mut self, off: usize, bytes: &[u8]) {
        Uint8Array::new(&self.mem.buffer())
            .subarray(off as u32, (off + bytes.len()) as u32)
            .copy_from(bytes);
    }
}

/// A guest instance's mutable state the host reaches during a call: its 16 ARM
/// register globals, its VFP single-precision argument globals, and the shared memory.
struct ThreadRt {
    regs: Vec<WebAssembly::Global>,
    vfp: Vec<WebAssembly::Global>,
    mem: WebAssembly::Memory,
    base: u32,
}

impl ThreadRt {
    fn read_regs(&self) -> [u32; abi::REG_COUNT] {
        let mut r = [0u32; abi::REG_COUNT];
        for (i, g) in self.regs.iter().enumerate() {
            r[i] = g.value().as_f64().unwrap_or(0.0) as i64 as u32;
        }
        r
    }
    fn write_regs(&self, regs: &[u32; abi::REG_COUNT]) {
        for (i, g) in self.regs.iter().enumerate() {
            g.set_value(&JsValue::from_f64(regs[i] as f64));
        }
    }
    fn read_vfp(&self) -> [u32; VFP_ARG_COUNT] {
        let mut v = [0u32; VFP_ARG_COUNT];
        for (i, g) in self.vfp.iter().enumerate() {
            v[i] = g.value().as_f64().unwrap_or(0.0) as i64 as u32;
        }
        v
    }
    fn write_vfp(&self, vfp: &[u32; VFP_ARG_COUNT]) {
        for (i, g) in self.vfp.iter().enumerate() {
            g.set_value(&JsValue::from_f64(vfp[i] as f64));
        }
    }
    fn set_reg(&self, i: usize, v: u32) {
        self.regs[i].set_value(&JsValue::from_f64(v as f64));
    }
    fn read_reg(&self, i: usize) -> u32 {
        self.regs[i].value().as_f64().unwrap_or(0.0) as i64 as u32
    }
    fn view(&self) -> SharedView {
        SharedView { mem: self.mem.clone() }
    }
}

/// One guest thread on the browser engine: its own instance (its register file), the
/// `promising`-wrapped entries it runs in sequence, and the two one-slot channels JSPI
/// uses to hand control back and forth with the scheduler loop.
///
/// The main thread runs several entries in load order (a linked title's `module_init`s
/// then the eboot entry, each on the same instance with a fresh stack) - the browser
/// twin of native's `instantiate_thread_seq`. A spawned worker is just one entry.
pub struct BrowserThread {
    thid: i32,
    priority: i32,
    rt: Rc<ThreadRt>,
    /// One `promising`-wrapped entry per address, run in order.
    entries: Vec<Function>,
    /// Index of the entry currently running (or about to start).
    entry_idx: usize,
    /// Whether the current entry has been started (its `promising` call made). Cleared
    /// when advancing to the next entry.
    entry_started: bool,
    sp: u32,
    r0: u32,
    r1: u32,
    r2: u32,
    /// The resolver for the *current* resume's step Promise. The import closure (on a
    /// block/yield) or an entry's completion fills it with the encoded event; the
    /// scheduler awaits the matching Promise. Reset each loop turn.
    signal: Rc<RefCell<Option<Function>>>,
    /// The resolver of the Promise a suspended thread is parked on; the scheduler calls
    /// it to un-park (resume) the thread.
    cont: Rc<RefCell<Option<Function>>>,
    /// The shared host, so the un-park path can claim any return code owed to this
    /// thread (a timed wait that expired -> WAIT_TIMEOUT) and write it into r0 before
    /// the guest stack resumes. Native does this inside its import closure after the
    /// block await; the browser has no such re-entry, so it applies it here.
    host: Host,
    /// The import closure must outlive every call the instance can make into it.
    _import: Closure<dyn FnMut(i32) -> JsValue>,
}

impl ThreadHandle for BrowserThread {
    fn thid(&self) -> i32 {
        self.thid
    }
    fn priority(&self) -> i32 {
        self.priority
    }
}

/// One raw event a running guest entry reports over the JS step channel. Distinct
/// from [`ThreadStep`] because the main thread runs several entries in sequence (the
/// linked title's `module_init`s then the eboot entry): a bare [`Returned`](Ev::Returned)
/// or [`ThreadExit`](Ev::ThreadExit) on a non-final entry advances to the next entry
/// rather than ending the thread. [`resume`] folds these into a `ThreadStep`.
enum Ev {
    /// The entry suspended at a switch point (host call blocked / flipped / preempted).
    Suspend(Stop),
    /// The entry returned normally; the value is r0.
    Returned(u32),
    /// The entry called `sceKernelExitThread`; the value is r0.
    ThreadExit(u32),
    /// A host call halted the whole process; the value is r0.
    Halt(u32),
    /// The entry trapped.
    Error(String),
}

/// Encode an event as a small JS array `[tag, a, b]` for the step channel.
fn encode(ev: &Ev) -> JsValue {
    let f = JsValue::from_f64;
    match ev {
        Ev::Suspend(stop) => {
            let code = match stop {
                Stop::Quantum => 0.0,
                Stop::Blocked => 1.0,
                Stop::Flip => 2.0,
            };
            Array::of2(&f(0.0), &f(code)).into()
        }
        Ev::Returned(c) => Array::of2(&f(1.0), &f(*c as f64)).into(),
        Ev::ThreadExit(c) => Array::of2(&f(2.0), &f(*c as f64)).into(),
        Ev::Halt(c) => Array::of2(&f(3.0), &f(*c as f64)).into(),
        Ev::Error(m) => Array::of3(&f(4.0), &f(0.0), &JsValue::from_str(m)).into(),
    }
}

/// Decode an event the JS channel resolved with.
fn decode(val: &JsValue) -> Ev {
    let arr: Array = val.clone().into();
    let a = arr.get(1).as_f64().unwrap_or(0.0) as u32;
    match arr.get(0).as_f64().unwrap_or(4.0) as u32 {
        0 => Ev::Suspend(match a {
            1 => Stop::Blocked,
            2 => Stop::Flip,
            _ => Stop::Quantum,
        }),
        1 => Ev::Returned(a),
        2 => Ev::ThreadExit(a),
        3 => Ev::Halt(a),
        _ => Ev::Error(arr.get(2).as_string().unwrap_or_default()),
    }
}

/// Deliver `ev` to the current resume's awaiting Promise (one-shot: takes the
/// resolver so a later stray call is a no-op).
fn deliver(signal: &Rc<RefCell<Option<Function>>>, ev: &Ev) {
    if let Some(res) = signal.borrow_mut().take() {
        let _ = res.call1(&JsValue::UNDEFINED, &encode(ev));
    }
}

/// The browser execution engine: the transpiled module, the one shared memory every
/// instance imports, the shared host, and the JSPI primitives. Implements
/// [`GuestEngine`] so it stands up threads for [`SchedCore`].
pub struct BrowserEngine {
    module: WebAssembly::Module,
    shared_mem: WebAssembly::Memory,
    host: Host,
    base: u32,
    /// `WebAssembly.promising` (not in the wasm-bindgen bindings; fetched by name).
    promising: Function,
    /// `WebAssembly.Suspending` constructor.
    suspending: Function,
    /// Shared, non-suspending env stubs (`env.svc`, `env.dispatch_miss`), kept alive.
    _svc: Closure<dyn FnMut(i32)>,
    svc_fn: JsValue,
    _dispatch_miss: Closure<dyn FnMut(i32, i32)>,
    dispatch_miss_fn: JsValue,
    /// Linear-memory offset of the host-mirror block, when this build inlined any read
    /// of it (`vitaslop_transpiler::Artifact::mirror_off`). The scheduler refreshes it
    /// before every resume.
    mirror_off: Option<u64>,
}

impl BrowserEngine {
    /// Instantiate one guest thread: a fresh instance importing the shared memory and
    /// a `Suspending` host-call trap, with each of `entries` wrapped by `promising` to
    /// be run in sequence. `(r0, r1)` seed only the first entry.
    fn make_thread(
        &self,
        thid: i32,
        entries: &[u32],
        r0: u32,
        r1: u32,
        r2: u32,
        sp: u32,
        priority: i32,
    ) -> Result<BrowserThread, JsValue> {
        let signal: Rc<RefCell<Option<Function>>> = Rc::new(RefCell::new(None));
        let cont: Rc<RefCell<Option<Function>>> = Rc::new(RefCell::new(None));
        // The import closure needs the instance's globals, which only exist after
        // instantiation - the chicken-and-egg the runtime cell resolves (imports fire
        // only during execution, by when the cell is filled).
        let rt_cell: Rc<RefCell<Option<Rc<ThreadRt>>>> = Rc::new(RefCell::new(None));

        let import_closure = {
            let host = self.host.clone();
            let rt_cell = rt_cell.clone();
            let signal = signal.clone();
            let cont = cont.clone();
            Closure::wrap(Box::new(move |selector: i32| -> JsValue {
                let rt = rt_cell.borrow().as_ref().expect("rt set before first call").clone();
                let mut regs = rt.read_regs();
                let mut vfp = rt.read_vfp();
                let outcome = {
                    let mut mem = rt.view();
                    let mut host = host.lock().unwrap();
                    host.set_current_thread(thid);
                    host.dispatch(selector as u32, &mut regs, &mut vfp, &mut mem, rt.base)
                };
                rt.write_regs(&regs);
                rt.write_vfp(&vfp);
                match outcome {
                    // Plain return: the guest continues without suspending (the cheap,
                    // common path - most host calls just return a value).
                    SvcOutcome::Continue => JsValue::UNDEFINED,
                    // A switch point: tell the scheduler why we stopped, then return a
                    // pending Promise so the guest stack suspends until it resolves.
                    SvcOutcome::Reschedule => suspend(&signal, &cont, Stop::Quantum),
                    SvcOutcome::Block => suspend(&signal, &cont, Stop::Blocked),
                    SvcOutcome::Flip => suspend(&signal, &cont, Stop::Flip),
                    // The thread (or process) ends here: report the event and park on a
                    // never-resolving Promise (this stack is abandoned - on a thread exit
                    // the scheduler may still start the thread's next entry on a fresh
                    // stack; on a halt the run is over).
                    SvcOutcome::ThreadExit => {
                        deliver(&signal, &Ev::ThreadExit(regs[0]));
                        never()
                    }
                    SvcOutcome::Halt => {
                        deliver(&signal, &Ev::Halt(regs[0]));
                        never()
                    }
                    // Unfaithful call (e.g. unimplemented NID): stop the run loudly as
                    // an error rather than fake a success (which would desync the guest).
                    SvcOutcome::Fatal(msg) => {
                        deliver(&signal, &Ev::Error(msg));
                        never()
                    }
                }
            }) as Box<dyn FnMut(i32) -> JsValue>)
        };

        // env.import wrapped as Suspending; env.memory the shared memory; env.svc /
        // env.dispatch_miss the shared non-suspending stubs.
        let suspending_import = Reflect::construct(
            &self.suspending,
            &Array::of1(import_closure.as_ref().unchecked_ref()),
        )?;
        let env = Object::new();
        Reflect::set(&env, &JsValue::from_str(abi::MEMORY_EXPORT), &self.shared_mem)?;
        Reflect::set(&env, &JsValue::from_str(abi::IMPORT_NAME), &suspending_import)?;
        Reflect::set(&env, &JsValue::from_str(abi::SVC_NAME), &self.svc_fn)?;
        Reflect::set(&env, &JsValue::from_str(abi::DISPATCH_MISS_NAME), &self.dispatch_miss_fn)?;
        let imports = Object::new();
        Reflect::set(&imports, &JsValue::from_str(abi::IMPORT_MODULE), &env)?;

        let instance = WebAssembly::Instance::new(&self.module, &imports)?;
        let exports = instance.exports();

        // This thread's thread-local storage, mirroring native `instantiate_thread_seq`:
        // allocate the private block whose base is the thread pointer (TPIDRURO), copy
        // the template's initialized `.tdata` head into it (the `.tbss` tail is already
        // zero), and seed the instance's per-thread `tp` global before any entry runs
        // (a `MRC p15,0,Rt,c13,c0,3` reads it). No guest code is running yet, so the
        // shared-memory copy is safe. A title with no TLS template yields tp == 0 and
        // this is a no-op.
        let (tp, tls_src, tls_len) = self.host.lock().unwrap().thread_tls_base(thid);
        if tp != 0 {
            if tls_len != 0 {
                let view = Uint8Array::new(&self.shared_mem.buffer());
                let src = tls_src.wrapping_sub(self.base);
                let dst = tp.wrapping_sub(self.base);
                let head = view.subarray(src, src + tls_len).to_vec();
                view.subarray(dst, dst + tls_len).copy_from(&head);
            }
            let tp_global = Reflect::get(&exports, &JsValue::from_str(abi::TP_EXPORT))?
                .dyn_into::<WebAssembly::Global>()?;
            tp_global.set_value(&JsValue::from(tp));
        }

        let regs = read_globals(&exports, |i| abi::reg_export(i), abi::REG_COUNT)?;
        let vfp = read_globals(&exports, |i| abi::vfp_s_export(i as u8), VFP_ARG_COUNT)?;
        let rt = Rc::new(ThreadRt { regs, vfp, mem: self.shared_mem.clone(), base: self.base });
        *rt_cell.borrow_mut() = Some(rt.clone());

        let mut wrapped = Vec::with_capacity(entries.len());
        for &entry in entries {
            let entry_fn = Reflect::get(&exports, &JsValue::from_str(&abi::func_export(entry & !1)))?
                .dyn_into::<Function>()?;
            wrapped.push(
                self.promising
                    .call1(&JsValue::UNDEFINED, &entry_fn)?
                    .dyn_into::<Function>()?,
            );
        }

        Ok(BrowserThread {
            thid,
            priority,
            rt,
            entries: wrapped,
            entry_idx: 0,
            entry_started: false,
            sp,
            r0,
            r1,
            r2,
            signal,
            cont,
            host: self.host.clone(),
            _import: import_closure,
        })
    }
}

impl GuestEngine for BrowserEngine {
    type Thread = BrowserThread;

    fn spawn(&mut self, r: &Reentry) -> Result<BrowserThread, ()> {
        self.make_thread(r.thid, &[r.entry], r.arg_len, r.arg_ptr, r.r2, r.stack_top, r.priority)
            .map_err(|_| ())
    }

    fn write_mem(&mut self, addr: u32, bytes: &[u8]) {
        let off = addr.wrapping_sub(self.base) as usize;
        let view = Uint8Array::new(&self.shared_mem.buffer());
        if off + bytes.len() <= view.length() as usize {
            view.subarray(off as u32, (off + bytes.len()) as u32).copy_from(bytes);
        }
    }

    fn mirror_base(&self) -> Option<u32> {
        // The block sits above the guest region, so its guest address is the rebase
        // origin plus the offset - the same convention `write_mem` undoes.
        self.mirror_off.map(|off| self.base.wrapping_add(off as u32))
    }
}

/// Build the pending-Promise a suspended thread parks on, and signal the scheduler
/// with the stop reason. Returned from the import closure so the guest stack suspends.
fn suspend(
    signal: &Rc<RefCell<Option<Function>>>,
    cont: &Rc<RefCell<Option<Function>>>,
    stop: Stop,
) -> JsValue {
    let cont = cont.clone();
    let park = Promise::new(&mut |resolve, _reject| {
        *cont.borrow_mut() = Some(resolve);
    });
    deliver(signal, &Ev::Suspend(stop));
    park.into()
}

/// A Promise that never resolves - a finished thread's stack parks here forever (it is
/// never resumed), the browser analog of a fiber that has returned.
fn never() -> JsValue {
    Promise::new(&mut |_resolve, _reject| {}).into()
}

/// Fetch `n` exported globals named `name(0)..name(n-1)`.
fn read_globals(
    exports: &JsValue,
    name: impl Fn(usize) -> String,
    n: usize,
) -> Result<Vec<WebAssembly::Global>, JsValue> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(
            Reflect::get(exports, &JsValue::from_str(&name(i)))?
                .dyn_into::<WebAssembly::Global>()?,
        );
    }
    Ok(v)
}

/// Resume `t` to its next scheduler-visible switch point (or its end), asynchronously.
/// A suspend returns immediately; a bare entry return / thread-exit on a NON-final
/// entry does not return to the scheduler - it starts the next entry (a fresh stack)
/// and keeps going, so the whole `module_init` sequence runs as one uninterrupted main
/// thread (matching native's `instantiate_thread_seq`). Only a suspend, a halt, a trap,
/// or the final entry ending yields a [`ThreadStep`].
async fn resume(t: &mut BrowserThread) -> ThreadStep {
    loop {
        // A fresh step channel for this turn; the import closure or the entry's
        // completion fills its resolver.
        let mut resolver = None;
        let step_promise = Promise::new(&mut |res, _rej| resolver = Some(res));
        *t.signal.borrow_mut() = Some(resolver.expect("Promise executor runs synchronously"));

        if !t.entry_started {
            t.entry_started = true;
            // Each entry starts on a fresh stack; only the first carries args.
            t.rt.set_reg(abi::SP, t.sp);
            t.rt.set_reg(0, if t.entry_idx == 0 { t.r0 } else { 0 });
            t.rt.set_reg(1, if t.entry_idx == 0 { t.r1 } else { 0 });
            t.rt.set_reg(2, if t.entry_idx == 0 { t.r2 } else { 0 });
            let done: Promise = match t.entries[t.entry_idx].call0(&JsValue::UNDEFINED) {
                Ok(p) => p.unchecked_into(),
                Err(e) => return ThreadStep::Finished(FiberEnd::Error(format!("start: {e:?}"))),
            };
            // When this entry returns (with no final host call) or traps, deliver the
            // event through whatever step channel is current at that time.
            let sig_ok = t.signal.clone();
            let rt = t.rt.clone();
            let on_ok = Closure::once(Box::new(move |_v: JsValue| {
                deliver(&sig_ok, &Ev::Returned(rt.read_reg(0)));
            }) as Box<dyn FnOnce(JsValue)>);
            let sig_err = t.signal.clone();
            let on_err = Closure::once(Box::new(move |e: JsValue| {
                let msg = e.as_string().unwrap_or_else(|| format!("{e:?}"));
                deliver(&sig_err, &Ev::Error(msg));
            }) as Box<dyn FnOnce(JsValue)>);
            let _ = done.then2(&on_ok, &on_err);
            on_ok.forget();
            on_err.forget();
        } else if let Some(res) = t.cont.borrow_mut().take() {
            // A timed wait that expired owes this thread a return code other than the
            // 0 it parked with (a WAIT_TIMEOUT); write it into r0 before the guest
            // stack resumes. A signal wake has no code and keeps r0 = 0. (Native does
            // the equivalent inside its import closure after the block await.)
            if let Some(code) = t.host.lock().unwrap().take_resume_code(t.thid) {
                t.rt.set_reg(0, code);
            }
            // Un-park: resolving the parked Promise resumes the suspended guest stack.
            let _ = res.call0(&JsValue::UNDEFINED);
        }

        let ev = match JsFuture::from(step_promise).await {
            Ok(v) => decode(&v),
            Err(e) => return ThreadStep::Finished(FiberEnd::Error(format!("resume: {e:?}"))),
        };

        let last = t.entry_idx + 1 >= t.entries.len();
        match ev {
            Ev::Suspend(stop) => return ThreadStep::Suspended(stop),
            Ev::Halt(c) => return ThreadStep::Finished(FiberEnd::ProcessHalt(c)),
            Ev::Error(m) => return ThreadStep::Finished(FiberEnd::Error(m)),
            Ev::Returned(c) if last => return ThreadStep::Finished(FiberEnd::Returned(c)),
            Ev::ThreadExit(c) if last => return ThreadStep::Finished(FiberEnd::ThreadExit(c)),
            // A non-final entry ended (returned or called ExitThread): advance to the
            // next one on a fresh stack, without returning to the scheduler.
            Ev::Returned(_) | Ev::ThreadExit(_) => {
                t.entry_idx += 1;
                t.entry_started = false;
            }
        }
    }
}

/// The browser preemptive run loop: the async twin of native's
/// `Scheduler::run_frames`, composing the shared [`SchedCore`]. Runs until the process
/// halts, all threads finish, the run deadlocks, or `max_frames`/`max_rounds` is hit.
pub async fn run_frames(
    core: &mut SchedCore<BrowserEngine, VitaEnv>,
    max_frames: u64,
    max_rounds: u64,
) -> RunReport {
    let mut rounds = 0u64;
    loop {
        if rounds >= max_rounds {
            return RunReport::RoundLimit;
        }
        rounds += 1;

        let Some(idx) = core.pick_next() else {
            match core.handle_idle() {
                IdleStep::Done(report) => return report,
                IdleStep::Continue => continue,
            }
        };

        let step = resume(core.thread_mut(idx)).await;
        let done = match step {
            ThreadStep::Finished(end) => core.on_finished(idx, end),
            ThreadStep::Suspended(stop) => core.on_suspended(idx, stop, max_frames),
        };
        if let Some(report) = done {
            return report;
        }
        // A host call in this resume may have started threads or woken parked ones.
        core.drain();
    }
}

/// Compile a transpiled module asynchronously. `WebAssembly.Module::new` (sync) is
/// disallowed on the main thread for modules over 8 MB - a real title easily exceeds
/// that - so use async `WebAssembly.compile`, which the caller (already async) awaits.
pub async fn compile_module(wasm: &[u8]) -> Result<WebAssembly::Module, JsValue> {
    let promise = WebAssembly::compile(&Uint8Array::from(wasm).into());
    JsFuture::from(promise).await?.dyn_into::<WebAssembly::Module>()
}

/// The JSPI primitives and a fresh shared memory, ready to build a [`SchedCore`].
pub struct BrowserSched {
    pub core: SchedCore<BrowserEngine, VitaEnv>,
    pub host: Host,
}

impl BrowserSched {
    /// Stand up a preemptive run of `wasm` (the transpiler's `import_memory` module for
    /// a guest loaded at `base`, sized `mem_pages`), seeding `image` into a fresh shared
    /// memory and the main thread ready to run from `entry`. `env` is the single-owner
    /// host every thread dispatches its NID calls to.
    pub fn new(
        module: WebAssembly::Module,
        image: &[u8],
        base: u32,
        mem_pages: u32,
        mirror_off: Option<u64>,
        entry: u32,
        main_sp: u32,
        env: VitaEnv,
    ) -> Result<BrowserSched, JsValue> {
        let (engine, host) = build_engine(module, image, base, mem_pages, mirror_off, env)?;
        let main = engine.make_thread(
            0,
            &[entry & !1],
            0,
            0,
            0,
            main_sp,
            vitaslop_runtime::host::DEFAULT_THREAD_PRIORITY,
        )?;
        let core = SchedCore::new(engine, host.clone(), main);
        Ok(BrowserSched { core, host })
    }

    /// Stand up a preemptive run whose main thread runs `entries` in sequence (a linked
    /// title's `module_init`s in load order, then the eboot entry) on one instance - the
    /// browser twin of native's `ThreadedScheduler::from_linked`. `env` should already
    /// have its alloc base / process param / preemptive flag set and its guest files
    /// preloaded.
    pub fn from_linked(
        module: WebAssembly::Module,
        image: &[u8],
        base: u32,
        mem_pages: u32,
        mirror_off: Option<u64>,
        entries: &[u32],
        main_sp: u32,
        env: VitaEnv,
    ) -> Result<BrowserSched, JsValue> {
        let (engine, host) = build_engine(module, image, base, mem_pages, mirror_off, env)?;
        let main = engine.make_thread(
            0,
            entries,
            0,
            0,
            0,
            main_sp,
            vitaslop_runtime::host::DEFAULT_THREAD_PRIORITY,
        )?;
        let core = SchedCore::new(engine, host.clone(), main);
        Ok(BrowserSched { core, host })
    }
}

/// Build the browser engine (JSPI primitives, module, a fresh seeded shared memory) and
/// the shared host - the common setup both [`BrowserSched`] constructors share.
fn build_engine(
    module: WebAssembly::Module,
    image: &[u8],
    base: u32,
    mem_pages: u32,
    mirror_off: Option<u64>,
    env: VitaEnv,
) -> Result<(BrowserEngine, Host), JsValue> {
    {
        let wasm_global =
            Reflect::get(&js_sys::global(), &JsValue::from_str("WebAssembly"))?;
        let promising = Reflect::get(&wasm_global, &JsValue::from_str("promising"))?
            .dyn_into::<Function>()
            .map_err(|_| JsValue::from_str("WebAssembly.promising missing (needs JSPI)"))?;
        let suspending = Reflect::get(&wasm_global, &JsValue::from_str("Suspending"))?
            .dyn_into::<Function>()
            .map_err(|_| JsValue::from_str("WebAssembly.Suspending missing (needs JSPI)"))?;

        // One shared memory of exactly the transpiler's declared size, imported into
        // every instance. A shared memory needs a maximum and a cross-origin-isolated
        // page (COOP/COEP).
        let desc = Object::new();
        Reflect::set(&desc, &JsValue::from_str("initial"), &JsValue::from_f64(mem_pages as f64))?;
        Reflect::set(&desc, &JsValue::from_str("maximum"), &JsValue::from_f64(mem_pages as f64))?;
        Reflect::set(&desc, &JsValue::from_str("shared"), &JsValue::TRUE)?;
        let shared_mem = WebAssembly::Memory::new(&desc)?;
        // Seed the image at offset 0.
        Uint8Array::new(&shared_mem.buffer())
            .subarray(0, image.len() as u32)
            .copy_from(image);

        // Shared non-suspending env stubs. svc is unused on the Vita path; a dispatch
        // miss (an indirect call to an untranslated target) throws a clear error.
        let svc = Closure::wrap(Box::new(|_sel: i32| {}) as Box<dyn FnMut(i32)>);
        let svc_fn: JsValue = svc.as_ref().clone();
        let dispatch_miss = Closure::wrap(Box::new(|target: i32, caller: i32| -> () {
            let msg = format!(
                "indirect dispatch to unknown target {:#010x} from f_{:x}",
                target as u32, caller as u32
            );
            wasm_bindgen::throw_str(&msg)
        }) as Box<dyn FnMut(i32, i32)>);
        let dispatch_miss_fn: JsValue = dispatch_miss.as_ref().clone();

        let host: Host = Arc::new(Mutex::new(env));
        let engine = BrowserEngine {
            module,
            shared_mem,
            host: host.clone(),
            base,
            promising,
            suspending,
            _svc: svc,
            svc_fn,
            _dispatch_miss: dispatch_miss,
            dispatch_miss_fn,
            mirror_off,
        };

        Ok((engine, host))
    }
}
