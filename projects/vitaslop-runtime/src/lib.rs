//! Executes the transpiler's emitted WASM: guest memory, the guest-address to
//! wasm-function dispatch map, and the execution driver (later the worker-pool
//! scheduler, safepoints, and atomics). Hosts the transpiler-to-runtime ABI
//! and exposes the ROM-to-wasm-blob transform (loader plus transpiler), which
//! runtime already needs for runtime-loaded modules.
//!
//! This crate is engine-agnostic and compiles to wasm32 (the browser blob), so
//! it holds no wasm engine. The concrete engine (wasmtime on native, the
//! browser's `WebAssembly` on the web) lives in the host crate. What lives here
//! is the ABI-level logic both engines share.

use vitaslop_transpiler::abi;

// So `#[hostcall]`'s generated `::vitaslop_runtime::...` paths resolve inside this
// crate itself (the handlers in `vita/` that use the macro live here), not only in
// downstream crates.
extern crate self as vitaslop_runtime;

pub mod audio;
pub mod capture;
pub mod host;
pub mod ingest;
pub mod link;
pub mod nid;
pub mod recipe;
pub mod sched;
pub mod render;
pub mod vita;
pub mod world;

pub use host::{
    GuestCtx, GuestMemory, ImportDispatch, Ptr, Reentry, SliceMemory, SvcDispatch, VitaEnv,
    VitaState, VFP_ARG_COUNT,
};
pub use audio::{AudioFormat, AudioSink, NullSink};
pub use recipe::{Recipe, RecipeError, RecipeWorld};
pub use sched::{
    FiberEnd, GuestEngine, GuestThread, IdleStep, RunReport, SchedCore, Scheduler, Stop,
    ThreadHandle, ThreadStep,
};
pub use world::{
    CtrlFrame, DeterministicWorld, Record, Replay, TouchFrame, TouchPoint, World, WorldEvent,
    MAX_TOUCH_POINTS,
};

/// Write a Vita host handler as a typed function; the macro generates the AAPCS-
/// VFP argument marshalling and return write. See [`vitaslop_hostcall`].
pub use vitaslop_hostcall::hostcall;

/// The N,Z,C,V condition flags read back after a run.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Flags {
    pub n: bool,
    pub z: bool,
    pub c: bool,
    pub v: bool,
}

/// Guest state read back after a run: the register file, condition flags, and
/// captured output.
pub struct RunResult {
    pub regs: [u32; abi::REG_COUNT],
    pub flags: Flags,
    pub output: Vec<u8>,
}

/// What the host's `svc`/`import` handler tells the driver to do after a trap.
pub enum SvcOutcome {
    /// Keep running.
    Continue,
    /// The call was serviced, but the guest hit a blocking primitive and should
    /// now suspend here, yielding to the cooperative scheduler. The classic case
    /// is the per-frame display flip: the frame is complete and on hardware the
    /// thread would wait for the flip. A host with no scheduler (the run-to-
    /// completion conformance and capture paths) treats this exactly like
    /// `Continue`, so the outcome is a safe hint, never a requirement.
    Yield,
    /// The program asked to exit the whole process; unwind and stop the run.
    Halt,
    /// The calling thread must wait on an unavailable primitive (an empty
    /// semaphore, a held mutex, an unset event flag). The preemptive scheduler
    /// ([`vitaslop_native::ThreadedScheduler`]) parks this thread and runs another;
    /// the thread resumes - and its wait call returns - once another thread makes
    /// the primitive available (see [`ImportDispatch::take_wakes`]). A single-
    /// worker host never produces this (its uncontended waits succeed at once), and
    /// if one somehow received it, it treats it as [`Continue`](Self::Continue).
    ///
    /// [`vitaslop_native::ThreadedScheduler`]: https://docs.rs/vitaslop-native
    Block,
    /// The call serviced fine and the thread stays runnable, but the scheduler
    /// should re-pick now rather than let this thread run on. Emitted when a host
    /// call makes a higher-priority thread runnable (e.g. `sceKernelStartThread` of
    /// a higher-priority worker): the real kernel preempts the caller and runs the
    /// higher-priority thread until it blocks, then resumes the caller. A host with
    /// no scheduler treats this like [`Continue`](Self::Continue).
    Reschedule,
    /// The calling thread has ended (a worker returned, or called
    /// `sceKernelExitThread`), but the process keeps running. The preemptive
    /// scheduler finishes just this thread's fiber, keeping its siblings alive; the
    /// exit code is the guest's r0. A single-worker host has only the one thread,
    /// so it treats this exactly like [`Halt`](Self::Halt).
    ThreadExit,
}
