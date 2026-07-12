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

pub mod capture;
pub mod host;
pub mod nid;
pub mod render;
pub mod vita;
pub mod world;

pub use host::{
    GuestCtx, GuestMemory, ImportDispatch, Ptr, SliceMemory, SvcDispatch, VitaEnv, VitaState,
    VFP_ARG_COUNT,
};
pub use world::{CtrlFrame, DeterministicWorld, Record, Replay, World, WorldEvent};

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
    /// The program asked to exit; unwind and stop the run.
    Halt,
}
