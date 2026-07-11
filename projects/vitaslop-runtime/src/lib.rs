//! Executes the transpiler's emitted WASM: guest memory, the guest-address to
//! wasm-function dispatch map, and the execution driver (later the worker-pool
//! scheduler, safepoints, and atomics). Hosts the transpiler-to-runtime ABI
//! and exposes the ROM-to-wasm-blob transform (loader plus transpiler), which
//! runtime already needs for runtime-loaded modules.
//!
//! This crate is engine-agnostic and compiles to wasm32 (the browser blob), so
//! it holds no wasm engine. The concrete engine (wasmtime on native, the
//! browser's `WebAssembly` on the web) lives in the host crate. What lives here
//! is the ABI-level logic both engines share: the register-file view over
//! linear memory and the `svc` host-trap semantics.

use vitaslop_transpiler::abi;

/// Guest state read back after a run: the register file and captured output.
pub struct RunResult {
    pub regs: [u32; abi::REG_COUNT],
    pub output: Vec<u8>,
}

/// What the host's `svc` handler tells the driver to do after a trap.
pub enum SvcOutcome {
    /// Keep running.
    Continue,
    /// The program asked to exit.
    Halt,
}
