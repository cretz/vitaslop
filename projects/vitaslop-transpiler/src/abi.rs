//! The transpiler-to-runtime ABI: the contract the emitted WASM and its host
//! agree on. The transpiler emits to it; the runtime (native or browser) hosts
//! it. Lives here because the transpiler is the dependency leaf.
//!
//! # Guest state
//! - The ARM register file is 16 mutable i32 wasm **globals**, `r0..r15`
//!   (`r15` = pc), each exported under its name (see [`reg_export`]). Globals
//!   (not linear memory) so guest memory stores cannot alias register accesses,
//!   which would otherwise block the wasm engine from optimizing them.
//! - Guest memory is the module's linear memory, exported as `memory`. Guest
//!   addresses are identity-mapped to offsets; the image loads at `base`.
//!
//! # Register caching
//! Within a block the transpiler may promote a hot register to a wasm local
//! (see [`LOCAL_PROMOTION_THRESHOLD`]): loaded from its global on entry, used as
//! a local, flushed back to the global at every boundary (block exit and before
//! any host `svc`, which observes registers through the globals).
//!
//! # Module shape
//! - imports `env.svc : (i32 imm) -> ()` (the host trap entry)
//! - defines and exports 16 mutable i32 globals `r0..r15`
//! - exports the linear memory as `memory`
//! - exports one function per transpiled block, named `b_<hexaddr>`, with
//!   signature `() -> i32` returning the next guest pc to run, or `HALT`.

/// Number of general-purpose ARM registers (r0..r15).
pub const REG_COUNT: usize = 16;

/// Index of the program counter within the register file (r15).
pub const PC: usize = 15;

/// WASM global index holding ARM register `i` (r0..r15 are globals 0..15).
pub const fn reg_global(i: usize) -> u32 {
    i as u32
}

/// Export name of the global holding ARM register `i`.
pub fn reg_export(i: usize) -> String {
    format!("r{i}")
}

/// A register accessed more than this many times within a block is promoted to
/// a wasm local for that block (loaded from its global on entry, flushed back at
/// exits and before host calls) instead of touching its global on each access.
/// At or below this count it is read/written through its global directly, since
/// the load/flush framing would not pay for itself.
pub const LOCAL_PROMOTION_THRESHOLD: u32 = 2;

/// Exported name of the linear memory.
pub const MEMORY_EXPORT: &str = "memory";

/// Import module/name of the host trap entry (`svc`).
pub const SVC_MODULE: &str = "env";
pub const SVC_NAME: &str = "svc";

/// Value a block function returns to signal "stop running". No guest block lives
/// at address 0, so it is unambiguous.
pub const HALT: i32 = 0;

/// WASM page size in bytes.
pub const PAGE_SIZE: u32 = 65536;

/// Exported name of the block function starting at guest address `addr`.
pub fn block_export(addr: u32) -> String {
    format!("b_{addr:x}")
}
