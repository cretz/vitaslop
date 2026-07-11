//! The transpiler-to-runtime ABI: the contract the emitted WASM and its host
//! agree on. The transpiler emits to it; the runtime (native or browser) hosts
//! it. Lives here because the transpiler is the dependency leaf.
//!
//! # Guest state
//! - The ARM register file is 16 mutable i32 wasm **globals**, `r0..r15`
//!   (`r13` = sp, `r14` = lr, `r15` = pc), each exported under its name (see
//!   [`reg_export`]). Globals (not linear memory) so guest memory stores cannot
//!   alias register accesses, which would otherwise block the wasm engine from
//!   optimizing them.
//! - The four `NZCV` condition flags are four more mutable i32 globals (0 or 1),
//!   exported as `nf`/`zf`/`cf`/`vf` (see [`flag_export`]). Separate globals (not
//!   a packed CPSR) so testing a single flag for a conditional branch is one
//!   `global.get`, the hot path; `mrs`/`msr` pack/unpack across them.
//! - Guest memory is the module's linear memory, exported as `memory`. It is the
//!   guest address space **rebased**: guest address `A` maps to linear offset
//!   `A - base` (see [the memory model](#memory-model)).
//!
//! # Memory model
//! Identity-mapping guest addresses into linear memory is impossible: a Vita
//! module loads at `0x81000000`, which would force a 2 GB minimum memory. Instead
//! the transpiler subtracts the image `base` from every guest address, so linear
//! memory starts at the image and stays compact. The host keeps all guest memory
//! (image, stack, host allocations) at guest addresses `>= base`, so every
//! translated offset is non-negative. Small `[reg, #imm]` displacements fold into
//! the wasm load/store `offset` immediate; the `- base` rebase is one `i32.sub`
//! the JIT folds into the address computation.
//!
//! # Host calls
//! Two imported host functions, both reading/writing guest state through the
//! globals and guest memory:
//! - `env.svc : (i32 imm) -> ()` services an ARM `svc` (the Linux-EABI corpus).
//! - `env.import : (i32 index) -> ()` services a Vita NID call: a `bl`/`blx` to an
//!   import stub becomes a call with the import's dense index (see
//!   [`Extern`](crate::Extern)).
//! Either may abort the run by trapping (host returns an error / throws), which
//! the driver catches as "halted" - that is how `exit` and fatal errors unwind.
//!
//! # Control flow
//! Each guest function is one wasm function, exported as `f_<hexaddr>` (see
//! [`func_export`]). Its body is a dispatch loop over its basic blocks; intra-
//! function branches stay inside it, direct `bl`/`blx` become wasm `call`s to the
//! callee's function, and `bx lr` / `pop {pc}` become wasm `return`. The host
//! enters guest code (initial entry and callbacks) by calling an exported
//! `f_<addr>`.

/// Number of general-purpose ARM registers (r0..r15).
pub const REG_COUNT: usize = 16;

/// Index of the stack pointer within the register file (r13).
pub const SP: usize = 13;
/// Index of the link register within the register file (r14).
pub const LR: usize = 14;
/// Index of the program counter within the register file (r15).
pub const PC: usize = 15;

/// The four condition flags, in bit order N(31) Z(30) C(29) V(28).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Flag {
    N = 0,
    Z = 1,
    C = 2,
    V = 3,
}

/// Number of flag globals.
pub const FLAG_COUNT: usize = 4;

/// WASM global index holding ARM register `i` (r0..r15 are globals 0..15).
pub const fn reg_global(i: usize) -> u32 {
    i as u32
}

/// WASM global index holding condition flag `f` (flags follow the 16 registers).
pub const fn flag_global(f: Flag) -> u32 {
    REG_COUNT as u32 + f as u32
}

/// Total number of exported globals (registers + flags).
pub const GLOBAL_COUNT: u32 = REG_COUNT as u32 + FLAG_COUNT as u32;

/// Export name of the global holding ARM register `i`.
pub fn reg_export(i: usize) -> String {
    format!("r{i}")
}

/// Export name of the global holding condition flag `f`.
pub fn flag_export(f: Flag) -> &'static str {
    match f {
        Flag::N => "nf",
        Flag::Z => "zf",
        Flag::C => "cf",
        Flag::V => "vf",
    }
}

/// A register accessed more than this many times within a basic block is promoted
/// to a wasm local for that block instead of touching its global on each access.
pub const LOCAL_PROMOTION_THRESHOLD: u32 = 2;

/// Exported name of the linear memory.
pub const MEMORY_EXPORT: &str = "memory";

/// Import module every host function/memory comes from.
pub const IMPORT_MODULE: &str = "env";
/// Import name of the ARM `svc` host trap: `(i32 imm) -> ()`.
pub const SVC_NAME: &str = "svc";
/// Legacy alias kept for existing hosts.
pub const SVC_MODULE: &str = IMPORT_MODULE;
/// Import name of the Vita NID host trap: `(i32 index) -> ()`.
pub const IMPORT_NAME: &str = "import";

/// WASM page size in bytes.
pub const PAGE_SIZE: u32 = 65536;

/// Exported name of the wasm function for the guest function at `addr`.
pub fn func_export(addr: u32) -> String {
    format!("f_{addr:x}")
}
