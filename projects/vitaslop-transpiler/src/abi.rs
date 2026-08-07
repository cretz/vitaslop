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

/// Number of imported functions the module declares before any guest function
/// (`svc`, `host_import`, `dispatch_miss`), so wasm function index
/// `IMPORT_FUNC_COUNT + i` is the i-th translated guest function in
/// ascending-address order. A wasmtime backtrace's "wasm function N" is a module
/// index that counts these imports, so mapping it back to a guest function is
/// `funcs[N - IMPORT_FUNC_COUNT]`.
pub const IMPORT_FUNC_COUNT: u32 = 3;

/// Number of general-purpose ARM registers (r0..r15).
pub const REG_COUNT: usize = 16;

/// Index of the stack pointer within the register file (r13).
pub const SP: usize = 13;
/// Index of the link register within the register file (r14).
pub const LR: usize = 14;
/// Index of the program counter within the register file (r15).
pub const PC: usize = 15;

/// The four condition flags, in bit order N(31) Z(30) C(29) V(28).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

// --- VFP / NEON floating-point state --------------------------------------
//
// The VFP/NEON register file (Cortex-A9 = VFPv3-D32 + NEON) is modeled to respect
// S/D/Q aliasing exactly while splitting into two banks by how the host uses them:
//
// * **Low bank (S0..S31 / D0..D15 / Q0..Q7).** The 32 single-precision registers
//   S0..S31 are the low 32-bit halves of the doubles D0..D15, stored as 32
//   **raw-bit i32 globals** (`s0..s31`): reading S`n` as an f32 is an
//   `f32.reinterpret_i32`, and D`n` (n < 16) is the concatenation of `s[2n]` (low)
//   and `s[2n+1]` (high). These stay i32 (not a wider type) because the host must
//   read/write them to marshal VFP call arguments and returns, and the JS
//   WebAssembly API cannot access `v128` globals - so the browser host needs the
//   low bank to be plain scalars.
//
// * **Upper bank (D16..D31 / Q8..Q15).** These have no single-precision alias and
//   the host never marshals them, so they are stored as 8 **`v128` globals**
//   (`q8..q15`), one per quad register: D`n` (n >= 16) is `i64x2` lane `n & 1` of
//   `q(n/2)`. Storing the upper bank as `v128` lets NEON data-processing (which
//   gcc's auto-vectorizer parks in Q8..Q15) map straight onto wasm 128-bit SIMD
//   with no per-op gather/scatter; NEON on the low bank assembles a `v128` from the
//   `s` globals instead (correct, and rare in auto-vectorized output).
//
// Floating-point compare results (FPSCR N,Z,C,V) live in four more i32 globals;
// `vmrs APSR_nzcv, fpscr` copies them into the integer condition flags.

/// Number of single-precision VFP registers (S0..S31), stored as raw-bit i32.
pub const VFP_S_COUNT: usize = 32;
/// First double register in the upper (`v128`) bank - D16, the low half of Q8.
pub const VFP_D_HI_FIRST: usize = 16;
/// First quad register in the upper (`v128`) bank (Q8).
pub const VFP_Q_HI_FIRST: usize = 8;
/// Number of upper quad registers (Q8..Q15), stored as `v128` globals.
pub const VFP_Q_HI_COUNT: usize = 8;
/// Number of floating-point condition-flag globals (FPSCR N,Z,C,V).
pub const FP_FLAG_COUNT: usize = 4;

/// Base global index of the register + integer-flag block (r0..r15, nf..vf).
const CORE_GLOBALS: u32 = REG_COUNT as u32 + FLAG_COUNT as u32;

/// WASM global index of single-precision register S`n` (raw bits, i32).
pub const fn vfp_s_global(n: u8) -> u32 {
    CORE_GLOBALS + n as u32
}

/// WASM global index of an upper quad register Q`q` (8 <= q < 16; `v128`).
pub const fn vfp_qhi_global(q: u8) -> u32 {
    CORE_GLOBALS + VFP_S_COUNT as u32 + (q as u32 - VFP_Q_HI_FIRST as u32)
}

/// WASM global index of floating-point condition flag `f` (FPSCR N,Z,C,V; i32).
pub const fn fp_flag_global(f: Flag) -> u32 {
    CORE_GLOBALS + VFP_S_COUNT as u32 + VFP_Q_HI_COUNT as u32 + f as u32
}

/// Total number of registers + integer flags (the i32-only core block, exported
/// under their `r*`/`nf`.. names and seeded/read by the host).
pub const GLOBAL_COUNT: u32 = CORE_GLOBALS;

/// Total number of globals including the VFP/NEON register file and FP flags.
pub const TOTAL_GLOBAL_COUNT: u32 =
    CORE_GLOBALS + VFP_S_COUNT as u32 + VFP_Q_HI_COUNT as u32 + FP_FLAG_COUNT as u32;

/// Export name of single-precision register S`n`.
pub fn vfp_s_export(n: u8) -> String {
    format!("s{n}")
}

/// Export name of upper quad register Q`q` (8 <= q < 16).
pub fn vfp_qhi_export(q: u8) -> String {
    format!("q{q}")
}

/// Export name of floating-point condition flag `f`.
pub fn fp_flag_export(f: Flag) -> &'static str {
    match f {
        Flag::N => "fpn",
        Flag::Z => "fpz",
        Flag::C => "fpc",
        Flag::V => "fpv",
    }
}

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

/// Exported name of the diagnostic guest-PC tracker global. Holds the address of
/// the basic block currently executing when `VITASLOP_TRACK_PC` is set at emit time;
/// zero otherwise. Hosts read it on a trap to name the faulting guest instruction.
pub const GUEST_PC_EXPORT: &str = "guest_pc";

/// WASM global index of the per-thread pointer (ARM `TPIDRURO`, the read-only
/// user-mode thread ID register read by `MRC p15,0,Rt,c13,c0,3`). It holds the base
/// of this thread's thread-local-storage block; the compiler reaches `__thread`
/// variables at `tp + offset`. A per-instance global (like the register file), so
/// every thread sees its own value; the host sets it when it instantiates the
/// thread. Placed after the three diagnostic globals so their indices stay stable.
pub const TP_GLOBAL: u32 = TOTAL_GLOBAL_COUNT + 3;

/// Exported name of the per-thread pointer global (see [`TP_GLOBAL`]). The host
/// writes it per thread with that thread's TLS block base.
pub const TP_EXPORT: &str = "tp";

/// WASM global index of the software fuel counter, appended after the store-watchpoint
/// counter. Per-instance, and in this engine an instance IS a guest thread, so each
/// thread carries its own quantum with no host bookkeeping. Zero and never read unless
/// the build opted into fuel (see `emit::set_fuel_interval`).
pub const FUEL_GLOBAL: u32 = TOTAL_GLOBAL_COUNT + 5;

/// Exported name of the software fuel counter (see [`FUEL_GLOBAL`]). Exported so a host
/// can read how much of a thread's quantum is left; nothing needs to WRITE it - the
/// emitted code reloads it itself after each yield.
pub const FUEL_EXPORT: &str = "fuel";

/// Reserved [`IMPORT_NAME`] selector meaning "this thread's fuel ran out - reschedule
/// it". Not a NID: it is above any import index a real title can have, and the host
/// intercepts it before the import table is consulted.
///
/// # Why the fuel yield reuses `env.import` instead of a fourth import
/// `env.import` is already the one call the host has wrapped for suspension on both
/// engines - JSPI `Suspending` in the browser, a fiber switch natively. A separate
/// import would have to be wrapped identically in both hosts to do the same thing, and
/// would shift [`IMPORT_FUNC_COUNT`], which every wasm-backtrace-to-guest-function
/// mapping depends on. A reserved selector costs neither.
pub const FUEL_SELECTOR: u32 = u32::MAX;

/// Import module every host function/memory comes from.
pub const IMPORT_MODULE: &str = "env";
/// Import name of the ARM `svc` host trap: `(i32 imm) -> ()`.
pub const SVC_NAME: &str = "svc";
/// Legacy alias kept for existing hosts.
pub const SVC_MODULE: &str = IMPORT_MODULE;
/// Import name of the Vita NID host trap: `(i32 index) -> ()`.
pub const IMPORT_NAME: &str = "import";
/// Import name of the indirect-dispatch miss reporter: `(i32 target, i32 caller) ->
/// ()`. The dispatcher calls it when a runtime function-pointer resolves to no
/// translated function; the host records the addresses and traps, turning an
/// otherwise opaque `unreachable` into a debuggable "unknown target from caller".
pub const DISPATCH_MISS_NAME: &str = "dispatch_miss";

/// WASM page size in bytes.
pub const PAGE_SIZE: u32 = 65536;

/// Exported name of the wasm function for the guest function at `addr`.
pub fn func_export(addr: u32) -> String {
    format!("f_{addr:x}")
}
