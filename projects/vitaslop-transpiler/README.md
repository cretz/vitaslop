# vitaslop-transpiler

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

ARMv7-A / Thumb-2 / NEON / VFP to WASM. Vita-agnostic. Runs in-browser at
module-load time, so it stays lean: allocation-light, per-function passes, no
whole-program analysis on the load path.

- Pipeline: decode + discover -> per-function CFG IR -> emit WASM.
- Decode comes from our yaxpeax-arm consolidation fork (see its repo). The
  transpiler consumes decoded operands, not disassembly text.

## Emitted module

- Imports two host traps: an svc trap (ARM `svc` imm) and an import trap (NID
  index). Either may trap to unwind, which is how a run stops.
- Defines 16 register globals + 4 flag globals + the VFP register/flag globals.
- Exports `memory` and one `() -> ()` function per guest function, named by
  address.

## Translation unit: the guest function

- Each guest function becomes one wasm function.
- Functions are discovered as the transitive direct-call closure from the entry
  point(s). Decoding follows only real control-flow edges (fall-through, taken
  and not-taken branches), so inline literal pools and post-branch padding are
  never decoded as instructions - nothing branches into them.
- A call target is recorded as a separate function to discover, never inlined.

## Control flow: guest edges onto wasm

The hard problem in any CPU-to-wasm recompiler is mapping address-based guest
control flow onto wasm's structured, label-based control flow.

- **Intra-function branches** (b, conditional b, cbz/cbnz, loops) compile to a
  dispatch loop: one `loop` wrapping a `br_table` on a current-block local, with
  basic blocks emitted in ascending address order. Adjacent fall-through needs no
  branch. Only real jumps and loop back-edges pay a branch back to the dispatch,
  so hot loops stay in-function with no host round-trips.
- **Direct calls** are wasm calls. **Returns** (`bx lr`, `pop {..,pc}`) are wasm
  returns. The wasm call stack mirrors the guest call stack and stays out of the
  dispatch machinery. `lr` is still set on a call, but control transfer uses
  wasm's stack.
- **Host calls**: a call/branch to an import stub becomes a call to the import
  trap with the import's dense index. An ARM `svc` becomes a call to the svc trap.
- **IT blocks**: the decoder yields predicated instructions as unconditional, so
  the transpiler tracks ITSTATE itself along fall-through edges and guards each
  body instruction's effects on its condition. Inside an IT block the S bit is
  suppressed (only cmp/cmn/tst set flags), matching ARM and required because the
  16-bit encodings decode as the flag-setting form.

Deferred (the IR already accommodates them): a relooper for structured
loops/ifs, and general indirect branches/calls via an indirect-call table plus a
host reentry trampoline. Today only direct calls plus register/stack returns are
handled, which covers compiled C.

## Condition flags

- N,Z,C,V are four mutable globals, computed eagerly by flag statements.
  Separate globals so a branch testing one flag is a single read (the hot path).
  mrs/msr pack and unpack across them.
- Add and sub share one primitive (`a + b + carry_in`, subtract passes `~b`,
  carry_in 1) with an i64 widening for an always-correct unsigned carry.
- Lazy flags (materialize a condition only where consumed) are a later
  optimization the IR seam already isolates.

## Memory

- Guest memory is the module's linear memory, rebased: guest address A maps to
  offset `A - base`.
- Why rebased: a Vita module loads at 0x81000000, and identity mapping would
  force a 2 GB minimum. Every load/store subtracts the image base (one sub the
  JIT folds into the address); small immediate displacements fold into the wasm
  access offset.
- The host keeps all guest memory (image, stack, host allocations) at addresses
  >= base, so every translated offset is non-negative. The transpiler only knows
  "subtract base"; placement policy is the host's, keeping the transpiler
  Vita-agnostic.

## Registers

- ARM r0..r15 live in 16 mutable globals (r13 = sp, r14 = lr, r15 = pc), not
  linear memory. VFP registers live in their own globals (32 single-precision as
  raw bits, aliasing the low double-precision set, plus the high doubles).
- Guest functions take no register params. Registers are ambient state read and
  written via global access.
- **Why globals, not linear memory**: a register file in memory would alias guest
  data (any guest store to a computed address might hit a register slot), which
  blocks the engine from optimizing register values across stores. Globals live
  outside memory, so no aliasing and the engine optimizes freely within a
  call-free span.
- **Why exported globals**: the host reads and writes registers for seeding, for
  svc/NID args and returns, and for reentry. Exported globals let the JIT keep
  optimizing within a call-free span while the host caches the global handles once
  after instantiation to avoid per-access name lookup.

### Reentry (why the host needs write access)

- Host callbacks (thread entry, kernel callbacks, host libc calling a guest
  comparator) set the callback's argument registers, then call the guest entry
  block. Same mechanism as initial seeding, nested.
- This is why registers must stay host-accessible globals: the host mutates
  arbitrary registers before re-entering guest code.

### Deferred: local promotion

- Caching a hot register in a wasm local within a call-free span is planned but
  deferred until the register/calling model is exercised more widely. Today all
  register access goes through globals, which is correct and JIT-friendly.
