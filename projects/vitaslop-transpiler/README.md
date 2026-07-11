# vitaslop-transpiler

ARMv7-A / Thumb-2 (later NEON/VFP) to WASM. Vita-agnostic.

Pipeline: decode + discover (`lower`) -> per-function CFG IR (`ir`) -> emit
(`emit`). It runs in-browser at module-load time, so it stays lean:
allocation-light, per-function passes, no whole-program analysis on the load path.

Emitted module: imports `env.svc` and `env.import` (host traps), defines 16
register + 4 flag globals, exports `memory`, and exports one `() -> ()` function
per guest function, named `f_<hexaddr>`.

## Translation unit: the guest function

Each **guest function** becomes one **wasm function**. Functions are discovered
as the transitive direct-call closure from the entry point(s): decoding follows
only real control-flow edges (fall-through, taken/not-taken branches), so inline
literal pools and the padding after an unconditional branch are never decoded as
instructions - nothing branches into them. A `bl`/`blx` target is recorded as a
separate function to discover, never inlined into the caller.

### Control flow: guest edges onto wasm

The hard problem in any CPU->wasm recompiler is mapping address-based guest
control flow onto wasm's structured, label-based control flow. The mapping:

- **Intra-function branches** (`b`, `b<cond>`, `cbz`/`cbnz`, loops) compile to a
  **dispatch loop**: one `loop` wrapping a `br_table` on a `$bb` (current-block)
  local, with each basic block emitted in ascending address order. Straight-line
  fall-through between adjacent blocks needs no branch (control flows through the
  block boundary); only real jumps and loop back-edges pay a `br` back to the
  dispatch. So hot loops stay entirely in-function - no host round-trips - which
  the block-return-to-a-host-dispatcher model could never achieve.
- **Direct calls** (`bl`/`blx` to guest) are wasm `call`s; **returns** (`bx lr`,
  `pop {..,pc}`) are wasm `return`s. The wasm call stack therefore mirrors the
  guest call stack and stays out of the dispatch machinery. `lr` is still set on a
  call (guest code may save/inspect it), but control transfer uses wasm's stack.
- **Host calls**: a `bl`/`blx` to an import stub becomes `call $import` with the
  import's dense index; an ARM `svc` becomes `call $svc`. Either may trap to
  unwind (that is how `exit` stops a run).
- **IT blocks**: yaxpeax decodes predicated instructions as unconditional, so the
  transpiler tracks ITSTATE itself (along fall-through edges) and wraps each body
  instruction's effects in a condition `Guard`. Inside an IT block the S bit is
  suppressed (only `cmp`/`cmn`/`tst` still set flags) - matching ARM, and needed
  because the 16-bit encodings decode as the flag-setting form.

Deferred (the IR/dispatch model already accommodates them): a relooper that
recovers structured loops/ifs for hotter codegen; general indirect
branches/calls via a `call_indirect` table and a host reentry trampoline (only
direct calls + register/stack returns are handled today, which covers compiled C).

### Condition flags

N,Z,C,V are four mutable i32 globals (`nf`/`zf`/`cf`/`vf`), computed **eagerly**
by the flag statements: separate globals so testing one flag for a branch is a
single `global.get` (the hot path); `mrs`/`msr` pack/unpack across them. Add/sub
share one primitive (`a + b + cin`; subtract passes `~b, cin=1`) and use an i64
widening for an always-correct unsigned carry. Lazy flags (materialize a
condition only where consumed) are a later optimization the IR seam already
isolates.

## Memory

Guest memory is the module's linear memory, but **rebased**: guest address `A`
maps to linear-memory offset `A - base`. Identity-mapping is impossible - a Vita
module loads at `0x81000000`, which would force a 2 GB minimum memory. So every
load/store subtracts the image `base` (one `i32.sub` the JIT folds into the
address); small `[reg, #imm]` displacements fold into the wasm access offset. The
host keeps all guest memory (image, stack, host allocations) at addresses
`>= base`, so every translated offset is non-negative. The transpiler stays
Vita-agnostic: it only knows "subtract base"; the placement policy is the host's.

## Registers

- ARM `r0..r15` live in **16 mutable wasm globals** (`r13` = sp, `r14` = lr,
  `r15` = pc), not linear memory. Condition flags are 4 more globals.
- Guest functions take **no register params**. Registers are ambient state read/
  written via `global.get`/`global.set`.

### Why globals, not linear memory
- The register file in linear memory aliases guest data: any guest `str` to a
  computed address *might* hit a register slot, so the engine can't keep register
  values optimized across guest stores. Globals live outside memory -> no aliasing
  -> the engine is free to optimize register access within a call-free span.

### Host register transfer: why exported globals
The host reads/writes registers for seeding, `svc`/NID args and return values,
and reentry (see below). Exported globals were chosen over params/returns and
getter/setter funcs: the JIT can still optimize globals freely within a call-free
span, and the host's per-access name lookup is removed by caching the global
handles once after instantiation. A hot NID call with a known signature could
later pass args as params as a targeted optimization.

### Deferred: local promotion
Caching a hot register in a wasm local within a call-free span is a planned
optimization (`LOCAL_PROMOTION_THRESHOLD` is reserved for it). It is deferred
until the register file and calling model are exercised more widely; today all
register access goes straight through the globals, which is correct and lets the
JIT optimize freely between calls.

### Reentry (why the host needs write access)
- Host callbacks (thread entry, `sceKernel` callbacks, host libc calling a guest
  comparator) set the callback's argument registers in the globals, then call the
  guest entry block. Same mechanism as initial seeding, nested.
- This is why registers must stay host-accessible globals, not private/params-only:
  the host mutates arbitrary registers *before* re-entering guest code.
