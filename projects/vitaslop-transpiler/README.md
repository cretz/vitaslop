# vitaslop-transpiler

ARMv7-A / Thumb-2 (later NEON/VFP) to WASM. Vita-agnostic.

Pipeline: decode (yaxpeax) -> small IR -> emit (wasm-encoder). It runs in-browser
at module-load time, so it stays lean: allocation-light, per-function passes only,
no cross-function analysis on the load path.

Emitted module: imports `env.svc`, defines 16 register globals, exports `memory`
and one `() -> i32` function per block (returns next guest pc, or `HALT`).

## Registers

- ARM `r0..r15` live in **16 mutable wasm globals** (`r15` = pc), not linear memory.
- Block functions take **no register params**. Registers are ambient state read/
  written via `global.get`/`global.set` (or cached locals, below).

### Why globals, not linear memory
- The register file in linear memory aliases guest data: any guest `str` to a
  computed address *might* hit a register slot, so the engine can't keep register
  values optimized across guest stores. Globals live outside memory -> no aliasing
  -> the engine is free to optimize register access.

### Local promotion (in-block speed)
- A block is one wasm function, split into **segments** at each `svc` (a host
  boundary: registers must be current there, since the host may read/change them).
- A register hot **within a segment** (accessed past `LOCAL_PROMOTION_THRESHOLD`)
  is cached in a wasm local: loaded from its global at segment start, used as a
  local, flushed back at the boundary.
- Per **segment**, not per block: every boundary spills to globals regardless, so
  only access density *between* boundaries earns a local. Whole-block counts would
  over-credit.
- Local slots are declared once per function (the union of registers promoted in
  any segment); each segment independently decides whether to use its slot.
- Decided during the single decode pass, not by re-reading emitted wasm.

### Host register transfer: why exported globals
The host reads/writes registers for seeding, `svc` args/returns, and reentry.
Considered params/returns and getter/setter funcs; chose **exported globals**:
- Known downside: a JIT (V8, Cranelift) must assume a call can clobber an exported
  mutable global, so it cannot cache one across a call. This is real in general but
  **perf-neutral here**: a segment is a call-free span, so the engine still
  optimizes globals freely *within* a segment (no intervening call), and at every
  boundary we already flush/reload for reentry correctness - so we never cache a
  register across a `svc` regardless, not even the high registers the host never
  touches. Nothing is left for the clobber assumption to spoil.
- Their other cost - a host-side name lookup per access - is removed by caching the
  global handles once after instantiation.
- params/returns then buys ~nothing and adds per-syscall ABI knowledge plus codegen
  complexity, for a path that is not hot (thousands of ALU ops per syscall). Rejected.
- getter/setter funcs existed only to avoid exporting globals; exporting is fine, so
  they are pure overhead. Rejected.
- Future exception: a hot Vita NID call has a known signature, so that one call could
  pass args as params. Targeted later optimization only.

### Reentry (why the host needs write access)
- Host callbacks (thread entry, `sceKernel` callbacks, host libc calling a guest
  comparator) set the callback's argument registers in the globals, then call the
  guest entry block. Same mechanism as initial seeding, nested.
- This is why registers must stay host-accessible globals, not private/params-only:
  the host mutates arbitrary registers *before* re-entering guest code.
