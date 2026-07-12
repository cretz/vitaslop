# vitaslop-runtime

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

Engine-agnostic core (compiles to wasm32): guest memory, the NID dispatch, the
scheduler, the host-call ABI, and the determinism seam. Holds no wasm engine; the
host supplies one (vitaslop-native or the browser). This README is the
authoritative design record for the host-call and concurrency model; other crates
point here.

## Host-call ABI

- Guest calls a NID import stub, which the transpiler turned into an import trap
  carrying a dense index. The runtime maps that index to a handler and dispatches.
- **Typed handlers**: each host function is written as a normal typed Rust
  function. A hostcall attribute generates the marshalling glue, so behavior stays
  hand-written and only the boring register/memory shuffling is generated.
- **Arg marshalling**: reads AAPCS args and writes the return. The Vita is
  hardfloat (VFP), so each arg is classified int-class vs float-class and pulled
  from the right register file, with stack spill for the rest.
- **Type taxonomy across the boundary** (both directions):
  - Scalars: direct in registers.
  - Guest pointers: a guest address wrapper, used for in-params and out-params
    (out-params are pervasive in GXM create-calls).
  - Structs: repr-C mirrors read/written field by field from guest memory, with
    layout asserts. These are cold control-plane calls, so safety over raw casts.
  - Guest callbacks: a guest code address the host can re-enter (GXM ring-buffer
    callbacks). Re-entrancy uses the same register-seed-then-call mechanism as
    thread entry.
- **Autogen later**: generate typed signatures and struct mirrors from the
  vita-headers NID database once the hand-written pattern is proven. Generate the
  glue, never behavior. No monolithic trait of thousands of methods.

## Handles vs heap (kept distinct)

- **Heap** = the single shared linear guest arena. Hot path. Native wasm
  loads/stores. This is what needs to be fast, and it already is.
- **Handles** = opaque IDs returned by create-calls (context, render target, sync
  object). Cold path, created rarely. Small integers into a host-side table. The
  guest never reads inside them.
- Objects the guest does read fields from are instead backed by a real struct in
  guest memory (guest-pointer-backed), decided per type. Most GXM handles are pure
  opaque.

## Memory model

- One shared linear memory (SharedArrayBuffer-backed in the browser) in every
  mode, so representation never changes between single- and multi-worker. Other
  workers (GPU, audio) read the same buffer.
- Memory is behind an abstraction: shared linear memory in the browser, an mmap or
  vector arena on native. The fast path (transpiled loads/stores) is identical;
  only the backing differs.
- Guest addresses are rebased (see the transpiler README). The CPU worker is the
  sole writer; service workers only read at defined sync points, so sharing does
  not introduce races.
- **Host-call memory seam (realized): `GuestMemory`.** The abstraction above is
  live for host calls. `ImportDispatch`/`GuestCtx` reach guest memory through the
  `GuestMemory` trait, not a raw `&mut [u8]`. Native wasmtime shares one address
  space with the guest, so its impl (`SliceMemory`) is a zero-copy slice. In the
  browser the guest runs as its **own** `WebAssembly` instance with its own linear
  memory that this (wasm-bindgen) module cannot borrow as a Rust slice, so the
  impl copies through a `Uint8Array` over the guest `ArrayBuffer` (in
  `vitaslop-web::web_vm`). This is cheap because host calls happen only at
  kernel/GXM boundaries, never in the hot loop. NOTE this is the two-separate-
  memories model, not yet the single-shared-SharedArrayBuffer vision above:
  unifying them (so GPU/audio service workers read one buffer) is future work tied
  to the cooperative scheduler.
- Leak discipline: guest-internal alloc/free stays inside the one arena (no
  per-object host allocation). Host bookkeeping that can leak - handle tables, the
  capture stream, the World log - is bounded and tied to guest destroy-calls.

## Concurrency model

- **Two modes, config choice, shared memory in both**:
  - Single-worker cooperative: guest threads are coroutines we schedule.
    Deterministic and step-debuggable.
  - Multi-worker: guest cores run in parallel. Faithful, non-deterministic replay
    accepted.
- **Scheduler**: cooperative. A guest thread that hits a blocking primitive yields
  to the scheduler, which runs another ready thread, exactly like an OS switching
  threads on a core. A quantum (fixed retired-instruction count) preempts even
  without a blocking call, so guest busy-waits cannot deadlock the worker. The
  fixed quantum keeps this deterministic in single-worker mode.
- **Realized (native, single guest thread)**: a blocking host call returns
  `SvcOutcome::Yield` (a hint; run-to-completion hosts treat it as Continue). The
  native scheduler (`vitaslop-native::Scheduler`) runs the guest on a wasmtime
  async fiber: `Yield` suspends the fiber, and wasmtime fuel
  (`fuel_async_yield_interval`) supplies the quantum. The host steps the guest one
  frame per call (`run_frame`), injecting input between frames - this is what
  drives the live desktop window. Proven bit-identical to the sync run-to-
  completion path, and preemption is transparent (see the harness
  `cube_scheduler` tests). Still future: multiple guest threads / a real run queue
  (only the main thread exists so far), and the browser mechanism (JSPI or
  Asyncify - the browser guest is its own WebAssembly instance and cannot suspend
  mid-call the way a wasmtime fiber can).
- **Locks, not spinlocks**: our primitives use shared-memory atomics with
  wait/notify. We never busy-spin the host.
- **Determinism by construction**: scheduling order and allocation addresses are
  pure functions of the guest's request sequence in single-worker mode, so neither
  is a recorded input.

## Faithful blocking and host-service workers

- **Rule**: guest-observable blocking must match the real Vita exactly. If an API
  blocks the calling thread on hardware, it blocks (suspends the coroutine) here.
  If it returns immediately there, it returns immediately here. How the host
  implements the work underneath is free, as long as that observable behavior
  holds.
- **Dedicated workers** for units that are asynchronous on real hardware, so the
  CPU does not block where hardware would not:
  - GPU: command submission returns without waiting; only explicit finish/sync
    waits block.
  - Display queue: flip is queued to a separate thread, blocks only on backpressure.
  - Audio: output deliberately blocks to pace to the audio clock. We keep that block.
  - Async IO: async calls return immediately, synchronous reads block. Match each.

## Capture and software raster

- **Capture** records the GXM command stream (surfaces, per-draw vertex/index/
  uniform snapshots taken from guest memory at draw time) without emulating a GPU
  or drawing a pixel. This is the blob-free "it works" signal.
- **Software raster** is the CPU reference renderer over that capture: transform
  each vertex by the captured MVP, perspective-divide, depth-test, interpolate
  per-vertex color. A fixed-function equivalent of the placeholder shaders, so no
  Sony shader blob is needed. A wgpu backend over the same capture comes later and
  uses this as its oracle.
- Status: implemented. The cube runs end to end and rasterizes to a spinning 3D
  cube (see the conformance harness cube_run and cube_render tests).

## Determinism seam: the World trait

- One small trait is the single choke point for external-world inputs: monotonic
  clock, wall clock, control input, touch, motion, random. IO/net added when a NID
  first needs them.
- **Semantic, not per-NID**: handlers translate NID semantics and only ask World
  for abstract inputs, so the trait stays small and stable as the NID surface
  grows.
- **Clock notions kept separate**: monotonic and wall are distinct queries;
  execution speed / pacing is not a World concern (it is a presentation-layer
  throttle). Whether the notions move together is left to the implementation, so
  we can decide coupling later.
- **Implementations and decorators**:
  - Real-time, TAS (input frames indexed by frame), mock-clock.
  - A record decorator logs every answer over any inner World; a replay reads it
    back. Bug-replay is answer-level (robust). TAS is input-level (compact,
    editable) and relies on single-worker determinism.
- **Perf**: World is called only at host-call boundaries (per frame, occasional
  clock, rare random), never per instruction or per memory access, so the
  abstraction costs nothing measurable. Keep it that way.
