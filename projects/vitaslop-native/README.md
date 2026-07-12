# vitaslop-native

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

OS-level mechanisms shared by every non-browser host (desktop and, someday,
mobile): the wasm engine that runs the transpiler's output, plus worker threads
and the mmap image source. Never built for wasm32. The browser supplies its own
equivalents; the engine-agnostic core lives in vitaslop-runtime.

## Host engine (exists today)

- Transpile a module to WASM, instantiate it, seed guest memory and registers,
  then call a guest function by address. Read back memory, registers, and flags.
- Two host imports back the guest module: an svc trap (ARM svc imm) and an import
  trap (NID index). A handler receives the selector, the guest registers, memory,
  and the base, and writes return values back into registers. Exit is a returned
  trap that unwinds the run.
- This is the reusable host used by the conformance harness and is the mechanism
  the GXM-recording host will build on.

## Workers and the CPU/host-service split (design)

See vitaslop-runtime for the concurrency model, ABI, and World trait. This crate
owns the native realization of it.

- **Two runtime modes** (a config choice, shared memory in both):
  - Single-worker cooperative: all guest threads are coroutines on one worker,
    scheduled by us, fully deterministic and step-debuggable.
  - Multi-worker: guest cores run in parallel across workers. Faithful to
    hardware, but replay is not fully deterministic (accepted).
- **Host-service workers**: GPU, display queue, audio, and async IO run on their
  own workers because they are asynchronous units on real hardware. Keeping them
  off the CPU worker is what lets guest-observable blocking stay faithful (the CPU
  blocks only where the real Vita blocks).
- **Locks, not spinlocks**: guest synchronization and our scheduler use
  shared-memory atomics with wait/notify, never busy-spin. Guest code that
  hand-rolls a busy-wait is handled by the scheduler's quantum preemption, not by
  spinning the host.
- **Image source**: mmap-backed guest image on native (the browser uses its own
  asset storage via the platform seam).
