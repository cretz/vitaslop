# vitaslop-platform

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

The web/native seam: the trait contracts a frontend implements, plus the shared
GPU render path both frontends present through. The browser is the target; the
desktop app exists to debug the same code against a real debugger and a real GPU,
which only holds if the two differ in their trait impls and nothing else.

## The seam

- **Storage** is async random access because the browser's is (OPFS on web, mmap
  on desktop). The awkward side wins the API - a synchronous read would work
  natively and be impossible in a browser.
- **Input** and **audio** are trait contracts here; impls live in the frontends
  and are injected at startup.
- Renderer and window are deliberately not abstracted - wgpu and winit already
  span web and native, and a wrapper would just be a layer to keep in sync.

## Dependency posture

- Stays light: no wasm-bindgen, no js-sys, no OS crates. The engine-agnostic
  runtime depends on this for the neutral types, so anything heavy here lands in
  every build.
- The GPU stack is therefore behind a `gpu` feature, off by default. Frontends
  enable it; the runtime does not.

## `gpu`: the shared render path

- A neutral draw-batch type is the currency from capture to renderer, so the
  renderer never sees guest structures.
- `GxmRenderer` is the real path: it links a guest vertex+fragment shader pair
  through the clean-room GXP recompiler and draws with the title's own shading.
- Fixed-function is the **fallback only**, and a pair that falls back **reports
  itself unconditionally**. A silent fallback looks like a rendering bug forever
  after.
- The recompiler's lib half is wasm-safe, which keeps the browser on the same
  path rather than on a permanent fallback.

## Known limitation: first-draw pipeline compile stalls

Understood, measured, and deliberately NOT patched around. Read this before
spending effort on "shader hitches".

- **What it looks like.** The display freezes ~0.5-1.2 s on a phone when a
  title first draws a batch of new shader pairs (a fighter intro, a round
  start, a new course). The guest keeps running; presents are declined until
  the pipelines exist (async compile, never a skipped draw).
- **Why it happens.**
  - Translation is NOT the cost: patcher-named pairs are already translated to
    WGSL at load, where the hardware's patcher does its work.
  - The cost is the DRIVER compile, and WebGPU only compiles a whole
    PIPELINE: depth compare/write, stencil, cull, blend, target format and
    vertex layout are baked in. On the Vita those are per-draw state the GPU
    applies independently, so a Vita shader is finished at load; here the
    state is first known at the draw. Only viewport, scissor, blend constant
    and stencil reference are dynamic in WebGPU.
  - Mobile drivers are slow compilers, and translated shaders are large
    (a skinned vertex program is ~650 lines of register-file WGSL).
- **Measured (PowerVR phone, Chrome).** Plain pairs 5-40 ms, skinned
  100-180 ms per FIRST compile; ~2 compiles run concurrently; a second
  pipeline over an already-compiled pair with different depth/cull/blend
  costs ~85% of a full one (no reuse across state); even an empty vertex
  shader leaves a ~35-47 ms floor per pipeline. One retail fighting title puts
  21-32 new pipelines into its intro: ~1.1 s frozen.
  - Chrome caches compiled pipelines across page loads for an identical
    text, so a bench that recompiles a known module reads 14-25 ms; only a
    FIRST compile is a measurement. It did not help live play in a repeat run.
- **Rejected, with reasons.**
  - Speculative load-time pipelines (guess the state): 27-62% hit rate across
    two measurements; the misses are wasted ~135 ms compiles on a two-slot
    compiler, competing with the real ones. Not Vita-faithful work.
  - WGSL emission tweaks: scalarised register arrays, unrolled copies,
    dropped index guards - no gain. Removing dynamic register indexing is
    the only cut found (~10-20%, near noise). Bounding dynamically-read banks
    (already shipped) was the one worthwhile one.
  - A persistent cache of our own: helps only a second play; ruled out.
- **How it could actually be solved (heavy).**
  - An interpreter ("ubershader") pipeline, as Dolphin does: one prebuilt
    pipeline per depth/stencil/blend/format variant interprets any USSE pair
    from data, so a new pair draws at once while its real pipeline compiles
    in the background, then switches. Removes the stall without guessing or
    skipping; costs a large project and slower GPU time on interim frames.
  - Making the remaining baked state runtime: cull (discard on
    `front_facing`) and vertex layout (vertex pulling) can move into the
    shader today, but on the measured title they account for almost none of
    the variation - depth/stencil is what is unknown at load, and WebGPU
    cannot make that dynamic.
- **Upstream.** gpuweb "extending dynamic state" (gpuweb/gpuweb#4014, the
  `VK_EXT_extended_dynamic_state` analogue covering depth, stencil and blend)
  is open, labelled `large`, milestone "4+", with no implementer commitment as
  of 2026-10. If it ships, building pipelines at the patcher call becomes
  exact and faithful - revisit then.
