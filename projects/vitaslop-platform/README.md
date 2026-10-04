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

Understood, measured, deliberately NOT patched around. Read before chasing "shader hitches".

- **Symptom.** The picture holds ~0.5-1.2 s on a phone (~0.1-0.25 s on desktop) when a title
  first draws a batch of new shader pairs (an intro, a round start). Draws are never skipped.
- **Cause.** Not translation (patcher-named pairs are translated at load, as the hardware's
  patcher does). It is the DRIVER compile of a whole PIPELINE: depth, stencil, cull, blend,
  target format and vertex layout are baked in, where the Vita applies them per draw - so the
  state is first known at the draw. Mobile compilers are slow and translated shaders are large.
- **Scale.** PowerVR/Chrome: 5-40 ms per plain pair, 100-180 ms skinned, ~2 in parallel, no
  reuse across state; one fighting title's intro needs 21-32 new pipelines.
- **Rejected.** Guessing state at load (27-62% hit rate; misses steal the compiler); WGSL
  emission tweaks (near noise); a persistent cache of our own (helps only a second play, and a
  run must not depend on the one before).
- **Real fixes (heavy).** An interpreter ("ubershader") pipeline per depth/stencil/blend/format
  variant that draws any pair from data while the real one compiles (Dolphin's approach); or
  WebGPU dynamic depth/stencil/blend state (gpuweb/gpuweb#4014 - open, milestone "4+", no
  implementer as of 2026-10), which would make building at the patcher call exact. Revisit then.
