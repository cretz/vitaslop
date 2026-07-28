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
