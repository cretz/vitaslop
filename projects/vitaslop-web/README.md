# vitaslop-web

> Keep this README terse: sectioned and bulleted, not large prose. Explain
> concepts and the why, not exact type names. Update it as the code changes so it
> never goes stale.

The browser app: a wasm-bindgen cdylib that runs the whole path client-side -
load the guest executable, transpile ARM/Thumb/VFP to wasm, run it on the
browser's own WebAssembly engine, capture the GXM stream, present through WebGPU.

**No Sony blob and no server.** That is the point of the project, and this crate
is where it is either true or not.

## Shape

- The whole crate is gated to `wasm32`. On a native build it is empty, so a
  workspace build never drags the browser stack onto the desktop toolchain.
- Build it with the wasm target explicitly; the `wasm-bindgen` crate version and
  the installed CLI must match exactly (the CLI reads a schema the macro embeds),
  so bump both together.
- Implements the platform seam for the browser: OPFS-backed storage, and
  pointer/keyboard events on the canvas mapped to the touch panel and SceCtrl.

## The front end

- `web/index.html` + `app.js` is the product: a hash-routed library (`#/`), title page
  (`#/title/<id>`), settings (`#/settings[/<id>]`), import (`#/import`) and the player.
  Plain ES modules and CSS, no framework, no bundler; GitHub Pages is the target and
  `coi.js` (a service worker) supplies the cross-origin-isolation headers a static host
  cannot.
- Settings are one record (`vitaslop-frontend`); the global one is stored whole, a
  title stores only its patch. `store.js` keeps both in localStorage and the library
  records (`library/<id>/meta.json` + images) in OPFS beside the titles (`games/<id>/`).
- Importing streams: the page hands the picked `File`s to `import-worker.js`, which
  reads ranges with `FileReaderSync` and writes OPFS sync handles while the Rust
  streaming ingest peels zip/pkg/PFS/SELF. Nothing is ever resident.
- The old debug pages (`live.html`, the cube, conformance) live under `web/debug/`
  and the e2e rigs drive them there.

## The guest engine

- The guest's transpiled wasm runs on the *browser's* engine, not on an
  interpreter shipped inside the wasm. That is what makes browser performance
  worth measuring at all.
- The scheduler is one worker instance-per-thread over JSPI, so a guest thread
  can block without blocking the page.

## Conformance parity

- The browser runs the **same committed ARM corpus** as the native test, through
  the browser engine. Two engines agreeing on a corpus is the only evidence that
  they are the same emulator; a browser-only smoke test is not.
- The summary is serialised back to the page, so a headless browser run is
  machine-checkable.

## Rendering

- Presents through the shared pipeline in `vitaslop-platform`, the same one the
  native headless oracle uses, so a browser frame and a native frame are
  comparable by construction.
