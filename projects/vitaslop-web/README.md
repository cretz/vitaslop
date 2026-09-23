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
- Each route makes a new `current` object; a screen's renderer captures it and stops
  after its awaits if the router has moved on, so a slow screen never paints over the
  next one (and its failure never paints the error card there).
- Settings are one record (`vitaslop-frontend`); the global one is stored whole, a
  title stores only its patch. `store.js` keeps both in localStorage and the library
  records (`library/<id>/meta.json` + images) in OPFS beside the titles (`games/<id>/`).
- Importing streams: the page hands the picked `File`s to `import-worker.js`, which
  reads ranges with `FileReaderSync` and writes OPFS sync handles while the Rust
  streaming ingest peels zip/pkg/PFS/SELF. Nothing is ever resident.
- The old debug pages (`live.html`, the cube, conformance) live under `web/debug/`
  and the e2e rigs drive them there.

## The player

- A canvas hands its drawing to a worker once, so each run gets a fresh `#screen`
  element in place of the last (`cloneNode`); no reload between games.
- Fullscreen and the landscape lock are asked for once per run, from a gesture (Play,
  or the menu button), never from a resize/orientation/focus/visibility handler, and
  not again if already fullscreen. Chrome for Android shows its "swipe down to exit"
  toast on every fullscreen layout and every window-focus regain while fullscreen
  (`FullscreenHtmlApiHandlerBase.onWindowFocusChanged`), which the page cannot
  suppress; it can only avoid causing them. A screen wake lock is held while a run is
  on (re-asked on each return to the foreground, released on stop) so the screen does
  not dim and wake into one.
- The in-game menu's Restart and Quit go through a Yes/No panel inside the menu, not
  `window.confirm`, which a fullscreen phone browser hides. Restart tears the run down
  (`stop(false)`: everything but leaving the screen) and the app starts the same title
  through `play()`, the one start path.

## Controllers

- In a game `gamepad.js` owns the pad: mapped buttons post their keyboard codes to the
  worker. The `home` control (index 16, exposed by some pads and browsers) opens the
  menu unless a Vita button is mapped to it; start+select held for a second opens it
  on any pad (the game sees a short press of each first).
- Outside a game, and inside the menu while it is open, `navpad.js` owns the pad: one
  focus-based navigator over one root (the document, or the menu). D-pad or left stick
  moves focus spatially with DOM order as the fallback, south picks, east goes back
  (`history.back`, or the menu's own back), start opens the settings (resumes, in the
  menu), left/right change a select, checkbox or slider. The ring is the `navfocus`
  class, not `:focus-visible`.
- Hand-over is exclusive and edge-safe: the menu suspends `gamepad.js` (releasing
  everything held) before the navigator attaches, and each side ignores whatever is
  still down at its hand-over until it is released, so the press that opened or
  closed the menu is seen by one owner only.

## The guest engine

- The guest's transpiled wasm runs on the *browser's* engine, not on an
  interpreter shipped inside the wasm. That is what makes browser performance
  worth measuring at all.
- The scheduler is one worker instance-per-thread over JSPI, so a guest thread
  can block without blocking the page.
- The guest's memory is a region INSIDE the run worker's own linear memory: the page
  asks the run worker to reserve it first (`reserve` message), the throwaway transpile
  worker builds the module for that offset (`hostOff`), and every guest instance imports
  the emulator's memory. A host read of guest memory is then a load, not a JavaScript
  call - which was several crossings per draw and the phone's biggest CPU item.
  `VITASLOP_BROWSER_SPLIT_MEMORY=1` restores a separate, exactly-sized guest memory
  (a wild guest pointer traps there instead of reaching the emulator's heap).

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
