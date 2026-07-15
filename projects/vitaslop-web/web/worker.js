// The production run home: a module Web Worker that runs the whole emulator - the JSPI
// scheduler, the guest, and the WebGPU render - off the main thread. A worker allows
// synchronous instantiation of the title's large transpiled module at any size, so this
// path needs no WebAssemblyUnlimitedSyncCompilation flag (the one main-thread caveat).
//
// The page transfers an OffscreenCanvas (the on-page <canvas>'s render control) and the
// fetched container bytes to this worker, which renders straight to that canvas. Since a
// worker has no DOM, metrics come back as { type: "report", id, text } messages the page
// applies to its FPS/status elements.
import init, {
  run_game_worker,
  worker_input_key,
  worker_input_pointer,
} from "./pkg/vitaslop_web.js";

// Start loading the wasm module immediately; the first message awaits it.
const ready = init();

self.onmessage = async (e) => {
  const d = e.data;
  // Live input forwarded from the page (keyboard/pointer). These arrive after the run
  // has started; the wasm has its shared input cell registered by then.
  if (d.type === "key") {
    worker_input_key(d.code, d.pressed);
    return;
  }
  if (d.type === "pointer") {
    worker_input_pointer(d.x, d.y, d.down);
    return;
  }

  // Otherwise this is the start message: { offscreen, files, recipe, maxFrames }.
  const { offscreen, files, recipe, maxFrames } = d;
  try {
    await ready;
    // Forward each (id, text) metric the run publishes to the page.
    const report = (id, text) => self.postMessage({ type: "report", id, text });
    const status = await run_game_worker(offscreen, files, recipe || "", maxFrames, report);
    self.postMessage({ type: "setup", status });
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};
