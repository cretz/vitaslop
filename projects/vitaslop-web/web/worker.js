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
  set_knob,
  worker_input_key,
  worker_input_pointer,
  worker_input_stick,
} from "./pkg/vitaslop_web.js";
import { openTitleSync, syncReader } from "./opfs.js";

// Start loading the wasm module immediately; the first message awaits it.
const ready = init();

// The live loop runs as a DETACHED future (`spawn_local`), so nothing it throws reaches
// the try/catch around the start message - it surfaces as an unhandled rejection, or as a
// worker-level error, and by default neither is reported anywhere. A worker that dies
// that way simply stops, and from the page it is indistinguishable from a run that is
// merely slow. Both are forwarded so the failure names itself.
self.addEventListener("error", (e) =>
  self.postMessage({ type: "error", message: `worker error: ${e.message || e}` })
);
self.addEventListener("unhandledrejection", (e) =>
  self.postMessage({
    type: "error",
    message: `worker unhandled rejection: ${(e.reason && (e.reason.stack || e.reason.message)) || e.reason}`,
  })
);

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
  // An analog stick, in the guest's own 0..255 encoding (128 centred). `active: false`
  // releases it back to whatever a scripted recipe says, which is NOT the same as sending
  // 128 - see InputState::left_stick.
  if (d.type === "stick") {
    worker_input_stick(d.stick, d.x, d.y, d.active);
    return;
  }

  // Otherwise this is the start message:
  // { offscreen, titleId | files, recipe, maxFrames, knobs }.
  //
  // `titleId` is the OPFS path: the title was imported once (streamed to storage, never
  // fully resident) and is read a piece at a time from here on. `files` is the older
  // in-memory form, kept for fixtures - a retail container cannot use it, because holding
  // it in JS and again in the wasm heap exceeds the wasm32 address space.
  // `audioRing` is the SharedArrayBuffer the page's AudioWorklet drains (see
  // web/audio.js). It is shared, not transferred, so it needs no transfer list - and a
  // start message without one simply runs silent, which the setup says on the console.
  const { offscreen, titleId, files, recipe, maxFrames, knobs, prebuilt, audioRing } = d;
  try {
    await ready;
    // A worker is its own wasm instance, so it needs the knobs set here, not on the page.
    for (const [k, v] of Object.entries(knobs || {})) set_knob(k, String(v));
    // Forward each (id, text) metric the run publishes to the page.
    const report = (id, text) => self.postMessage({ type: "report", id, text });
    // Sync access handles can only be opened here (Workers only) and only
    // asynchronously - which is precisely why it happens now, before any guest code
    // runs. Once open, every read the emulator makes is a plain synchronous call, which
    // is what a guest file read inside a host call requires.
    let source;
    if (titleId) {
      source = { kind: "opfs", payload: syncReader(await openTitleSync(titleId)) };
    } else if (files) {
      source = { kind: "memory", payload: files };
    } else {
      throw new Error("start message names neither titleId (OPFS) nor files (in-memory)");
    }
    // `prebuilt` is the module the throwaway transpile worker already produced. Running
    // against it keeps this worker's heap at the mounted-and-linked size (~24 MB) instead
    // of the transpiled size (~487 MB) it could never give back.
    const status = await run_game_worker(
      offscreen,
      source,
      recipe || "",
      maxFrames,
      report,
      prebuilt ?? undefined,
      audioRing ?? undefined
    );
    self.postMessage({ type: "setup", status });
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};
