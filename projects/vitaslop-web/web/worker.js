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
  set_system_font,
  worker_input_key,
  worker_input_pointer,
  worker_input_stick,
  worker_set_paused,
  worker_set_keymap,
  worker_location_fix,
  worker_location_error,
  worker_location_unavailable,
  worker_location_note,
  flush_game_data,
} from "./pkg/vitaslop_web.js";
import { openTitleCached } from "./opfs.js";
import * as gamedata from "./gamedata.js";

// >>> THE SYSTEM FONT, IF THE DEPLOYMENT SUPPLIES ONE.
//
// `sceFontOpen`/`scePvfOpen` open one of the console's own installed fonts. Those are the
// vendor's assets and are not shipped here, so the open is refused - and a title that renders
// its strings through the system font then draws them all from an EMPTY GLYPH ATLAS, which
// reaches the screen as blank or black areas where its dynamic text belongs. On the golf title
// that was an opaque black rectangle over the club list and black bars over half of the
// course-settings screen.
//
// The desktop can probe a host font path. A browser can do neither, so the bytes have to come
// from the page: drop any TTF/OTF at `web/system-font.ttf` and it is used. A 404 is a NORMAL
// outcome, not an error - the run then reports the refusal and shows no dynamic text, exactly
// as a device with no font installed would.
// >>> A RUN NOTE: onto the page's diagnostics panel always, onto the console only under
// `VITASLOP_CONSOLE=1`. The product page's console is EMPTY on a clean run; every line a
// normal run used to log is a status note and lives in the panel instead.
let consoleOn = false;
const note = (text) => {
  if (consoleOn) console.log(text);
  self.postMessage({ type: "note", text });
};

async function loadSystemFont() {
  try {
    const res = await fetch("./system-font.ttf", { cache: "force-cache" });
    if (!res.ok) {
      note("[font] no web/system-font.ttf - the title's dynamic text will be BLANK");
      return;
    }
    const bytes = new Uint8Array(await res.arrayBuffer());
    set_system_font(bytes);
    note(`[font] system-font substitute loaded, ${bytes.length} bytes`);
  } catch (err) {
    note(`[font] system-font substitute not loaded (${err}); dynamic text will be BLANK`);
  }
}

// >>> FALL BACK TO A COMPATIBILITY-MODE ADAPTER WHEN THE NORMAL ONE IS BLOCKLISTED.
//
// Placed after the imports deliberately: ES module imports HOIST, so code written above them
// still runs after they evaluate. What matters is only that this runs before the first adapter
// request, which happens when the page posts the run message.
//
// Chrome keeps a WebGPU-specific blocklist, and in June 2026 it gained an entry for the
// Imagination "ImgTec" driver v25.1 (crbug.com/520126488, CL 7952154). That driver has a real
// defect - `textureSample` returns BLACK on PowerVR B- and D-series - so the block is correct
// and must not be worked around with `enable-unsafe-webgpu`, which would put us back on a GPU
// that returns wrong pixels. It shipped in Chrome 151 and took WebGPU away from a device that
// had been running this emulator fine on Chrome 150.
//
// What the blocklist leaves standing is the COMPATIBILITY-MODE adapter, on the OpenGLES/ANGLE
// backend rather than Vulkan - the Chromium engineer who wrote the CL says so explicitly
// ("though compat mode still works"), and the device's own chrome://gpu reports that adapter as
// Available while the Vulkan one is Blocklisted.
//
// It has to be asked for: Chrome only hands out a compat adapter for an explicit
// `featureLevel: "compatibility"`. `wgpu` 30's WebGPU backend has no option for that, so the
// request is patched here, at the one boundary both wgpu and our own probe pass through. The
// shim is INERT unless the ordinary request has already failed, so a healthy device takes the
// exact path it always did and pays one extra property read at startup.
if (typeof navigator !== "undefined" && navigator.gpu && !navigator.gpu.__vitaslopCompatShim) {
  const gpu = navigator.gpu;
  const original = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async (options) => {
    const first = await original(options);
    if (first) return first;
    // Ask again in compatibility mode, keeping whatever else the caller wanted.
    const compat = await original({ ...(options || {}), featureLevel: "compatibility" });
    if (compat) {
      // The renderer has to KNOW, not infer. Compatibility mode is not a slower version of
      // WebGPU, it is a different validation regime: a texture may not carry a view of another
      // format (so no sRGB twin on a render target), and `textureLoad` is refused on depth. Code
      // that assumes the full regime does not run slowly there, it produces an invalid texture
      // and every view, bind group and render pass built on it cascades into nothing - which
      // arrives on screen as BLACK, with the cause 4,000 validation errors upstream.
      globalThis.__vitaslopWebgpuCompat = true;
      // A warning, not a note: this is a degraded mode a person should know they are in.
      console.warn(
        "[gpu] no ordinary WebGPU adapter (this driver is likely blocklisted); " +
          "running on a COMPATIBILITY-MODE adapter instead"
      );
    }
    return compat;
  };
  gpu.__vitaslopCompatShim = true;
}

// >>> THE PANIC SINK. A Rust panic in here is the most valuable line this emulator can emit and
// it was the one line a phone could not read.
//
// Under `panic = "abort"` the panic arrives at JS as `Uncaught RuntimeError: unreachable at
// ...vitaslop_web_bg.wasm:1:3542933` - an offset into a fat-LTO, one-codegen-unit blob, which
// resolves to nothing. The message and the `src/....rs:NNN` location exist only inside the Rust
// panic hook, which used to print them to the console. There is no console on a phone.
//
// `logging::install_panic_hook` calls this if it is defined, so the text is posted to the page
// the instant the panic happens, not on the next perf window - a panicking run publishes no
// further reports, so anything that waits for one waits for ever.
//
// Defined BEFORE `init()`: a panic during instantiation is still a panic worth reading.
globalThis.__vitaslopPanic = (text) => {
  try {
    self.postMessage({ type: "panic", message: text });
  } catch {
    // A panic hook that throws replaces a diagnosable crash with an undiagnosable one.
  }
};

// Start loading the wasm module immediately; the first message awaits it.
const ready = init();

// Where this run's saves go, once the start message names a title. Held at module scope so
// the page's `flush-game-data` message can reach it - that arrives on the way out, long
// after the start message has returned.
let saveSink = null;

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
  // The page's hard pause (tab hidden / window blurred) - see live.html. The live loop
  // reads it at the top of every tick and runs no guest frame while it is set.
  if (d.type === "pause") {
    worker_set_paused(!!d.paused);
    return;
  }
  // The person's keyboard map (see vitaslop-frontend); the on-screen pad and a gamepad
  // post keyboard codes too, so this one table serves all three.
  if (d.type === "keymap") {
    try {
      await ready;
      worker_set_keymap(String(d.json));
    } catch (err) {
      self.postMessage({ type: "error", message: `keymap rejected: ${err}` });
    }
    return;
  }

  // Position from the page's watchPosition (see web/location.js). `navigator.geolocation`
  // is Window-only, so the page owns the API and this worker only receives its answers.
  //
  // The nullable fields are passed through UNCHANGED - `null` and `NaN` both mean the
  // browser could not supply that component, and the Rust side turns either into the
  // guest's INVALID sentinel. Substituting a 0 here would be a heading of due north and a
  // speed of standing still, which is a measurement the device never made.
  if (d.type === "location-fix") {
    worker_location_fix(
      d.latitude,
      d.longitude,
      d.altitude ?? undefined,
      d.accuracy ?? undefined,
      d.heading ?? undefined,
      d.speed ?? undefined,
      d.timestamp
    );
    return;
  }
  if (d.type === "location-error") {
    worker_location_error(d.code);
    return;
  }
  if (d.type === "location-unavailable") {
    worker_location_unavailable();
    return;
  }
  // A note from the page's relay. It goes through the wasm logger so it reaches the
  // on-page WARN mirror and the /diag sink - a phone has no console to read.
  if (d.type === "location-note") {
    worker_location_note(String(d.message));
    return;
  }

  // >>> THE PAGE IS GOING AWAY: GET THE SAVE OUT NOW.
  //
  // The run writes the save on a 3-second floor, so a tab closed just after the guest saved
  // would otherwise lose that write. `flush_game_data` returns the container only if there
  // is something unwritten AND the guest is not mid-host-call, so this is a no-op on the
  // common path and cannot block the worker on the way out.
  if (d.type === "flush-game-data") {
    if (!saveSink) return;
    try {
      const bytes = flush_game_data();
      if (bytes) saveSink(bytes);
    } catch (err) {
      self.postMessage({ type: "error", message: `game data flush failed: ${err}` });
    }
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
  const { offscreen, titleId, files, recipe, maxFrames, knobs, prebuilt, audioRing, profile } = d;
  // Which game-data profile this run saves into; the page picked it from the settings.
  gamedata.setProfile(profile || "default");
  try {
    await ready;
    // A worker is its own wasm instance, so it needs the knobs set here, not on the page.
    for (const [k, v] of Object.entries(knobs || {})) set_knob(k, String(v));
    consoleOn = String((knobs || {}).VITASLOP_CONSOLE ?? "") === "1";
    await loadSystemFont();
    // Forward each (id, text) metric the run publishes to the page.
    const report = (id, text) => self.postMessage({ type: "report", id, text });
    // The title's files are opened on a STORAGE WORKER (see storage-worker.js), which
    // serves them into a shared page ring and reads ahead between requests. That open is
    // asynchronous, which is why it happens now, before any guest code runs. Once it is
    // up, every read the emulator makes is a plain synchronous call - out of shared memory
    // on a hit, one `Atomics.wait` on a miss - which is what a guest file read inside a
    // host call requires.
    let source;
    if (titleId) {
      source = { kind: "opfs", payload: await openTitleCached(titleId) };
    } else if (files) {
      source = { kind: "memory", payload: files };
    } else {
      throw new Error("start message names neither titleId (OPFS) nor files (in-memory)");
    }
    // >>> THE GUEST'S OWN SAVED STATE, READ BACK BEFORE IT RUNS.
    //
    // Read here, asynchronously, for the same reason the title's handles are opened here:
    // by the time guest code executes there is no await left to take. The emulator is
    // handed the bytes and hands back new ones; only this file knows where they live
    // (`gamedata/<titleId>/`, which is not where the title lives - see gamedata.js).
    //
    // Only for an OPFS run. The in-memory `files` path is the e2e fixture path and has no
    // title id to key storage by, so it plays without persistence rather than guessing one.
    let persist;
    if (titleId) {
      const stored = await gamedata.read(titleId);
      if (stored) note(`[gamedata] restoring ${stored.length} bytes for ${titleId}`);
      saveSink = gamedata.sink(titleId, (err) => {
        // The emulator reports the failure on the diagnostics panel; this reaches the page's
        // status line as well, because losing progress is not a console-only event.
        self.postMessage({
          type: "error",
          message:
            `this title's save could not be written to this browser's storage (${err}). ` +
            `Play continues, but progress from here will be lost when the tab closes.`,
        });
      });
      persist = { data: stored ?? null, save: saveSink, titleId };
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
      audioRing ?? undefined,
      persist
    );
    self.postMessage({ type: "setup", status });
  } catch (err) {
    self.postMessage({ type: "error", message: String((err && err.message) || err) });
  }
};
