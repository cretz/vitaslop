// The production run home: a module Web Worker that runs the whole emulator - the JSPI
// scheduler, the guest, and the WebGPU render - off the main thread. A worker allows
// synchronous instantiation of the title's large transpiled module at any size, so this
// path needs no WebAssemblyUnlimitedSyncCompilation flag (the one main-thread caveat).
//
// The page transfers an OffscreenCanvas (the on-page <canvas>'s render control) and the
// fetched container bytes to this worker, which renders straight to that canvas. Since a
// worker has no DOM, metrics come back as { type: "report", id, text } messages the page
// applies to its FPS/status elements.
//
// >>> WHICH BUNDLE: `worker.js?smp=1` loads `pkg-threads` (wasm threads, a shared memory -
// what `VITASLOP_SMP=1` needs to run guest threads on several workers); anything else loads
// `pkg`, the single-threaded build the default engine has always run on. The page decides,
// because it has to decide BEFORE this worker's first line: the bundle is this worker's code.
// See build.mjs for why there are two.
const SMP_BUNDLE = new URL(self.location.href).searchParams.get("smp") === "1";
let run_game_worker,
  set_knob,
  set_system_font,
  worker_input_key,
  worker_input_pointer,
  worker_input_stick,
  worker_set_paused,
  worker_set_output_size,
  worker_set_keymap,
  worker_location_fix,
  worker_location_error,
  worker_location_unavailable,
  worker_location_note,
  flush_game_data,
  reserve_guest_region;
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

// >>> THE SMP TIMELINE (`VITASLOP_SMP_TRACE`) IS PRINTED BY THE WASM TO THE CONSOLE, which a
// phone does not have. Its lines are posted to the page as notes as well, so a device run
// (web/runner/live.js) brings the timeline back with the rest of the diagnostics.
// `forwardConsole` in the start message (a device-runner live job's `params.console`) forwards
// EVERY console line as well, capped, so any trace knob can be read off a phone.
let forwardConsole = 0;
for (const level of ["log", "info", "warn", "error"]) {
  const orig = console[level].bind(console);
  console[level] = (...a) => {
    const text = a.map((x) => (typeof x === "string" ? x : String(x))).join(" ");
    // `rtt probe:` = `VITASLOP_RTT_PROBE_LOG`'s per-frame line (reaches the console under VITASLOP_CONSOLE=1).
    if (text.startsWith("smptrace") || text.startsWith("presentlog") || text.startsWith("drawnote") || text.startsWith("presentshot") || text.includes("rtt probe:") || text.includes("host write watch") || text.startsWith("jsprofile") || text.startsWith("guestprof")) self.postMessage({ type: "note", text });
    else if (forwardConsole > 0) {
      forwardConsole -= 1;
      self.postMessage({ type: "note", text: `console.${level}: ${text.slice(0, 60000)}` });
    }
    orig(...a);
  };
}

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

// Start loading the glue immediately; the bundle itself is initialised by the first message
// that needs it - see `bundle`.
// `loadTimes`: where the worker's start-up went (import of the glue, then `init` = fetch +
// compile + instantiate of the bundle), reported with the reservation.
const loadTimes = { t0: performance.now() };
const glue = import(SMP_BUNDLE ? "./pkg-threads/vitaslop_web.js" : "./pkg/vitaslop_web.js");
let readyP = null;
// >>> THE PAGE HANDS US THE BUNDLE ALREADY COMPILED, when it can. Each worker used to fetch and
// compile the 9 MB bundle itself, and a phone on the dev server's self-signed certificate gets
// NO HTTP cache: MEASURED (runner, MLB) 8.7 s and the full 9 MB over the wire, per worker, per
// play. The page compiles it once (web/bundle.js) and posts the `WebAssembly.Module` with the
// reserve; the first caller decides, and a page that sends none (the debug pages) gets the old
// self-fetch.
const bundle = (module) =>
  (readyP ??= glue.then(async (m) => {
    loadTimes.imported = performance.now();
    loadTimes.given = !!module;
    ({
      run_game_worker,
      set_knob,
      set_system_font,
      worker_input_key,
      worker_input_pointer,
      worker_input_stick,
      worker_set_paused,
      worker_set_output_size,
      worker_set_keymap,
      worker_location_fix,
      worker_location_error,
      worker_location_unavailable,
      worker_location_note,
      flush_game_data,
      reserve_guest_region,
    } = m);
    await m.default(module ? { module_or_path: module } : undefined);
    loadTimes.inited = performance.now();
  }));

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
  self.postMessage({
    type: "error",
    message:
      `worker error: ${e.message || e}` +
      (e.filename ? ` at ${e.filename}:${e.lineno || "?"}:${e.colno || "?"}` : "") +
      (e.error && e.error.stack ? `
${e.error.stack}` : ""),
  })
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
  // The page's hard pause (tab hidden / window blurred) - see player.js. The live loop
  // reads it at the top of every tick and runs no guest frame while it is set.
  if (d.type === "pause") {
    worker_set_paused(!!d.paused);
    return;
  }
  // The canvas's size in DEVICE pixels (see player.js `postOutputSize`): the renderer sizes the
  // canvas to it and scales the 960x544 picture into it crisply - see present_scale.rs.
  if (d.type === "output-size") {
    if (worker_set_output_size) worker_set_output_size(d.w | 0, d.h | 0);
    return;
  }
  // The person's keyboard map (see vitaslop-frontend); the on-screen pad and a gamepad
  // post keyboard codes too, so this one table serves all three.
  if (d.type === "keymap") {
    try {
      await bundle();
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
  // >>> RESERVE THE GUEST'S MEMORY INSIDE THIS WORKER'S OWN, before anything is transpiled.
  //
  // The guest region lives in the emulator's linear memory (a host read of guest memory
  // is then a load, not a JavaScript call - see `browser_sched::HostRegion`), and the
  // transpiled module is emitted for the region's exact offset. So the page asks THIS
  // worker for the offset first, hands it to the throwaway transpile worker, and only then
  // sends the start message. The knobs are set first: `VITASLOP_BROWSER_SPLIT_MEMORY` is
  // read by the reservation itself.
  if (d.type === "reserve") {
    try {
      await bundle(d.module);
      for (const [k, v] of Object.entries(d.knobs || {})) set_knob(k, String(v));
      const tr = performance.now();
      const hostOff = reserve_guest_region();
      const lt = loadTimes;
      // The bundle's own fetch, from resource timing: `transferSize` 0 means it came from the
      // cache (a 304 or a fresh entry), so the rest of `init` is compile + instantiate.
      const wasm = performance.getEntriesByType("resource").find((r) => /_bg\.wasm/.test(r.name));
      const fetchNote = wasm
        ? `wasm fetch ${Math.round(wasm.responseEnd - wasm.startTime)} ms (${Math.round(wasm.transferSize / 1024)} KB over the wire), then ${Math.round(lt.inited - wasm.responseEnd)} ms to ready`
        : lt.given
          ? "bundle handed over compiled by the page"
          : "wasm fetch not in resource timing";
      const split = `import ${Math.round(lt.imported - lt.t0)} ms, init (fetch+compile+instantiate) ${Math.round(lt.inited - lt.imported)} ms [${fetchNote}], reserve ${Math.round(performance.now() - tr)} ms`;
      self.postMessage({ type: "reserved", hostOff, split });
    } catch (err) {
      self.postMessage({ type: "error", message: `guest region: ${String((err && err.message) || err)}` });
    }
    return;
  }

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
  const { offscreen, titleId, files, recipe, maxFrames, knobs, prebuilt, audioRing, profile, noPersist } = d;
  forwardConsole = Number(d.forwardConsole || 0);
  // Which game-data profile this run saves into; the page picked it from the settings.
  gamedata.setProfile(profile || "default");
  try {
    await bundle();
    // A worker is its own wasm instance, so it needs the knobs set here, not on the page.
    for (const [k, v] of Object.entries(knobs || {})) set_knob(k, String(v));
    consoleOn = String((knobs || {}).VITASLOP_CONSOLE ?? "") === "1";
    await loadSystemFont();
    // Forward each (id, text) metric the run publishes to the page.
    //
    // `shots` (a device-runner live job's `params.shots`, frame numbers): the canvas as it
    // stands when the run first reports reaching each one, posted back as a JPEG data URL note
    // - the only way to SEE a phone's picture from the desktop.
    const shots = (d.shots || []).map(Number).sort((a, b) => a - b);
    const report = (id, text) => {
      self.postMessage({ type: "report", id, text });
      if (shots.length && id === "status") {
        const m = /frame (\d+)/.exec(text);
        if (m && Number(m[1]) >= shots[0]) {
          const at = Number(m[1]);
          while (shots.length && shots[0] <= at) shots.shift();
          offscreen
            .convertToBlob({ type: "image/jpeg", quality: 0.8 })
            .then((b) => new Promise((ok) => { const r = new FileReader(); r.onload = () => ok(r.result); r.readAsDataURL(b); }))
            .then((url) => self.postMessage({ type: "note", text: `shot f${at} ${url}` }))
            .catch((err) => self.postMessage({ type: "note", text: `shot f${at} FAILED ${err}` }));
        }
      }
    };
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
    // `noPersist` (the device runner's reproducible runs): no saved state read, none written,
    // so every run boots as a first boot - a recipe keyed to first-boot screens stays in step.
    if (titleId && !noPersist) {
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
