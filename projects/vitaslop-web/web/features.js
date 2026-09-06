// What THIS browser can do, found out by asking it - never by reading its name.
//
// Every check is a feature test with a one-line reason and a fix a person can act on.
// Three of them are properties of the ORIGIN rather than the browser (secure context,
// cross-origin isolation, storage), and all three are invisible on localhost, which is
// itself secure; the first phone to open the LAN address failed inside a gigabyte
// import with a property access on `undefined`. This runs before anything else and
// the result is on the page.
//
// `fatal` blocks play. `warn` degrades (silent audio, no vibration). `ok` is the rest.

/// Registers the isolation service worker when the page is not isolated and reloads
/// once. Returns true if a reload was started (the caller should stop).
export async function ensureIsolation() {
  if (self.crossOriginIsolated) return false;
  if (!("serviceWorker" in navigator)) return false;
  if (!isSecureContext) return false;
  // One attempt per page load, remembered in the session so a host that strips the
  // headers anyway (or a worker that cannot install) does not loop.
  const KEY = "vitaslop.coi.tried";
  try {
    if (sessionStorage.getItem(KEY)) return false;
    sessionStorage.setItem(KEY, "1");
  } catch {
    return false;
  }
  try {
    const reg = await navigator.serviceWorker.register("./coi.js", { scope: "./" });
    await navigator.serviceWorker.ready;
    if (reg.active) {
      location.reload();
      return true;
    }
  } catch {
    // Not registrable here; the isolation check below will say so.
  }
  return false;
}

/// A tiny worker that tries the one OPFS call the emulator needs and cannot make on
/// the main thread: a synchronous access handle.
function probeSyncHandles() {
  return new Promise((resolve) => {
    let w;
    try {
      w = new Worker("./probe-worker.js");
    } catch (e) {
      resolve({ ok: false, err: String(e) });
      return;
    }
    const t = setTimeout(() => {
      w.terminate();
      resolve({ ok: false, err: "timed out" });
    }, 5000);
    w.onmessage = (e) => {
      clearTimeout(t);
      w.terminate();
      resolve(e.data);
    };
    w.onerror = (e) => {
      clearTimeout(t);
      w.terminate();
      resolve({ ok: false, err: e.message || "worker failed" });
    };
    w.postMessage(0);
  });
}

/// One entry per capability: { id, level: "ok"|"warn"|"fatal", text, fix }.
export async function checkFeatures() {
  const out = [];
  const add = (id, level, text, fix = "") => out.push({ id, level, text, fix });

  if (!isSecureContext) {
    add(
      "secure",
      "fatal",
      "not a secure context",
      "open this page over https:// (or on localhost). Storage and WebGPU are only offered on a secure origin."
    );
  } else add("secure", "ok", "secure context");

  if (!("WebAssembly" in self)) add("wasm", "fatal", "no WebAssembly", "use a current browser.");
  else if (typeof WebAssembly.Suspending !== "function") {
    add(
      "jspi",
      "fatal",
      "no JavaScript Promise Integration",
      "the emulator's scheduler needs WebAssembly JSPI. Chrome 137+, Edge 137+ and Firefox 139+ have it; Safari 26 is expected to. On Android, Chrome 137+."
    );
  } else add("jspi", "ok", "WebAssembly + JSPI");

  if (!("gpu" in navigator)) {
    add("webgpu", "fatal", "no WebGPU", "use a browser with WebGPU: Chrome/Edge 113+, Firefox 141+, Safari 26+.");
  } else {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) add("webgpu", "fatal", "WebGPU present but no adapter", "the GPU driver is blocklisted or unavailable here.");
      else {
        let name = "";
        try {
          const info = adapter.info || (adapter.requestAdapterInfo && (await adapter.requestAdapterInfo()));
          if (info) name = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" ");
        } catch {}
        add("webgpu", "ok", `WebGPU${name ? ": " + name : ""}`);
      }
    } catch (e) {
      add("webgpu", "fatal", "WebGPU adapter request failed", String(e && e.message ? e.message : e));
    }
  }

  if (typeof SharedArrayBuffer === "undefined" || !self.crossOriginIsolated) {
    add(
      "isolation",
      "warn",
      "not cross-origin isolated (no shared memory)",
      "the page tried to install a service worker that adds the isolation headers; if this stays, the host is stripping them. Audio will be silent and title storage falls back to a slower path."
    );
  } else add("isolation", "ok", "cross-origin isolated (shared memory)");

  if (!(navigator.storage && navigator.storage.getDirectory)) {
    add("opfs", "fatal", "no origin-private file system", "titles are stored in the browser's private file system; this browser does not offer one here.");
  } else {
    const r = await probeSyncHandles();
    if (r.ok) add("opfs", "ok", "private file system with synchronous access");
    else add("opfs", "fatal", "private file system without synchronous access", `the emulator reads a title with synchronous handles inside a worker; this browser refused (${r.err}).`);
  }

  if (!("OffscreenCanvas" in self) || !HTMLCanvasElement.prototype.transferControlToOffscreen) {
    add("offscreen", "fatal", "no OffscreenCanvas", "the emulator draws from a worker; this browser cannot hand it the canvas.");
  } else add("offscreen", "ok", "OffscreenCanvas");

  if (!("AudioContext" in self) || !("AudioWorklet" in self)) add("audio", "warn", "no AudioWorklet", "sound will be silent.");
  else add("audio", "ok", "AudioWorklet");

  if (!navigator.getGamepads) add("gamepad", "warn", "no Gamepad API", "controllers will not be seen; keyboard and touch still work.");
  else add("gamepad", "ok", "Gamepad API");

  add("fullscreen", document.fullscreenEnabled || document.webkitFullscreenEnabled ? "ok" : "warn",
    document.fullscreenEnabled || document.webkitFullscreenEnabled ? "fullscreen" : "no fullscreen API",
    "on iPhone, add the page to the home screen to play full screen.");

  return out;
}

export const canPlay = (features) => !features.some((f) => f.level === "fatal");
