// A worker whose entire purpose is to be thrown away.
//
// # Why the transpile cannot happen where the game runs
// Transpiling a retail title costs about 463 MB of transient heap (measured: 24 MB after
// link, 487 MB after transpile), and **a wasm linear memory never shrinks**. A worker that
// transpiles its own guest therefore carries that half-gigabyte for the rest of its life,
// on top of the guest's 512 MB and the machine code the engine generates for the module -
// and it was killed part-way through every long run.
//
// So the transpile happens here, and this worker is terminated afterwards. The peak dies
// with it. What crosses back is the compiled `WebAssembly.Module`, which is
// structured-cloneable, plus the two layout numbers the scheduler needs.
//
// The sync access handles are opened and CLOSED here before the run worker opens its own:
// an OPFS sync access handle is an exclusive lock, so the two workers must not hold the
// same files at once.
import init, { transpile_title, set_knob } from "./pkg/vitaslop_web.js";
import { openTitleSync, syncReader } from "./opfs.js";

const ready = init();

self.onmessage = async (e) => {
  const { titleId, files, knobs } = e.data;
  let reader = null;
  try {
    await ready;
    // The same knobs as the run worker, and not optional: `VITASLOP_BROWSER_FUEL` is baked
    // INTO the module here. A module transpiled without it would run in a worker that
    // believes it has software fuel and does not, which livelocks on the first guest loop
    // that makes no host call.
    for (const [k, v] of Object.entries(knobs || {})) set_knob(k, String(v));

    let source;
    if (titleId) {
      reader = syncReader(await openTitleSync(titleId));
      source = { kind: "opfs", payload: reader };
    } else if (files) {
      source = { kind: "memory", payload: files };
    } else {
      throw new Error("transpile worker got neither titleId nor files");
    }

    const built = await transpile_title(source);
    // Release the exclusive locks BEFORE the run worker is told to start, or its own
    // `createSyncAccessHandle` will fail on every file.
    if (reader) reader.close();
    reader = null;
    self.postMessage({ type: "built", built });
  } catch (err) {
    try {
      if (reader) reader.close();
    } catch {}
    self.postMessage({ type: "error", message: String((err && err.stack) || err) });
  }
};
