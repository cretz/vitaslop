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
import { openTitleSync, syncReader } from "./opfs.js";
import * as tcache from "./transpile-cache.js";

// The same bundle as the run worker it builds for (`?smp=1` = `pkg-threads`): the module
// imports that worker's memory, and whether it declares it SHARED is read off this bundle's
// own memory (`host_memory_is_shared`). See build.mjs and worker.js.
const SMP_BUNDLE = new URL(self.location.href).searchParams.get("smp") === "1";
let transpile_title, set_knob, transpile_settings_key;

// The panic sink, same contract as the run worker's (see web/worker.js). A panic in the
// transpiler aborts this worker before its `catch` can post anything, so without this the page
// sees only "transpile worker failed to start".
globalThis.__vitaslopPanic = (text) => {
  try {
    self.postMessage({ type: "panic", message: text });
  } catch {}
};

// The bundle, initialised from the page's compiled `WebAssembly.Module` when the message
// carries one (`bundleModule`), else fetched here - see worker.js `bundle` for why.
const glue = import(SMP_BUNDLE ? "./pkg-threads/vitaslop_web.js" : "./pkg/vitaslop_web.js");
const ready = (module) =>
  glue.then(async (m) => {
    ({ transpile_title, set_knob, transpile_settings_key } = m);
    await m.default(module ? { module_or_path: module } : undefined);
  });

self.onmessage = async (e) => {
  // `hostOff` is where the RUN worker reserved the guest region in its own memory (see
  // worker.js's "reserve" message); the module is emitted for exactly that offset. 0 means
  // a memory of the guest's own.
  const { titleId, files, knobs, hostOff, bundleModule } = e.data;
  let reader = null;
  try {
    const tStart = performance.now();
    await ready(bundleModule);
    const readyMs = Math.round(performance.now() - tStart);
    if (typeof hostOff !== "number") throw new Error("transpile worker needs the run worker's hostOff");
    // The same knobs as the run worker, and not optional: `VITASLOP_BROWSER_FUEL` is baked
    // INTO the module here. A module transpiled without it would run in a worker that
    // believes it has software fuel and does not, which livelocks on the first guest loop
    // that makes no host call.
    for (const [k, v] of Object.entries(knobs || {})) set_knob(k, String(v));

    // >>> THE TRANSPILE CACHE (transpile-cache.js): a module stored by THIS build, for these
    // emit settings and this host offset, is loaded instead of transpiled - and the game's files
    // are not even opened. `VITASLOP_TRANSPILE_CACHE=0` bypasses it (neither read nor written).
    const useCache = !!titleId && String((knobs || {}).VITASLOP_TRANSPILE_CACHE ?? "") !== "0";
    // A process the title EXEC'd (`VITASLOP_MAIN_EXEC`) is a different program from the one its
    // eboot boots, so it is cached under its own name - one namespace per executable, or the
    // launcher and the game it execs would evict each other on every boot (a miss clears the
    // namespace it looked in).
    const mainExec = String((knobs || {}).VITASLOP_MAIN_EXEC ?? "").trim();
    const variant = tcache.variantSuffix(mainExec);
    let name = null;
    let invalidated = "";
    if (useCache) {
      const tKey = performance.now();
      const build = await tcache.buildToken(SMP_BUNDLE ? "./pkg-threads/" : "./pkg/");
      name = `${build}-${transpile_settings_key()}-${hostOff.toString(16)}${variant}`;
      const swept = await tcache.sweepOtherBuilds(build);
      const hit = await tcache.lookup(titleId, name);
      if (hit) {
        const tRead = performance.now();
        const module = await WebAssembly.compile(hit.wasm);
        const tComp = performance.now();
        const m = hit.meta;
        const built = {
          module,
          hostOff,
          memPages: m.memPages,
          funcAddrs: Uint32Array.from(m.funcAddrs),
          mirrorOff: m.mirrorOff,
          dirtyOff: m.dirtyOff,
          readyMs,
          split: `transpile cache HIT: key+read ${Math.round(tRead - tKey)} ms, browser wasm compile ${Math.round(tComp - tRead)} ms, wasm ${Math.round(hit.wasm.length / 1024)} KB`,
        };
        self.postMessage({ type: "built", built });
        return;
      }
      // Invalidated or never made: whatever this title held goes BEFORE the transpile, so the
      // stale module's space is free for the new one.
      const cleared = await tcache.clearTitle(titleId, variant);
      invalidated = `${swept} module(s) from older builds and ${cleared} of this title's with other settings deleted`;
    }

    let source;
    if (titleId) {
      reader = syncReader(await openTitleSync(titleId));
      source = { kind: "opfs", payload: reader };
    } else if (files) {
      source = { kind: "memory", payload: files };
    } else {
      throw new Error("transpile worker got neither titleId nor files");
    }

    const built = await transpile_title(source, hostOff);
    built.readyMs = readyMs;
    // Release the exclusive locks BEFORE the run worker is told to start, or its own
    // `createSyncAccessHandle` will fail on every file.
    if (reader) reader.close();
    reader = null;
    const wasm = built.wasm;
    delete built.wasm;
    if (useCache) {
      // Stored after the locks are gone and before the result is posted: this worker is
      // terminated the moment the page has it. A failure to store costs only the next play.
      const tStore = performance.now();
      try {
        await tcache.store(titleId, name, wasm, {
          memPages: built.memPages,
          mirrorOff: built.mirrorOff,
          dirtyOff: built.dirtyOff,
          funcAddrs: Array.from(built.funcAddrs),
        }, variant);
        built.split = `transpile cache MISS (${invalidated}), stored in ${Math.round(performance.now() - tStore)} ms; ${built.split}`;
      } catch (err) {
        await tcache.clearTitle(titleId, variant);
        built.split = `transpile cache MISS, NOT stored (${err}); ${built.split}`;
      }
    }
    self.postMessage({ type: "built", built });
  } catch (err) {
    try {
      if (reader) reader.close();
    } catch {}
    self.postMessage({ type: "error", message: String((err && err.stack) || err) });
  }
};
