// gxpexec - run a directory of GXP shader cases (the conformance suite, or corpus cases) on
// THIS device's GPU and grade them against the reference, exactly as `e2e/gxpexec.mjs` does on
// the desktop: the batch code is the same file (../gxpbatch.js).
//
// params: { cases: "<dir under runner assets>", prefix?: "mlb", offset?: N, limit?: N }
// `prefix` keeps the case names that start with it; `offset`/`limit` take a SLICE, which is how
// a big directory is split into jobs that each fit the runner's short-job contract.
// Put a case directory in place with:
//   VITASLOP_GXP_CASES_OUT=<runner-dir>/assets/<name> cargo test -p vitaslop-gxp-shader \
//     --test conformance -- --ignored --nocapture

import { runBatch } from "../gxpbatch.js";
import "/runner/lane.js"; // globalThis.__laneValue etc. - e2e/caseinputs.mjs, served as-is

const BATCH = 96;
const TEX_SIZE = 64;
const TEX_UNITS = 16;
const TEX_LAYERS = 7;

export async function run(params, { progress, asset }) {
  const dir = params.cases;
  if (!dir) throw new Error("params.cases (a directory under the runner's assets) is required");
  const files = await asset(dir, "json");
  const names = files.filter((f) => f.endsWith(".json") && f !== "coverage.summary").map((f) => f.slice(0, -5)).sort();
  const pool = params.prefix ? names.filter((n) => n.startsWith(params.prefix)) : names;
  const offset = params.offset ?? 0;
  const picked = pool.slice(offset, offset + (params.limit ?? Infinity));
  const cases = [];
  for (const n of picked) {
    cases.push({ ...(await asset(`${dir}/${n}.json`, "json")), src: await asset(`${dir}/${n}.wgsl`) });
  }
  const renderCases = cases.filter((c) => c.stage === "fragment" && (c.units?.length ?? 0) > 0).length;
  if (renderCases > 0) {
    const tex = await asset(`${dir}/casetex.bin`, "bytes");
    const want = TEX_UNITS * TEX_SIZE * TEX_SIZE * 4 * TEX_LAYERS;
    if (tex.length !== want) throw new Error(`casetex.bin is ${tex.length} bytes, expected ${want}`);
    let s = "";
    for (let i = 0; i < tex.length; i += 0x8000) s += String.fromCharCode(...tex.subarray(i, i + 0x8000));
    globalThis.__texb64 = btoa(s);
    globalThis.__texSize = TEX_SIZE;
    globalThis.__texUnits = TEX_UNITS;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter");
  const features = ["shader-f16"].filter((x) => adapter.features.has(x));
  globalThis.__dev = await adapter.requestDevice({ requiredFeatures: features });
  globalThis.__uncaptured = [];
  globalThis.__dev.onuncapturederror = (e) => globalThis.__uncaptured.push(String(e.error.message));

  const counts = { exact: 0, close: 0, diverged: 0, compile: 0 };
  const report = [];
  let ambiguous = 0;
  let flushed = 0;
  for (let i = 0; i < cases.length; i += BATCH) {
    progress(`${i}/${cases.length} cases`);
    const res = await runBatch({ batch: cases.slice(i, i + BATCH) });
    if (res.invalid) {
      return { summary: `RIG FAILURE: ${res.error}`, rigFailure: res.error, counts, features };
    }
    for (const r of res) {
      ambiguous += r.ambiguous ?? 0;
      flushed += r.flushed ?? 0;
      if (r.status === "compile") (counts.compile++, report.push(r));
      else if (r.status === "diverged") (counts.diverged++, report.push(r));
      else if (r.status === "close") counts.close++;
      else counts.exact++;
    }
  }
  return {
    summary: `${cases.length} cases (${offset}.. of ${pool.length}): ${counts.exact} exact, ${counts.close} close, ${counts.diverged} DIVERGED, ${counts.compile} compile-fail`,
    cases: cases.length,
    counts,
    ambiguousLanes: ambiguous,
    flushedLanes: flushed,
    features,
    report,
    uncaptured: globalThis.__uncaptured.slice(0, 20),
  };
}
