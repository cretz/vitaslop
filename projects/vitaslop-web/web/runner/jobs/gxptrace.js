// gxptrace - WHERE does a diverging shader first go wrong ON THIS DEVICE? Runs trace cases
// (`execcases.rs::write_one_blob_as_a_trace_case`) and reports the first checkpoint whose state
// differs from the reference, with the instructions that ran just before it - the same code the
// desktop's `e2e/gxptrace.mjs` uses (../gxptrace-run.js).
//
// params: { cases: "<dir under runner assets>", names?: ["<stem>", ...] }

import { runTrace, locate } from "../gxptrace-run.js";
import "/runner/lane.js";

export async function run(params, { progress, asset }) {
  const dir = params.cases;
  if (!dir) throw new Error("params.cases is required");
  let stems = (await asset(dir, "json")).filter((f) => f.endsWith(".trace.json")).map((f) => f.slice(0, -".trace.json".length));
  if (params.names) stems = stems.filter((s) => params.names.includes(s));
  const out = [];
  for (const s of stems) {
    progress(s);
    const c = await asset(`${dir}/${s}.trace.json`, "json");
    const src = await asset(`${dir}/${s}.trace.wgsl`);
    const got = await runTrace({ src, c });
    if (got.error) out.push({ name: c.name, error: got.error });
    // The WHOLE per-checkpoint trace rides along, so two DEVICES can be diffed against each
    // other - where the phone first parts from the desktop is where its compiler differs, which
    // an exact comparison against the reference (last-bit noise first) cannot show.
    else out.push({ name: c.name, instrs: c.instrs.length, ...locate(c, got.trace), gpuTrace: c.checkpoints.map((k) => got.trace[k] >>> 0), checkpoints: c.checkpoints, instrList: c.instrs });
  }
  const summary = out
    .map((r) => (r.error ? `${r.name}: RIG FAILURE ${r.error}` : r.first === null ? `${r.name}: agrees everywhere` : `${r.name}: first difference at #${r.first} (culprit run from #${r.culprit})`))
    .join("; ");
  return { summary, traces: out };
}
