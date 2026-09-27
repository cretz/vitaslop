// PIPELINE COMPILE COST on the device: real linked GXP->WGSL pairs (the corpus the wgsl-compile
// job validates) built as RENDER PIPELINES, synchronously and asynchronously, with a probe of
// whether the compile blocks the GPU process.
//
// The question (2026-09-27): a phone browser freezes WHOLE for seconds at a title's loads, and
// the stalls carry tens of new pipelines. Chrome runs `createRenderPipeline` on the GPU process's
// main thread - the thread its compositor also runs on - and `createRenderPipelineAsync` on a
// worker pool. So:
//   sync   - `createRenderPipeline`, timed to its error-scope round trip (which the GPU process
//            answers only after it has handled the create);
//   async  - `createRenderPipelineAsync`, timed to resolution, on a DIFFERENT half of the corpus
//            (identical descriptors are deduplicated, so one half per mode);
//   probe  - a trivial `onSubmittedWorkDone` round trip issued while a burst of creates is in
//            flight: its latency is how long the GPU process's own thread was busy.
// Descriptors are approximated from the WGSL (vertex inputs as float32x4, layout "auto", one
// rgba8unorm target, no depth) - the compile is of the real shader code either way.

function vertexBuffers(code) {
  const m = /struct VsIn\s*\{([^}]*)\}/.exec(code);
  const locs = m ? [...m[1].matchAll(/@location\((\d+)\)/g)].map((x) => Number(x[1])) : [];
  if (!locs.length) return [];
  return [
    {
      arrayStride: 16 * locs.length,
      attributes: locs.map((loc, i) => ({ shaderLocation: loc, offset: 16 * i, format: "float32x4" })),
    },
  ];
}

function desc(dev, code) {
  const module = dev.createShaderModule({ code });
  return {
    layout: "auto",
    vertex: { module, entryPoint: "vs_main", buffers: vertexBuffers(code) },
    fragment: { module, entryPoint: "fs_main", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  };
}

const stats = (v) => {
  if (!v.length) return "none";
  const s = [...v].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  const sum = s.reduce((a, b) => a + b, 0);
  return `n=${s.length} median ${q(0.5).toFixed(1)} ms, p90 ${q(0.9).toFixed(1)}, max ${s[s.length - 1].toFixed(1)}, total ${sum.toFixed(0)} ms`;
};

async function probe(dev) {
  const t = performance.now();
  dev.queue.submit([]);
  await dev.queue.onSubmittedWorkDone();
  return performance.now() - t;
}

export async function run(params, { progress, asset }) {
  const dir = params.dir || "corpus-wgsl";
  const names = (await asset(dir, "json")).filter((f) => f.endsWith(".wgsl") && (!params.only || params.only.includes(f))).slice(params.offset ?? 0, (params.offset ?? 0) + (params.limit ?? 120));
  const adapter = await navigator.gpu.requestAdapter();
  const features = ["shader-f16", "dual-source-blending"].filter((x) => adapter.features.has(x));
  const dev = await adapter.requestDevice({ requiredFeatures: features });
  const codes = [];
  for (let i = 0; i < names.length; i++) codes.push(await asset(`${dir}/${names[i]}`));
  // `perModule`: WHICH shaders are slow, not how slow on average - every module created on its
  // own with `createRenderPipelineAsync` (so the page stays live) and timed to resolution, one at
  // a time so the times do not overlap. Returned slowest first with the module's name, so the
  // slow ones can be opened and compared against the fast ones offline.
  if (params.perModule) {
    const rows = [];
    for (let i = 0; i < codes.length; i++) {
      if (i % 10 === 0) progress(`per-module ${i}/${codes.length}`);
      const t = performance.now();
      let ok = true;
      if (params.mode === "sync") {
        // Timed to the error scope's answer, which the GPU process gives only after the create.
        dev.pushErrorScope("validation");
        dev.createRenderPipeline(desc(dev, codes[i]));
        ok = !(await dev.popErrorScope());
      } else {
        try {
          await dev.createRenderPipelineAsync(desc(dev, codes[i]));
        } catch {
          ok = false;
        }
      }
      rows.push({ name: names[i], ms: Math.round((performance.now() - t) * 10) / 10, ok, bytes: codes[i].length });
    }
    rows.sort((a, b) => b.ms - a.ms);
    return {
      summary: `PER-MODULE ${stats(rows.filter((r) => r.ok).map((r) => r.ms))}; slowest: ${rows.slice(0, 8).map((r) => `${r.name} ${r.ms} ms`).join(", ")}`,
      features,
      rows,
    };
  }

  const half = Math.floor(codes.length / 2);
  const burst = params.burst ?? 10;

  const idle = [];
  for (let i = 0; i < 5; i++) idle.push(await probe(dev));

  // SYNC, one at a time.
  const sync = [];
  let syncFailed = 0;
  for (let i = 0; i < half - burst; i++) {
    if (i % 10 === 0) progress(`sync ${i}/${half}`);
    const d = desc(dev, codes[i]);
    const t = performance.now();
    dev.pushErrorScope("validation");
    dev.createRenderPipeline(d);
    const err = await dev.popErrorScope();
    if (err) syncFailed++;
    else sync.push(performance.now() - t);
  }
  // SYNC burst + probe: how long the GPU process's thread is held by `burst` sync creates.
  const syncBurst = [];
  for (let i = half - burst; i < half; i++) {
    dev.pushErrorScope("validation");
    dev.createRenderPipeline(desc(dev, codes[i]));
    dev.popErrorScope();
  }
  syncBurst.push(await probe(dev));

  // ASYNC, one at a time.
  const asyncT = [];
  let asyncFailed = 0;
  for (let i = half; i < codes.length - burst; i++) {
    if (i % 10 === 0) progress(`async ${i - half}/${codes.length - half}`);
    const d = desc(dev, codes[i]);
    const t = performance.now();
    try {
      await dev.createRenderPipelineAsync(d);
      asyncT.push(performance.now() - t);
    } catch {
      asyncFailed++;
    }
  }
  // ASYNC burst + probe: the same number of creates in flight, then the same round trip.
  const pending = [];
  for (let i = codes.length - burst; i < codes.length; i++) pending.push(dev.createRenderPipelineAsync(desc(dev, codes[i])).catch(() => null));
  const asyncBurst = [await probe(dev)];
  await Promise.all(pending);

  return {
    summary:
      `SYNC ${stats(sync)} (${syncFailed} failed validation) | ASYNC ${stats(asyncT)} (${asyncFailed} failed) | ` +
      `GPU-thread probe: idle ${stats(idle)}; after ${burst} SYNC creates ${syncBurst[0].toFixed(0)} ms; with ${burst} ASYNC creates in flight ${asyncBurst[0].toFixed(0)} ms`,
    features,
    modules: codes.length,
  };
}
