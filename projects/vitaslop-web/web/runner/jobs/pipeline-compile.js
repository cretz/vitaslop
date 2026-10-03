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

// The vertex format each input's DECLARED type needs. Every input used to be `float32x4`, which
// a module with an integer input (a skinned shader's `vec4<u32>` bone indices) refuses - so the
// pipeline failed and the timing was of a validation error, not a compile.
function vertexFormat(ty) {
  const m = /^(?:vec([234])<)?(f32|u32|i32)>?$/.exec(ty.replace(/\s/g, ""));
  if (!m) return "float32x4";
  const n = m[1] ? Number(m[1]) : 1;
  const base = { f32: "float32", u32: "uint32", i32: "sint32" }[m[2]];
  return n === 1 ? base : `${base}x${n}`;
}

function vertexBuffers(code) {
  const m = /struct VsIn\s*\{([^}]*)\}/.exec(code);
  const ins = m ? [...m[1].matchAll(/@location\((\d+)\)\s*\w+\s*:\s*([^,\n]+)/g)].map((x) => ({ loc: Number(x[1]), ty: x[2].trim() })) : [];
  if (!ins.length) return [];
  return [
    {
      arrayStride: 16 * ins.length,
      attributes: ins.map((a, i) => ({ shaderLocation: a.loc, offset: 16 * i, format: vertexFormat(a.ty) })),
    },
  ];
}

export function desc(dev, code) {
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
  // `salt`: make every module's CODE unique, so no cache - the browser's, or the driver's keyed on
  // the translated shader - can answer for it. A COMMENT does not do this: the WGSL front end
  // drops it, the driver sees identical code, and a "fresh" repeat of a module compiled 3x
  // faster than its first build (MEASURED, job 188: 237.7 then 69.5 ms). The salt is a uniform
  // compare against a per-run random constant that no compiler can fold away, and it never
  // fires (the depth uniform's `range.w` is not that bit pattern).
  if (params.salt) {
    const run = (Math.random() * 0xffffffff) >>> 0;
    for (let i = 0; i < codes.length; i++) {
      const k = (((run + i * 0x9e3779b1) >>> 0) | 1) >>> 0;
      codes[i] = codes[i].replace(
        /(fn fs_main\([^{]*\{)/,
        `$1\n  if (bitcast<u32>(gxp_depth.range.w) == ${k}u) { discard; }`,
      );
    }
  }
  // One throwaway build first: the device's first pipeline pays a warm-up the rest do not.
  if (params.warmup && codes.length) {
    try {
      await dev.createRenderPipelineAsync(desc(dev, codes[0].replace(/(fn fs_main\([^{]*\{)/, "$1\n  if (bitcast<u32>(gxp_depth.range.w) == 3u) { discard; }")));
    } catch {}
  }
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
      let err = "";
      if (params.mode === "sync") {
        // Timed to the error scope's answer, which the GPU process gives only after the create.
        dev.pushErrorScope("validation");
        dev.createRenderPipeline(desc(dev, codes[i]));
        ok = !(await dev.popErrorScope());
      } else {
        try {
          await dev.createRenderPipelineAsync(desc(dev, codes[i]));
        } catch (e) {
          ok = false;
          err = String((e && e.message) || e).slice(0, 400);
        }
      }
      rows.push({ name: names[i], ms: Math.round((performance.now() - t) * 10) / 10, ok, bytes: codes[i].length, ...(err ? { err } : {}) });
    }
    rows.sort((a, b) => b.ms - a.ms);
    return {
      summary: `PER-MODULE ${stats(rows.filter((r) => r.ok).map((r) => r.ms))}; slowest: ${rows.slice(0, 8).map((r) => `${r.name} ${r.ms} ms`).join(", ")}`,
      features,
      rows,
    };
  }

  // `variants`: does a SECOND pipeline over shaders the driver has already compiled cost a full
  // compile again? The answer decides whether building one speculative pipeline per
  // patcher-named pair (at load time, before cull/depth/blend are known) would take the compile
  // out of the frame that first draws it. Per module: A = first pipeline; B = the SAME module
  // object under a different depth/cull/blend state; C = a NEW module from the same code under a
  // third state. Then `parallel`: the next N modules created all at once vs one at a time - how
  // many compiles the browser actually runs concurrently.
  if (params.variants) {
    const depthDesc = (d, func, write) => ({
      ...d,
      depthStencil: { format: "depth24plus", depthCompare: func, depthWriteEnabled: write },
    });
    const withState = (d, cull, blend) => ({
      ...d,
      primitive: { topology: "triangle-list", cullMode: cull },
      fragment: {
        ...d.fragment,
        targets: [
          blend
            ? { format: "rgba8unorm", blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "zero" } } }
            : { format: "rgba8unorm" },
        ],
      },
    });
    const timed = async (d) => {
      const t = performance.now();
      try {
        await dev.createRenderPipelineAsync(d);
        return performance.now() - t;
      } catch {
        return null;
      }
    };
    const n = Math.min(params.variants, codes.length);
    const a = [], b = [], c = [];
    for (let i = 0; i < n; i++) {
      progress(`variants ${i}/${n}`);
      const base = desc(dev, codes[i]);
      const ta = await timed(base);
      const tb = await timed(depthDesc(withState(base, "back", true), "less-equal", false));
      const tc = await timed(depthDesc(withState(desc(dev, codes[i]), "front", false), "greater", true));
      if (ta != null && tb != null && tc != null) {
        a.push(ta);
        b.push(tb);
        c.push(tc);
      }
    }
    const pn = Math.min(params.parallel ?? 16, codes.length - n);
    const serial = [];
    const half2 = Math.floor(pn / 2);
    for (let i = n; i < n + half2; i++) {
      const t = await timed(desc(dev, codes[i]));
      if (t != null) serial.push(t);
    }
    const tp = performance.now();
    await Promise.all(Array.from({ length: pn - half2 }, (_, k) => timed(desc(dev, codes[n + half2 + k]))));
    const parWall = performance.now() - tp;
    const serialSum = serial.reduce((x, y) => x + y, 0);
    return {
      summary:
        `A first pipeline ${stats(a)} | B same module, new state ${stats(b)} | C new module same code, new state ${stats(c)} | ` +
        `serial ${serial.length} creates sum ${serialSum.toFixed(0)} ms vs ${pn - half2} in flight together ${parWall.toFixed(0)} ms wall`,
      features,
      a, b, c,
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
