// wgsl-compile - compile a directory of WGSL modules with THIS device's WebGPU (its Tint AND its
// driver's pipeline compiler) and report every message. The desktop tintcheck proves Tint takes
// a module; only the device can say its driver does. A render/compute pipeline is also built
// for each module's entry points, because a driver rejects at pipeline creation, not parse.
//
// params: { dir: "<dir under runner assets>", limit?: N, pipelines?: true }
// Fill it from `corpus.rs::write_every_linked_pair_wgsl` (VITASLOP_GXP_WGSL_OUT=<assets>/<dir>).

import { desc } from "./pipeline-compile.js";

export async function run(params, { progress, asset }) {
  const dir = params.dir;
  if (!dir) throw new Error("params.dir is required");
  const names = (await asset(dir, "json")).filter((f) => f.endsWith(".wgsl")).slice(0, params.limit ?? Infinity);
  const adapter = await navigator.gpu.requestAdapter();
  const features = ["shader-f16", "dual-source-blending"].filter((x) => adapter.features.has(x));
  const dev = await adapter.requestDevice({ requiredFeatures: features });
  const failed = [];
  // Per module: ms to parse (createShaderModule + compilation info) and ms for the pipeline's
  // async compile - the second is the in-game freeze (a fighting title: p90 557 ms on a desktop, 30b).
  const times = [];
  let warned = 0;
  let ok = 0;
  const t0 = performance.now();
  for (let i = 0; i < names.length; i++) {
    if (i % 25 === 0) progress(`${i}/${names.length} modules`);
    const code = await asset(`${dir}/${names[i]}`);
    const tm = performance.now();
    dev.pushErrorScope("validation");
    const mod = dev.createShaderModule({ code });
    const info = await mod.getCompilationInfo();
    const scoped = await dev.popErrorScope();
    const errors = info.messages.filter((m) => m.type === "error");
    const warns = info.messages.filter((m) => m.type !== "error");
    if (warns.length) warned++;
    // `pipelines`: a module that parses can still be refused when a PIPELINE is built from it
    // (a driver compiles the backend shader there), so build one - with `pipeline-compile`'s
    // descriptor - and count its rejection as this module's failure.
    let pipeline = null;
    const tp = performance.now();
    if (params.pipelines && !errors.length && !scoped) {
      try {
        await dev.createRenderPipelineAsync(desc(dev, code));
      } catch (e) {
        pipeline = String((e && e.message) || e).slice(0, 500);
      }
    }
    const entry = { name: names[i], moduleMs: Math.round(tp - tm), pipelineMs: Math.round(performance.now() - tp), bytes: code.length };
    // `variants`: the SAME shaders again under different fixed-function state - cull, depth
    // test/write, blend - which GXM sets per draw and a pipeline bakes in. Whether a state
    // variant costs a whole compile or reuses the first one's shader code decides whether
    // compiling one pipeline per patcher-named pair on a loading screen removes the in-game
    // compiles (30b, a fighting title).
    if (params.variants && !pipeline && !errors.length && !scoped) {
      const base = desc(dev, code);
      const vs = [
        { ...base, primitive: { topology: "triangle-list", cullMode: "back" }, depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" } },
        { ...base, primitive: { topology: "triangle-list", cullMode: "front" }, depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less-equal" }, fragment: { ...base.fragment, targets: [{ format: "rgba8unorm", blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "one", dstFactor: "zero" } } }] } },
      ];
      entry.variantMs = [];
      for (const v of vs) {
        const tv = performance.now();
        try {
          await dev.createRenderPipelineAsync(v);
          entry.variantMs.push(Math.round(performance.now() - tv));
        } catch (e) {
          entry.variantMs.push(`refused: ${String((e && e.message) || e).slice(0, 80)}`);
        }
      }
    }
    times.push(entry);
    if (errors.length || scoped || pipeline) {
      failed.push({
        name: names[i],
        messages: errors.slice(0, 5).map((m) => `${m.lineNum}:${m.linePos} ${m.message}`),
        scope: scoped ? scoped.message.slice(0, 500) : null,
        pipeline,
      });
    } else ok++;
  }
  // `parallel`: the same pipelines AGAIN but all started at once, under fresh state so nothing
  // is a cache hit (blend on, cull front) - total wall against the sum of one-at-a-time says
  // whether the browser compiles them concurrently or queues them (30b, fighting title: in-game compiles
  // took 156 ms p50 against 75-100 ms one at a time).
  let parallel = null;
  if (params.parallel) {
    const codes = [];
    for (const t of times) if (typeof t.pipelineMs === "number") codes.push(await asset(`${dir}/${t.name}`));
    const tp0 = performance.now();
    const done = await Promise.allSettled(
      codes.map((code) => {
        const d = desc(dev, code);
        d.primitive = { topology: "triangle-list", cullMode: "front" };
        d.depthStencil = { format: "depth32float", depthWriteEnabled: false, depthCompare: "greater" };
        return dev.createRenderPipelineAsync(d);
      })
    );
    parallel = {
      n: codes.length,
      ok: done.filter((r) => r.status === "fulfilled").length,
      wallMs: Math.round(performance.now() - tp0),
      sumOneAtATimeMs: times.reduce((a, t) => a + (t.pipelineMs || 0), 0),
    };
  }
  times.sort((a, b) => b.pipelineMs - a.pipelineMs);
  const pms = times.map((t) => t.pipelineMs).sort((a, b) => a - b);
  const pct = (p) => (pms.length ? pms[Math.min(pms.length - 1, Math.round((pms.length - 1) * p))] : 0);
  return {
    summary: `${names.length} modules: ${ok} ok, ${failed.length} FAILED, ${warned} with warnings, ${Math.round(performance.now() - t0)} ms | pipeline compile p50 ${pct(0.5)} p90 ${pct(0.9)} max ${pct(1)} ms`,
    slowest: times.slice(0, 20),
    parallel,
    variantSummary: params.variants
      ? (() => {
          const first = times.filter((t) => Array.isArray(t.variantMs)).map((t) => t.pipelineMs);
          const rest = times.flatMap((t) => (Array.isArray(t.variantMs) ? t.variantMs.filter((x) => typeof x === "number") : []));
          const sum = (v) => v.reduce((a, b) => a + b, 0);
          return `first compile sum ${sum(first)} ms over ${first.length}; state variants sum ${sum(rest)} ms over ${rest.length}`;
        })()
      : null,
    times,
    features,
    failed,
  };
}
