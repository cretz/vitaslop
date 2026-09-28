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
  let warned = 0;
  let ok = 0;
  const t0 = performance.now();
  for (let i = 0; i < names.length; i++) {
    if (i % 25 === 0) progress(`${i}/${names.length} modules`);
    const code = await asset(`${dir}/${names[i]}`);
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
    if (params.pipelines && !errors.length && !scoped) {
      try {
        await dev.createRenderPipelineAsync(desc(dev, code));
      } catch (e) {
        pipeline = String((e && e.message) || e).slice(0, 500);
      }
    }
    if (errors.length || scoped || pipeline) {
      failed.push({
        name: names[i],
        messages: errors.slice(0, 5).map((m) => `${m.lineNum}:${m.linePos} ${m.message}`),
        scope: scoped ? scoped.message.slice(0, 500) : null,
        pipeline,
      });
    } else ok++;
  }
  return {
    summary: `${names.length} modules: ${ok} ok, ${failed.length} FAILED, ${warned} with warnings, ${Math.round(performance.now() - t0)} ms`,
    features,
    failed,
  };
}
