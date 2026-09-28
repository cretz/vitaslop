// wgsl-probe - run ONE small authored compute shader on this device and return its output words.
// The primitive for targeted replications: pin one question ("what does THIS GPU return for
// f16(NaN)?") in a few lines of WGSL, run it on every device, diff the words.
//
// params: {
//   code | asset: WGSL source, or a path under the runner assets holding it. Entry `main`,
//                 @workgroup_size of the author's choosing; bindings: @binding(0) storage
//                 read_write `array<u32>` OUT, @binding(1) storage read `array<u32>` IN (optional).
//   words:        how many u32 words of OUT to return
//   input?:       array of u32 for IN (so a value arrives at runtime - a constant would be folded
//                 by the compiler, and the question is what the GPU does, not the compiler)
//   labels?:      names for the output words, carried into the result for reading
//   f16?:         request shader-f16 (default: when the adapter has it)
// }

export async function run(params, { asset }) {
  const code = params.code ?? (await asset(params.asset));
  const words = params.words ?? 64;
  const adapter = await navigator.gpu.requestAdapter();
  const features = params.f16 === false ? [] : ["shader-f16"].filter((x) => adapter.features.has(x));
  const dev = await adapter.requestDevice({ requiredFeatures: features });
  dev.pushErrorScope("validation");
  const mod = dev.createShaderModule({ code });
  const info = await mod.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error").map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
  if (errors.length) {
    await dev.popErrorScope();
    return { summary: `COMPILE FAILED: ${errors[0]}`, errors, features };
  }
  const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
  const out = dev.createBuffer({ size: words * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = dev.createBuffer({ size: words * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const entries = [{ binding: 0, resource: { buffer: out } }];
  if (params.input) {
    const inp = new Uint32Array(params.input);
    const ib = dev.createBuffer({ size: Math.max(16, inp.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(ib, 0, inp);
    entries.push({ binding: 1, resource: { buffer: ib } });
  }
  const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(1);
  pass.end();
  enc.copyBufferToBuffer(out, 0, read, 0, words * 4);
  dev.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const got = Array.from(new Uint32Array(read.getMappedRange().slice(0)));
  read.unmap();
  const scoped = await dev.popErrorScope();
  const hex = got.map((w) => "0x" + (w >>> 0).toString(16).padStart(8, "0"));
  const labelled = params.labels ? Object.fromEntries(params.labels.map((l, i) => [l, hex[i]])) : null;
  return { summary: `${words} words${scoped ? " - ERROR " + scoped.message : ""}`, words: hex, labelled, features, error: scoped?.message ?? null };
}
