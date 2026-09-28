// gxptrace-run.js - run ONE trace case on THIS browser's GPU and say where it first departs
// from the reference. Shared by `e2e/gxptrace.mjs` (desktop) and the device runner's `gxptrace`
// job, so the phone and the desktop locate a divergence with the same code.
//
// Needs `globalThis.__laneValue` / `__applyCaseInputs` (e2e/caseinputs.mjs, /runner/lane.js).

/// The GPU half: dispatch the trace module, return `{ trace }` (a checksum per checkpoint) or
/// `{ error }`.
export async function runTrace({ src, c }) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return { error: "no adapter" };
  const want = ["shader-f16"].filter((x) => adapter.features.has(x));
  const dev = await adapter.requestDevice({ requiredFeatures: want });
  const laneValue = globalThis.__laneValue;
  dev.pushErrorScope("validation");
  const mod = dev.createShaderModule({ code: src });
  const info = await mod.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error").map((m) => `${m.lineNum}: ${m.message}`);
  if (errs.length) {
    await dev.popErrorScope();
    return { error: errs.slice(0, 3).join(" | ") };
  }
  const pipe = dev.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "cs_main" } });
  const n = c.lanes;
  const input = new Float32Array(n * 2);
  for (let j = 0; j < n; j++) {
    input[j] = laneValue(c.seed, j);
    input[n + j] = laneValue(c.seed, n + j);
  }
  // The same per-program overrides the case carries - see `caseinputs.mjs`. A trace taken on
  // different inputs from the reference's would locate a divergence that is not there.
  globalThis.__applyCaseInputs(c, input);
  const inBuf = dev.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(inBuf, 0, input);
  const outBuf = dev.createBuffer({ size: n * 4 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const traceBytes = c.trace.length * 4;
  const trBuf = dev.createBuffer({ size: traceBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = dev.createBuffer({ size: traceBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const entries = [
    { binding: 0, resource: { buffer: inBuf } },
    { binding: 1, resource: { buffer: outBuf } },
    { binding: 3, resource: { buffer: trBuf } },
  ];
  // A program with 0xE8 loads declares its guest-memory WINDOW at binding 2. Leaving it out
  // fails the bind group, which fails the dispatch, which reads as "the trace module did not
  // run" - it ran nothing at all.
  if (c.mem && c.mem.length) {
    const words = new Uint32Array(c.mem);
    const memBuf = dev.createBuffer({
      size: words.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    dev.queue.writeBuffer(memBuf, 0, words);
    entries.push({ binding: 2, resource: { buffer: memBuf } });
  }
  const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(1);
  pass.end();
  enc.copyBufferToBuffer(trBuf, 0, read, 0, traceBytes);
  dev.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const trace = Array.from(new Uint32Array(read.getMappedRange().slice(0)));
  read.unmap();
  const err = await dev.popErrorScope();
  return err ? { error: err.message } : { trace };
}

/// The verdict: the first CHECKPOINT whose incoming state differs, and the instructions that ran
/// just before it. Only checkpointed indices mean anything - every other slot is zero on both
/// sides because nothing writes it, and counting those as agreement would flatter the instrument.
/// The comparison is EXACT: a first difference is where the two sides first stopped being
/// identical - where to LOOK - not a verdict that the program is wrong.
export function locate(c, gpuTrace) {
  const checked = c.checkpoints;
  let first = null;
  let agreed = 0;
  for (const k of checked) {
    if (c.trace[k] === gpuTrace[k]) {
      agreed++;
      continue;
    }
    if (first === null) first = k;
  }
  if (first === null) return { agreed, checkpoints: checked.length, first: null };
  const before = checked.filter((k) => k < first);
  const culprit = before.length ? before[before.length - 1] : null;
  const run = [];
  if (culprit !== null) for (let k = culprit; k < first; k++) if (c.instrs[k]) run.push(c.instrs[k]);
  return {
    agreed,
    checkpoints: checked.length,
    first,
    culprit,
    reference: c.trace[first] >>> 0,
    gpu: gpuTrace[first] >>> 0,
    run,
  };
}
