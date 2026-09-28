// F16 HELPER EQUIVALENCE on the device: the native arm's `gxp_hq` and `gxp_q2` as they ship
// now against the forms they replaced, over EVERY f32 bit pattern (or a chunk range of them),
// on this device's own shader compiler.
//
// The question (2026-09-27): the native round trip used to be `unpack2x16float(gxp_f16b(v))[0]`
// - narrow, pack, mask, unpack - and is now `gxp_f16r(v)` directly; the pair form is now one
// `vec2<f16>` conversion. Argued identical (a pack and an unpack of a value f16 holds exactly
// change nothing), and this MEASURES it: any input whose output bits differ is counted, NaN
// separately (a NaN must stay a NaN; its payload was device-defined in both forms).
//
// The helper text is the SHIPPED text: `assets/f16rounding/{native,common}.wgsl` are copies of
// `vitaslop-gxp-shader/src/f16rounding/` and `q2.wgsl` of `GXP_Q2_NATIVE`, refreshed with the
// build under test.
//
// params: chunks (of 2^28 inputs; 16 = all 2^32), from (first chunk).

export async function run(params, { progress, asset }) {
  const native = await asset("f16rounding/native.wgsl");
  const common = await asset("f16rounding/common.wgsl");
  const q2 = await asset("f16rounding/q2.wgsl");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter.features.has("shader-f16")) throw new Error("no shader-f16 on this device");
  const dev = await adapter.requestDevice({ requiredFeatures: ["shader-f16"] });
  const code = `enable f16;
${native}
${common}
${q2}
fn old_hq(v: f32) -> f32 { return unpack2x16float(gxp_f16b(v))[0]; }
struct Cnt { hq: atomic<u32>, q2: atomic<u32>, nan: atomic<u32>, first_hq: atomic<u32>, first_q2: atomic<u32>, done: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> cnt: Cnt;
@group(0) @binding(1) var<uniform> base: vec4<u32>;
fn is_nan(b: u32) -> bool { return (b & 0x7fffffffu) > 0x7f800000u; }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let lane = id.x + id.y * (4096u * 256u);
  for (var k = 0u; k < 16u; k = k + 1u) {
    let b = base.x + lane * 16u + k;
    let v = bitcast<f32>(b);
    let o = bitcast<u32>(old_hq(v));
    let n = bitcast<u32>(gxp_hq(v));
    if (is_nan(b)) {
      if (!is_nan(o) || !is_nan(n)) { atomicAdd(&cnt.nan, 1u); }
    } else if (o != n) {
      atomicAdd(&cnt.hq, 1u);
      atomicMin(&cnt.first_hq, b);
    }
    // The pair: this input with a second one derived from it, so both lanes see every class.
    let b2 = b ^ 0x5a5a5a5au;
    let w = vec2<f32>(v, bitcast<f32>(b2));
    let qo = bitcast<vec2<u32>>(vec2<f32>(old_hq(w.x), old_hq(w.y)));
    let qn = bitcast<vec2<u32>>(gxp_q2(w));
    let nan0 = is_nan(b);
    let nan1 = is_nan(b2);
    if ((!nan0 && qo.x != qn.x) || (!nan1 && qo.y != qn.y)) {
      atomicAdd(&cnt.q2, 1u);
      atomicMin(&cnt.first_q2, b);
    }
    if ((nan0 && !is_nan(qn.x)) || (nan1 && !is_nan(qn.y))) { atomicAdd(&cnt.nan, 1u); }
  }
  if (lane == 0u) { atomicAdd(&cnt.done, 1u); }
}`;
  // `probe: [bits...]`: the actual outputs for listed inputs - old gxp_hq, new gxp_hq, the pair's
  // first lane, and gxp_f16r alone - to NAME a mismatch rather than count it.
  if (params.probe) {
    const n = params.probe.length;
    const pcode = `enable f16;
${native}
${common}
${q2}
fn old_hq(v: f32) -> f32 { return unpack2x16float(gxp_f16b(v))[0]; }
@group(0) @binding(0) var<storage, read> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> outp: array<vec4<u32>>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let v = bitcast<f32>(inp[id.x]);
  outp[id.x] = vec4<u32>(bitcast<u32>(old_hq(v)), bitcast<u32>(gxp_hq(v)), bitcast<u32>(gxp_q2(vec2<f32>(v, v)).x), gxp_f16b(v));
}`;
    const pm = dev.createShaderModule({ code: pcode });
    const pp = await dev.createComputePipelineAsync({ layout: "auto", compute: { module: pm, entryPoint: "main" } });
    const ib = dev.createBuffer({ size: 4 * n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(ib, 0, new Uint32Array(params.probe.map((x) => Number(x) >>> 0)));
    const ob = dev.createBuffer({ size: 16 * n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rbp = dev.createBuffer({ size: 16 * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const bgp = dev.createBindGroup({ layout: pp.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: ib } }, { binding: 1, resource: { buffer: ob } }] });
    const e = dev.createCommandEncoder();
    const ps = e.beginComputePass();
    ps.setPipeline(pp);
    ps.setBindGroup(0, bgp);
    ps.dispatchWorkgroups(n);
    ps.end();
    e.copyBufferToBuffer(ob, 0, rbp, 0, 16 * n);
    dev.queue.submit([e.finish()]);
    await rbp.mapAsync(GPUMapMode.READ);
    const o = new Uint32Array(rbp.getMappedRange().slice(0));
    const h = (x) => `0x${x.toString(16).padStart(8, "0")}`;
    const rows = params.probe.map((x, i) => `${h(Number(x) >>> 0)}: old ${h(o[4 * i])} new ${h(o[4 * i + 1])} q2 ${h(o[4 * i + 2])} f16b ${h(o[4 * i + 3])}`);
    return { summary: rows.join(" | "), rows };
  }
  const module = dev.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === "error");
  if (errs.length) throw new Error(errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
  const pipe = await dev.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  const cnt = dev.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(cnt, 0, new Uint32Array([0, 0, 0, 0xffffffff, 0xffffffff, 0, 0, 0]));
  const ubo = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bg = dev.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: cnt } },
      { binding: 1, resource: { buffer: ubo } },
    ],
  });
  const from = params.from ?? 0;
  const chunks = params.chunks ?? 16;
  const t0 = performance.now();
  for (let c = from; c < from + chunks; c++) {
    dev.queue.writeBuffer(ubo, 0, new Uint32Array([(c * 0x10000000) >>> 0, 0, 0, 0]));
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(4096, 16);
    pass.end();
    dev.queue.submit([enc.finish()]);
    await dev.queue.onSubmittedWorkDone();
    progress(`chunk ${c + 1 - from}/${chunks} (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
  }
  const rb = dev.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(cnt, 0, rb, 0, 32);
  dev.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const [hq, q2m, nan, firstHq, firstQ2, done] = new Uint32Array(rb.getMappedRange().slice(0));
  const hex = (x) => (x === 0xffffffff ? "-" : `0x${x.toString(16).padStart(8, "0")}`);
  const inputs = chunks * 0x10000000;
  return {
    summary:
      `${inputs} inputs (chunks ${from}..${from + chunks - 1}, ${done} of ${chunks} dispatches ran): ` +
      `gxp_hq ${hq} MISMATCH (first ${hex(firstHq)}), gxp_q2 ${q2m} MISMATCH (first ${hex(firstQ2)}), ` +
      `NaN-lost ${nan}; ${((performance.now() - t0) / 1000).toFixed(1)} s`,
    hq,
    q2: q2m,
    nan,
    done,
  };
}
