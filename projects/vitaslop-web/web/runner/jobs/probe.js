// probe - the device's fingerprint and a GPU round trip: proves the runner, the device and the
// result path all work, in about a second. The first job to queue for any new phone.

import { fingerprint } from "../fingerprint.js";

export async function run() {
  const fp = await fingerprint();
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return { summary: "NO WEBGPU", fp };
  const dev = await adapter.requestDevice();
  const t0 = performance.now();
  const buf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const src = dev.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
  new Uint32Array(src.getMappedRange()).set([1, 2, 3, 4]);
  src.unmap();
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, buf, 0, 16);
  dev.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const back = [...new Uint32Array(buf.getMappedRange())];
  const ms = performance.now() - t0;
  return { summary: `${fp.adapter}; round trip ${ms.toFixed(1)} ms; readback ${back.join(",")}`, fp, roundTripMs: ms, back };
}
