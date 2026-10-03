// texenc-bench - time the GPU texture encoder (`vitaslop-platform/src/texenc.wgsl`) on THIS
// device: one `encode_etc2` dispatch over a synthetic image, RGB and RGBA, per shader variant.
// Written for MK's round start (30c): one frame measured 1034 ms of GPU with 18 PVRTC textures
// ETC2-encoded in it, and the question is how much of that is the encoder itself.
//
// params: {
//   dir:    "<dir under runner assets>" holding the variants
//   shaders: ["texenc.wgsl", ...] file names in `dir`
//   size?:  texels a side (default 512)
//   reps?:  timed repetitions per case (default 3; the minimum is reported)
//   stages?: ["encode_rgb", "encode_rgba", "decode_pvrtc", "halve"] (default the two encodes)
// }
// Output is hashed per case so variants that must be byte-identical can be checked on-device.

const FLAG_SWIZZLED = 1, FLAG_PVRTC2 = 2, FLAG_4BPP = 4, FLAG_ALPHA = 8;

// Texture-like content: smooth gradients, value noise at two scales, hard edges, and an alpha
// channel with both flat and structured regions. Deterministic.
function image(n) {
  const px = new Uint32Array(n * n);
  let s = 0x12345678;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) >>> 24);
  const lattice = new Uint8Array(65 * 65 * 3).map(() => rnd());
  const noise = (x, y, c, cell) => {
    const gx = x / cell, gy = y / cell;
    const x0 = Math.floor(gx) % 64, y0 = Math.floor(gy) % 64, fx = gx - Math.floor(gx), fy = gy - Math.floor(gy);
    const at = (i, j) => lattice[((j * 65) + i) * 3 + c];
    const a = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx;
    const b = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx;
    return a * (1 - fy) + b * fy;
  };
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const c = [0, 1, 2].map((k) => {
        let v = 0.45 * noise(x, y, k, 32) + 0.25 * noise(x, y, k, 5) + 0.3 * ((x + y * (k + 1)) % 256);
        if (((x >> 5) + (y >> 5)) % 7 === 0) v = k === 0 ? 240 : 20;
        return Math.max(0, Math.min(255, Math.round(v + (rnd() & 7) - 3)));
      });
      const a = (x >> 6) % 2 ? 255 : Math.max(0, Math.min(255, Math.round(noise(x, y, 0, 8))));
      px[y * n + x] = c[0] | (c[1] << 8) | (c[2] << 16) | (a << 24);
    }
  }
  return px;
}

function params(o) {
  const p = new Uint32Array(16);
  const f = ["width", "height", "blocks_x", "blocks_y", "padded_x", "padded_y", "src_word", "rgba_word", "out_word", "src_width", "src_height", "out_row_words", "flags", "src_format", "src_block_words", "pad0"];
  f.forEach((k, i) => (p[i] = o[k] ?? 0));
  return p;
}

function fnv(words) {
  let h = 0x811c9dc5;
  const b = new Uint8Array(words.buffer, words.byteOffset, words.byteLength);
  for (let i = 0; i < b.length; i++) h = Math.imul(h ^ b[i], 0x01000193) >>> 0;
  return h.toString(16).padStart(8, "0");
}

export async function run(prm, { progress, asset }) {
  const n = prm.size ?? 512;
  const reps = prm.reps ?? 3;
  const stages = prm.stages ?? ["encode_rgb", "encode_rgba"];
  const adapter = await navigator.gpu.requestAdapter();
  const tsOk = adapter.features.has("timestamp-query");
  const dev = await adapter.requestDevice({
    requiredFeatures: tsOk ? ["timestamp-query"] : [],
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize },
  });
  // The GPU's own clock around the dispatch: wall time to `onSubmittedWorkDone` carried
  // 10+ ms of callback noise on the phone (job 082: 29/17/22/29/16 ms for one case).
  const qs = tsOk ? dev.createQuerySet({ type: "timestamp", count: 2 }) : null;
  const qres = tsOk ? dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
  const qread = tsOk ? dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null;
  const img = image(n);
  const bw = n / 4;
  const outRowWords = Math.ceil((bw * 4 * 4) / 256) * 64; // RGBA8 ETC2 = 16 B/block; padded rows
  const mk = (size, usage) => dev.createBuffer({ size, usage });
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  // PVRTC source: random 4bpp blocks (content does not steer the decoder's cost).
  const src = mk(Math.max(256, (n / 4) * (n / 4) * 8), S);
  dev.queue.writeBuffer(src, 0, new Uint32Array((n / 4) * (n / 4) * 2).map((_, i) => Math.imul(i + 1, 2654435761) >>> 0));
  const rgba = mk(n * n * 4 * 2, S);
  const outb = mk(outRowWords * 4 * bw, S);
  const read = mk(outRowWords * 4 * bw, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const ubo = mk(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const results = [];
  for (const name of prm.shaders) {
    const code = await asset(`${prm.dir}/${name}`);
    progress(`${name}: compiling`);
    const shader = dev.createShaderModule({ code });
    const info = await shader.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === "error");
    if (errs.length) {
      results.push({ name, error: errs.slice(0, 3).map((m) => `${m.lineNum}:${m.linePos} ${m.message}`) });
      continue;
    }
    const pipes = {};
    const bgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const layout = dev.createPipelineLayout({ bindGroupLayouts: [bgl] });
    const tc = performance.now();
    for (const ep of ["encode_etc2", "decode_pvrtc", "halve"]) {
      pipes[ep] = await dev.createComputePipelineAsync({ layout, compute: { module: shader, entryPoint: ep } });
    }
    const row = { name, compileMs: Math.round(performance.now() - tc) };
    for (const st of stages) {
      let ep, p, groups;
      if (st === "encode_rgb" || st === "encode_rgba") {
        ep = "encode_etc2";
        p = params({ width: n, height: n, out_row_words: outRowWords, flags: st === "encode_rgba" ? FLAG_ALPHA : 0 });
        groups = [Math.ceil(bw / 8), Math.ceil(bw / 8)];
      } else if (st === "decode_pvrtc") {
        ep = "decode_pvrtc";
        p = params({ width: n, height: n, blocks_x: n / 4, blocks_y: n / 4, padded_x: n / 4, padded_y: n / 4, flags: FLAG_SWIZZLED | FLAG_4BPP | FLAG_PVRTC2, src_block_words: 2, rgba_word: n * n });
        groups = [Math.ceil(n / 4 / 8), Math.ceil(n / 4 / 8)];
      } else if (st === "halve") {
        ep = "halve";
        p = params({ width: n / 2, height: n / 2, src_width: n, src_height: n, rgba_word: n * n, src_word: 0 });
        groups = [Math.ceil(n / 2 / 8), Math.ceil(n / 2 / 8)];
      }
      const bg = dev.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: ubo } },
          { binding: 1, resource: { buffer: src } },
          { binding: 2, resource: { buffer: rgba } },
          { binding: 3, resource: { buffer: outb } },
        ],
      });
      const ms = [];
      let hash = null;
      dev.pushErrorScope("validation");
      for (let r = 0; r < reps; r++) {
        dev.queue.writeBuffer(rgba, 0, img);
        dev.queue.writeBuffer(ubo, 0, p);
        // Cleared per case, so a hash covers only what THIS case wrote - an RGB level fills
        // half of each padded row, and the rest held the previous variant's output (job 082).
        const clr = dev.createCommandEncoder();
        clr.clearBuffer(outb);
        dev.queue.submit([clr.finish()]);
        await dev.queue.onSubmittedWorkDone();
        const enc = dev.createCommandEncoder();
        const pass = enc.beginComputePass(qs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
        pass.setPipeline(pipes[ep]);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(groups[0], groups[1], 1);
        pass.end();
        if (qs) {
          enc.resolveQuerySet(qs, 0, 2, qres, 0);
          enc.copyBufferToBuffer(qres, 0, qread, 0, 16);
        }
        const t = performance.now();
        dev.queue.submit([enc.finish()]);
        await dev.queue.onSubmittedWorkDone();
        if (qs) {
          await qread.mapAsync(GPUMapMode.READ);
          const v = new BigUint64Array(qread.getMappedRange().slice(0));
          qread.unmap();
          ms.push(Number(v[1] - v[0]) / 1e6);
        } else ms.push(performance.now() - t);
      }
      const verr = await dev.popErrorScope();
      if (verr) throw new Error(`${name} ${st}: ${verr.message}`);
      if (ep === "encode_etc2") {
        const enc = dev.createCommandEncoder();
        enc.copyBufferToBuffer(outb, 0, read, 0, read.size);
        dev.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        hash = fnv(new Uint32Array(read.getMappedRange().slice(0)));
        read.unmap();
      }
      const best = Math.min(...ms);
      row[st] = { ms: Math.round(best * 10) / 10, all: ms.map((v) => Math.round(v)), mtexelPerS: Math.round(((n * n) / best / 1000) * 10) / 10, hash };
      progress(`${name} ${st}: ${Math.round(best)} ms`);
    }
    results.push(row);
  }
  return {
    clock: qs ? "gpu timestamps" : "wall to onSubmittedWorkDone",
    summary: results.map((r) => (r.error ? `${r.name} ERROR ${r.error[0]}` : `${r.name}: ${stages.map((s) => `${s} ${r[s].ms} ms${r[s].hash ? ` #${r[s].hash}` : ""}`).join(", ")}`)).join(" | ") + ` (${n}x${n})`,
    results,
  };
}
