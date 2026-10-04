// GPU OBJECT CENSUS (`VITASLOP_GPU_CENSUS=1`): live WebGPU objects by kind, from the JS side.
//
// Why it exists: a desktop-browser golf-title run (`long1`) held its wasm heap flat at 1,084 MB
// while the RENDERER process grew 1,616 -> 2,142 MB and the GPU process 2,103 -> 2,823 MB over
// seven minutes of golf - a leak no panel could name, because every panel counts what the Rust
// side holds and neither of those processes is the Rust heap. wgpu hands its objects to the
// browser and forgets them; this counts them where they live.
//
// It wraps the GPUDevice create* calls and the destroy()/unmap() that end them, and a
// FinalizationRegistry notices the ones the garbage collector reclaimed instead. A buffer or
// texture that is only ever COLLECTED (never destroyed) holds its GPU memory until a GC gets to
// it, which a busy worker may not do for a long time - that is a leak in practice even though
// nothing is unreachable. Bind groups and pipelines have no destroy: for them "collected" is the
// only end.
//
// Cost: one JS wrapper per create call and a WeakMap entry per object. Off unless asked for.

const KINDS = ["buffer", "texture", "bindGroup", "querySet", "shaderModule", "pipeline", "sampler"];

// Bytes per texel (or per 4x4 block / 16 for compressed formats), for a texture's estimated size.
function bytesPerTexel(format) {
  if (/^(bc1|bc4|etc2-rgb8|etc2-rgb8a1|eac-r11)/.test(format)) return 0.5;
  if (/^(bc|etc2|eac|astc-4x4)/.test(format)) return 1;
  if (/^astc/.test(format)) return 0.5;
  if (/32float|32uint|32sint/.test(format)) return format.startsWith("rgba") ? 16 : format.startsWith("rg") ? 8 : 4;
  if (/16float|16uint|16sint|16unorm|16snorm/.test(format)) return format.startsWith("rgba") ? 8 : format.startsWith("rg") ? 4 : 2;
  if (/^(r8|stencil8)/.test(format)) return 1;
  if (/^rg8/.test(format)) return 2;
  return 4;
}

function textureBytes(desc) {
  const s = desc.size || {};
  const w = Array.isArray(s) ? s[0] : s.width || 1;
  const h = Array.isArray(s) ? s[1] || 1 : s.height || 1;
  const d = Array.isArray(s) ? s[2] || 1 : s.depthOrArrayLayers || 1;
  const mips = desc.mipLevelCount || 1;
  const samples = desc.sampleCount || 1;
  return Math.round(w * h * d * bytesPerTexel(desc.format || "") * samples * (mips > 1 ? 4 / 3 : 1));
}

export function installGpuCensus(log, everyMs = 10000) {
  if (typeof GPUDevice === "undefined" || self.__vitaslopGpuCensus) return;
  self.__vitaslopGpuCensus = true;
  const t0 = performance.now();
  const stat = {};
  for (const k of KINDS) stat[k] = { made: 0, destroyed: 0, collected: 0, live: 0, bytes: 0 };
  // Mapped buffers: a mapping is renderer-side shared memory until unmap().
  const mapped = { now: 0, bytes: 0, maps: 0 };
  const rec = new WeakMap();
  const reg = new FinalizationRegistry((r) => {
    if (r.ended) return;
    r.ended = true;
    labelAdd(r, -1, "collected");
    const s = stat[r.kind];
    s.collected++;
    s.live--;
    s.bytes -= r.bytes;
  });
  // Live buffers and textures by LABEL (digits folded), so a growing count names its call site;
  // `made` and `collected` per label name a CHURN (created every frame, left to the collector)
  // that a live count alone hides.
  const byLabel = new Map();
  const labelAdd = (r, sign, how) => {
    if (r.kind !== "buffer" && r.kind !== "texture") return;
    const key = `${r.kind === "texture" ? "tex " : ""}${r.label}`;
    const e = byLabel.get(key) || { n: 0, bytes: 0, made: 0, collected: 0 };
    e.n += sign;
    e.bytes += sign * r.bytes;
    if (sign > 0) e.made++;
    if (how === "collected") e.collected++;
    byLabel.set(key, e);
  };
  const track = (kind, obj, bytes, label = "") => {
    if (!obj || typeof obj !== "object") return obj;
    const r = { kind, bytes, ended: false, mapped: false, label: String(label || "(none)").replace(/\d+/g, "#") };
    labelAdd(r, 1);
    rec.set(obj, r);
    reg.register(obj, r);
    const s = stat[kind];
    s.made++;
    s.live++;
    s.bytes += bytes;
    return obj;
  };
  const end = (obj) => {
    const r = rec.get(obj);
    if (!r || r.ended) return;
    r.ended = true;
    labelAdd(r, -1);
    const s = stat[r.kind];
    s.destroyed++;
    s.live--;
    s.bytes -= r.bytes;
    if (r.mapped) {
      r.mapped = false;
      mapped.now--;
      mapped.bytes -= r.bytes;
    }
  };
  const wrap = (proto, name, fn) => {
    const orig = proto[name];
    if (typeof orig !== "function") return;
    proto[name] = function (...args) {
      return fn.call(this, orig, args);
    };
  };
  const D = GPUDevice.prototype;
  const setMapped = (buf, on) => {
    const r = rec.get(buf);
    if (!r || r.mapped === on) return;
    r.mapped = on;
    mapped.now += on ? 1 : -1;
    mapped.bytes += on ? r.bytes : -r.bytes;
    if (on) mapped.maps++;
  };
  wrap(D, "createBuffer", function (orig, a) {
    const b = track("buffer", orig.apply(this, a), a[0]?.size || 0, a[0]?.label);
    if (a[0]?.mappedAtCreation) setMapped(b, true);
    return b;
  });
  wrap(D, "createTexture", function (orig, a) {
    return track("texture", orig.apply(this, a), textureBytes(a[0] || {}), a[0]?.label);
  });
  wrap(D, "createBindGroup", function (orig, a) {
    return track("bindGroup", orig.apply(this, a), 0);
  });
  wrap(D, "createQuerySet", function (orig, a) {
    return track("querySet", orig.apply(this, a), (a[0]?.count || 0) * 8);
  });
  wrap(D, "createShaderModule", function (orig, a) {
    return track("shaderModule", orig.apply(this, a), (a[0]?.code || "").length);
  });
  wrap(D, "createSampler", function (orig, a) {
    return track("sampler", orig.apply(this, a), 0);
  });
  wrap(D, "createRenderPipeline", function (orig, a) {
    return track("pipeline", orig.apply(this, a), 0);
  });
  wrap(D, "createComputePipeline", function (orig, a) {
    return track("pipeline", orig.apply(this, a), 0);
  });
  wrap(D, "createRenderPipelineAsync", function (orig, a) {
    return orig.apply(this, a).then((p) => track("pipeline", p, 0));
  });
  wrap(D, "createComputePipelineAsync", function (orig, a) {
    return orig.apply(this, a).then((p) => track("pipeline", p, 0));
  });
  for (const P of [GPUBuffer.prototype, GPUTexture.prototype, GPUQuerySet.prototype]) {
    wrap(P, "destroy", function (orig, a) {
      end(this);
      return orig.apply(this, a);
    });
  }
  wrap(GPUBuffer.prototype, "mapAsync", function (orig, a) {
    return orig.apply(this, a).then((v) => {
      setMapped(this, true);
      return v;
    });
  });
  wrap(GPUBuffer.prototype, "unmap", function (orig, a) {
    setMapped(this, false);
    return orig.apply(this, a);
  });
  const MB = (b) => (b / (1024 * 1024)).toFixed(1);
  const line = () => {
    const parts = KINDS.map((k) => {
      const s = stat[k];
      const bytes = k === "buffer" || k === "texture" ? ` ${MB(s.bytes)} MB` : "";
      return `${k} live ${s.live}${bytes} (made ${s.made}, destroyed ${s.destroyed}, collected ${s.collected})`;
    });
    const heap = performance.memory ? ` | js heap ${MB(performance.memory.usedJSHeapSize)} MB used` : "";
    const top = [...byLabel.entries()]
      .filter(([, e]) => e.n > 0)
      .sort((x, y) => y[1].bytes - x[1].bytes)
      .slice(0, 10)
      .map(([l, e]) => `${l} ${e.n}/${MB(e.bytes)} MB`)
      .join(", ");
    const churn = [...byLabel.entries()]
      .filter(([, e]) => e.collected > 0)
      .sort((x, y) => y[1].made - x[1].made)
      .slice(0, 6)
      .map(([l, e]) => `${l} made ${e.made}, collected ${e.collected}`)
      .join(", ");
    log(
      `[gpu-census] t=${((performance.now() - t0) / 1000).toFixed(0)}s ${parts.join(" | ")} | mapped now ${mapped.now} (${MB(mapped.bytes)} MB, ${mapped.maps} maps ever)${heap} | live by label: ${top} | LEFT TO THE COLLECTOR by label: ${churn || "none"}`
    );
  };
  setInterval(line, everyMs);
}
