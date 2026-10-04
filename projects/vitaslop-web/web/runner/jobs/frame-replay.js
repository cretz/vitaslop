// frame-replay - render a captured frame on THIS device's GPU (src/frame_replay.rs, the same
// renderer the live page uses) and report what each pass prefix looks like, so a phone-only
// picture defect is located pass by pass, then draw by draw, against the desktop's run of the
// same job.
//
// params: {
//   frame:     "<path under runner assets>" - a frame capsule; send phones a SLIM one
//              (`frame-replay <in.frame> --slim <out.frame>`, typically 50x smaller)
//   limits?:   [N, ...] pass prefixes to render (0 = the whole frame); default [0]
//   each?:     true = every prefix 1..passes (overrides limits)
//   drawLimit?: cut the LAST rendered pass to its first N draws
//   png?:      true = include the whole frame (limit 0) as a PNG data URL
//   knobs?:    { NAME: "value" } on top of the live page's base knobs (below)
// }
// The LIVE page's base knobs are applied first: the renderer's defaults are NOT the product's,
// and a replay without `VITASLOP_GXP_LIVE=1` drew a baseball title's frame through the fixed-function
// fallback - a whole-screen white HUD draw - while the page it was meant to reproduce was right.
const BASE_KNOBS = { VITASLOP_GXP_LIVE: "1", VITASLOP_FRAME_TOPUP: "0" };
// Per prefix: a hash of the pixels, the mean colour, and a 16x9 grid of cell means - enough to
// say WHICH pass first differs from the desktop and roughly where on screen, without shipping
// megabytes per prefix.

import init, { frame_replay, set_knob } from "/pkg/vitaslop_web.js";
import { cachedAsset } from "../cache.js";

const GX = 16;
const GY = 9;

function summarize(rgba, w, h) {
  let hsh = 0x811c9dc5;
  for (let i = 0; i < rgba.length; i += 7) {
    hsh ^= rgba[i];
    hsh = Math.imul(hsh, 0x01000193) >>> 0;
  }
  const grid = [];
  let tr = 0, tg = 0, tb = 0;
  for (let gy = 0; gy < GY; gy++) {
    for (let gx = 0; gx < GX; gx++) {
      const x0 = Math.floor((gx * w) / GX), x1 = Math.floor(((gx + 1) * w) / GX);
      const y0 = Math.floor((gy * h) / GY), y1 = Math.floor(((gy + 1) * h) / GY);
      let r = 0, g = 0, b = 0, n = 0;
      for (let y = y0; y < y1; y += 2) {
        for (let x = x0; x < x1; x += 2) {
          const k = (y * w + x) * 4;
          r += rgba[k];
          g += rgba[k + 1];
          b += rgba[k + 2];
          n++;
        }
      }
      grid.push([Math.round(r / n), Math.round(g / n), Math.round(b / n)]);
      tr += r / n; tg += g / n; tb += b / n;
    }
  }
  const cells = GX * GY;
  return { hash: hsh.toString(16).padStart(8, "0"), mean: [tr / cells, tg / cells, tb / cells].map((v) => Math.round(v)), grid };
}

async function png(rgba, w, h) {
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext("2d");
  ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, rgba.byteLength), w, h), 0, 0);
  const blob = await c.convertToBlob({ type: "image/png" });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return `data:image/png;base64,${btoa(s)}`;
}

export async function run(params, { progress, asset }) {
  if (!params.frame) throw new Error("params.frame is required");
  // `params.console` (a regex source): the renderer's console lines matching it ride back in
  // the result - its diagnostics (`VITASLOP_GXP_BIND_TRACE`, refused pipelines) go to the
  // console, which a job cannot otherwise return.
  const consoleLines = [];
  if (params.console) {
    const re = new RegExp(params.console);
    for (const level of ["log", "warn", "error", "info"]) {
      const orig = console[level].bind(console);
      console[level] = (...a) => {
        const t = a.map(String).join(" ");
        if (consoleLines.length < 400 && re.test(t)) consoleLines.push(t.slice(0, 1500));
        orig(...a);
      };
    }
  }
  progress("loading the renderer");
  await init();
  const knobs = { ...BASE_KNOBS, ...(params.knobs || {}) };
  for (const [k, v] of Object.entries(knobs)) set_knob(k, String(v));
  // Downloaded once per device and verified by SHA-256 on every reuse - see ../cache.js.
  const got = await cachedAsset(params.frame, progress);
  const bytes = got.bytes;
  const fetchMs = Math.round(got.ms);
  let limits = params.limits ?? [0];
  if (params.each) {
    // Every prefix: the pass count comes from a first render of the whole frame.
    const [whole] = await frame_replay(bytes.slice(), new Uint32Array([0]), -1);
    limits = Array.from({ length: whole.passes }, (_, i) => i + 1);
  }
  progress(`rendering ${limits.length} prefix(es)`);
  const t0 = performance.now();
  const res = await frame_replay(bytes, new Uint32Array(limits), params.drawLimit ?? -1);
  const out = [];
  let full = null;
  for (const r of res) {
    const s = summarize(r.rgba, r.width, r.height);
    out.push({ limit: r.limit, passes: r.passes, draws: r.draws, ms: Math.round(r.ms), errors: r.errors, ...(r.gpu ? { gpu: r.gpu.slice(0, 1500) } : {}), ...s });
    if (params.png && r.limit === r.passes && !full) full = await png(r.rgba, r.width, r.height);
  }
  return {
    knobs,
    fetchMs,
    cached: got.cached,
    summary: `${got.cached ? "cached" : "downloaded"} frame in ${fetchMs} ms; ${out.length} prefix(es) of ${res[0]?.passes ?? "?"} passes in ${Math.round(performance.now() - t0)} ms; whole-frame mean ${JSON.stringify(out[out.length - 1]?.mean)}`,
    prefixes: out,
    png: full,
    console: consoleLines,
  };
}
