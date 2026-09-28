// live-device.mjs - run the runner's LIVE job page (web/runner/live.html) in desktop Chrome
// against a running serve.mjs: the desktop reference for a phone live job, and the way to prove
// a live job works BEFORE a phone is asked to run it.
//
// The title has to be in the page's OPFS, as it is on a phone. A PERSISTENT profile holds it,
// so the one-time import (streamed from serve.mjs's /game/<id>/) happens only on the first run.
//
//   PARAMS='{"titleId":"PCSA00002","stopFrame":600}' OUT=result.json node e2e/live-device.mjs
//
// Env: URL (default https://127.0.0.1:8443), PROFILE_DIR (required), PARAMS (the job params),
// OUT (where the result JSON goes), HEADED=1, MAX_S (default 600).

import { chromium } from "playwright";
import { writeFileSync } from "node:fs";

const base = process.env.URL || "https://127.0.0.1:8443";
const params = JSON.parse(process.env.PARAMS || "{}");
const profile = process.env.PROFILE_DIR;
if (!profile) throw new Error("PROFILE_DIR is required (the title lives in its OPFS)");
const maxS = Number(process.env.MAX_S || 600);

const ctx = await chromium.launchPersistentContext(profile, {
  channel: process.env.PWCHANNEL || "chrome",
  headless: !process.env.HEADED,
  ignoreHTTPSErrors: true,
  viewport: { width: 987, height: 444 },
  args: ["--ignore-certificate-errors", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=default"],
});
const page = ctx.pages()[0] || (await ctx.newPage());
page.on("console", (m) => console.log(`[page ${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
page.on("worker", (w) => w.on("console", (m) => {
  if (m.type() === "error" || process.env.VERBOSE) console.log(`[worker ${m.type()}] ${m.text()}`);
}));
await page.goto(`${base}/runner/live.html`, { waitUntil: "load" });

// Import the title once, the way debug/game-worker.html does.
const imported = await page.evaluate(async (id) => {
  const { isComplete, importTitle } = await import("/opfs.js");
  if (await isComplete(id)) return "already imported";
  const files = await (await fetch(`/game-manifest.json?title=${id}`)).json();
  const entries = files.map((p) => ({
    path: p,
    source: async () => {
      const r = await fetch(`/game/${id}/${p}`);
      if (!r.ok) throw new Error(`GET /game/${id}/${p} -> ${r.status}`);
      const len = r.headers.get("content-length");
      return { body: r.body, bytes: len === null ? null : Number(len) };
    },
  }));
  await importTitle(id, entries, () => {});
  return `imported ${files.length} files`;
}, params.titleId);
console.log(`[live] ${params.titleId}: ${imported}`);
// isComplete also wants the title's library record on the product page; live.js checks only OPFS.

const t0 = Date.now();
const result = await page.evaluate(
  ([p, maxMs]) =>
    new Promise((resolve) => {
      const timer = setTimeout(() => resolve({ status: "timeout" }), maxMs);
      addEventListener("message", (e) => {
        const m = e.data || {};
        if (m.type === "progress") console.log(`[live] ${m.text}`);
        else if (m.type === "result" || m.type === "error") {
          clearTimeout(timer);
          resolve(m.type === "result" ? { status: "ok", ...m.result } : { status: "error", error: m.error });
        }
      });
      postMessage({ type: "job", params: p }, location.origin);
    }),
  [params, maxS * 1000],
);
result.ms = Date.now() - t0;
console.log(`[live] ${result.status} in ${result.ms} ms: ${result.summary || result.error || ""}`);
if (process.env.OUT) writeFileSync(process.env.OUT, JSON.stringify(result, null, 2));
await ctx.close();
