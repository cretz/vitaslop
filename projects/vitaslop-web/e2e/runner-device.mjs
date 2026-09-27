// runner-device.mjs - be a DEVICE for the runner: open `/runner/` in headless Chrome against a
// running serve.mjs and let it take jobs until the queue is empty (or a time bound passes).
//
// Two uses: the runner's own regression check (queue jobs, run this, read the results), and a
// DESKTOP REFERENCE for any job a phone runs - the same job on the desktop GPU is the other half
// of every phone-vs-desktop comparison.
//
//   URL=https://127.0.0.1:8443/runner/ NAME=desktop MAX_S=120 node e2e/runner-device.mjs

import { chromium } from "playwright";

const url = process.env.URL || "https://127.0.0.1:8443/runner/";
const name = process.env.NAME || "desktop-headless";
const maxS = Number(process.env.MAX_S || 120);
const base = new URL(url).origin;

// The installed Chrome (the project's launcher), not Playwright's bundled shell: the same GPU
// stack every other desktop measurement here uses.
const browser = await chromium.launch({
  channel: process.env.PWCHANNEL || "chrome",
  headless: process.env.HEADED ? false : true,
  args: ["--ignore-certificate-errors", "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=default"],
});
const ctx = await browser.newContext({ ignoreHTTPSErrors: true });
const page = await ctx.newPage();
// VERBOSE=1 prints every console line, the job Workers' included (the renderer's warnings
// land there); otherwise only errors.
page.on("console", (m) => {
  if (m.type() === "error" || process.env.VERBOSE) console.log(`[page ${m.type()}] ${m.text()}`);
});
page.on("worker", (w) => w.on("console", (m) => {
  if (m.type() === "error" || process.env.VERBOSE) console.log(`[worker ${m.type()}] ${m.text()}`);
}));
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
// A FIXED device id per NAME, so the desktop reference accumulates under one device rather
// than a new one per run (a fresh browser context has an empty localStorage).
const devId = process.env.DEVICE || [...name].reduce((h, c) => (Math.imul(h, 31) + c.charCodeAt(0)) >>> 0, 7).toString(16).padStart(8, "0").slice(-8);
await page.addInitScript(([n, d]) => {
  localStorage.setItem("vitaslop.runner.name", n);
  localStorage.setItem("vitaslop.runner.device", d);
}, [name, devId]);
await page.goto(url, { waitUntil: "load" });
const t0 = Date.now();
const status = async () => (await (await fetch(`${base}/runner/status`)).json());
process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";
let idleSince = null;
for (;;) {
  await new Promise((r) => setTimeout(r, 1500));
  const s = await status();
  const mine = await page.evaluate(() => document.getElementById("log").innerText);
  const busy = s.queue.length > 0 || (await page.evaluate(() => document.getElementById("state").textContent)).startsWith("running");
  if (!busy) {
    idleSince ??= Date.now();
    if (Date.now() - idleSince > 3000) {
      console.log(mine.split("\n").reverse().join("\n"));
      break;
    }
  } else idleSince = null;
  if ((Date.now() - t0) / 1000 > maxS) {
    console.log(`TIMED OUT after ${maxS}s with ${s.queue.length} job(s) queued`);
    console.log(mine.split("\n").reverse().join("\n"));
    await browser.close();
    process.exit(1);
  }
}
await browser.close();
