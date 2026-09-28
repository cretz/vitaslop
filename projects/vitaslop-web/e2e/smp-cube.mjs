// smp-cube.mjs - the MULTI-THREADED product path, end to end, on a homebrew guest.
//
// The player runs a title on the wasm-THREADS bundle with its guest threads spread over several
// workers (`VITASLOP_SMP`, on by default) whenever the page is cross-origin isolated - which is
// every real phone. `run.mjs` boots the cube on the single-threaded bundle and never touches that
// path, so this imports the cube's own homebrew eboot (`vitaslop-conformance-suite-vita/cube-src`)
// through the real import screen, presses Play, and requires from the in-game diagnostics that
// the run is on `pkg-threads`, that the SMP workers ran, and that frames advanced.
//
// Env: HEADLESS=1 (CI), SHOT_DIR=<dir> (default ./screenshots/smp-cube), PLAY_MS (default 8000),
//      BASE_URL=<url> (run against an already-served site instead - e.g. a staged Pages build
//      on a plain static server, where `coi.js` must supply the isolation),
//      ALLOW_SOFTWARE=1 (CI: a software adapter; the player refuses one unless the link arms
//      VITASLOP_ALLOW_SOFTWARE_GPU, and it renders slowly, so fewer frames are required)
// Run: node smp-cube.mjs

import { mkdtemp, mkdir, copyFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";
import { startServer, webDir, here } from "./harness.mjs";

const shotDir = process.env.SHOT_DIR || join(here, "screenshots", "smp-cube");
const playMs = Number(process.env.PLAY_MS || 8000);
const EBOOT = join(here, "..", "..", "vitaslop-conformance-suite-vita", "cube-src", "cube.eboot.bin");

/// A minimal `param.sfo` carrying TITLE_ID and TITLE (the layout `ingest/sfo.rs` reads).
function sfo(fields) {
  const keys = Object.keys(fields);
  let keyTable = Buffer.alloc(0);
  let dataTable = Buffer.alloc(0);
  const entries = [];
  for (const k of keys) {
    const value = Buffer.from(fields[k] + "\0", "utf8");
    const max = (value.length + 3) & ~3;
    entries.push({ keyOff: keyTable.length, len: value.length, max, dataOff: dataTable.length });
    keyTable = Buffer.concat([keyTable, Buffer.from(k + "\0", "ascii")]);
    dataTable = Buffer.concat([dataTable, value, Buffer.alloc(max - value.length)]);
  }
  while (keyTable.length % 4) keyTable = Buffer.concat([keyTable, Buffer.alloc(1)]);
  const keyTableOff = 20 + entries.length * 16;
  const dataTableOff = keyTableOff + keyTable.length;
  const head = Buffer.alloc(20);
  head.write("\0PSF", 0, "binary");
  head.writeUInt32LE(0x0101, 4);
  head.writeUInt32LE(keyTableOff, 8);
  head.writeUInt32LE(dataTableOff, 12);
  head.writeUInt32LE(entries.length, 16);
  const index = Buffer.alloc(entries.length * 16);
  entries.forEach((e, i) => {
    index.writeUInt16LE(e.keyOff, i * 16);
    index.writeUInt16LE(0x0204, i * 16 + 2);
    index.writeUInt32LE(e.len, i * 16 + 4);
    index.writeUInt32LE(e.max, i * 16 + 8);
    index.writeUInt32LE(e.dataOff, i * 16 + 12);
  });
  return Buffer.concat([head, index, keyTable, dataTable]);
}

const work = await mkdtemp(join(tmpdir(), "vitaslop-smp-cube-"));
const app = join(work, "cube");
await mkdir(join(app, "sce_sys"), { recursive: true });
await copyFile(EBOOT, join(app, "eboot.bin"));
await writeFile(join(app, "sce_sys", "param.sfo"), sfo({ TITLE_ID: "VSLP00001", TITLE: "vitaslop cube" }));
await mkdir(shotDir, { recursive: true });

// Cross-origin isolated, like the hosted page: without it there is no SharedArrayBuffer and the
// player falls back to the single-threaded bundle, which is exactly what this must not test.
const server = process.env.BASE_URL ? null : await startServer(webDir);
const software = !!process.env.ALLOW_SOFTWARE;
const base = process.env.BASE_URL || `http://127.0.0.1:${server.address().port}/`;
const url = base + (software ? "?knobs=VITASLOP_ALLOW_SOFTWARE_GPU=1" : "");
const context = await chromium.launchPersistentContext(join(work, "profile"), {
  channel: process.env.PWCHANNEL || "chrome",
  headless: !!process.env.HEADLESS,
  viewport: { width: 1100, height: 800 },
  // CI runs this under `xvfb-run`: without a display, Chrome's GPU process failed to start seven
  // times, the canvas stopped presenting ("1 shown of 60 run") and the WebGPU device was
  // destroyed seven seconds in (PR #4).
  args: [
    "--enable-unsafe-webgpu", "--enable-features=Vulkan", "--enable-gpu", "--use-angle=default", "--autoplay-policy=no-user-gesture-required",
    ...(software ? ["--enable-unsafe-swiftshader"] : []),
    ...(process.env.FORCE_SWIFTSHADER ? ["--use-webgpu-adapter=swiftshader"] : []),
  ],
});
const page = await context.newPage();
const logs = [];
page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
let ok = false;
try {
  await page.goto(url, { waitUntil: "load" });
  await page.waitForSelector("#grid, .card.error", { state: "attached", timeout: 30000 });
  if (!(await page.evaluate(() => self.crossOriginIsolated))) throw new Error("the page is not cross-origin isolated");
  await page.goto(url + "#/import");
  await page.waitForSelector("#f-dir", { state: "attached" });
  await page.setInputFiles("#f-dir", app);
  await page.waitForFunction(() => location.hash.startsWith("#/title/") || document.querySelector(".card.error"), null, { timeout: 120000 });
  if (await page.$(".card.error")) throw new Error("import: " + (await page.$eval(".card.error", (e) => e.innerText)));
  console.log("imported:", await page.evaluate(() => location.hash));
  await page.waitForSelector("#play", { timeout: 30000 });
  await page.click("#play");
  await page.waitForFunction(() => !document.getElementById("player").hidden, null, { timeout: 10000 });
  await page.waitForFunction(() => document.getElementById("loading").hidden || !document.getElementById("fatal").hidden, null, { timeout: 120000 });
  if (!(await page.$eval("#fatal", (e) => e.hidden))) throw new Error("fatal: " + (await page.$eval("#fatal-text", (e) => e.innerText)));
  // Watched second by second rather than slept through, so a run that dies says WHEN (wall
  // time and frame) instead of surfacing as a click that timed out on the fault panel.
  const t0 = Date.now();
  while (Date.now() - t0 < playMs) {
    await page.waitForTimeout(1000);
    const s = await page.evaluate(() => ({
      fatal: !document.getElementById("fatal").hidden,
      text: document.getElementById("fatal-text")?.innerText || "",
      fps: document.getElementById("fpsbadge")?.textContent || "",
      title: document.title,
    }));
    console.log(`  +${((Date.now() - t0) / 1000).toFixed(0)} s: ${s.fatal ? "FATAL" : "running"} ${s.fps}`);
    if (s.fatal) throw new Error(`the run stopped after ${((Date.now() - t0) / 1000).toFixed(1)} s: ${s.text.slice(0, 300)}`);
  }
  await page.screenshot({ path: join(shotDir, "playing.png") });
  await page.click("#menubtn");
  await page.waitForFunction(() => (document.getElementById("m-diag")?.textContent || "").length > 100, null, { timeout: 10000 });
  const diag = await page.$eval("#m-diag", (e) => e.textContent);
  await writeFile(join(shotDir, "diag.txt"), diag);
  const bundle = /this run's bundle: ([\w-]+)/.exec(diag)?.[1];
  const smp = /SMP (\d+) worker\(s\)/.exec(diag)?.[1];
  const frame = Number(/status: frame (\d+)/.exec(diag)?.[1] || 0);
  const errors = logs.filter((l) => /^\[(pageerror|error)\]/.test(l) && !/Failed to load resource/.test(l));
  console.log(`bundle: ${bundle}; SMP workers: ${smp}; frame: ${frame}; page errors: ${errors.length}`);
  for (const e of errors.slice(0, 10)) console.log("  " + e);
  if (bundle !== "pkg-threads") throw new Error(`ran on ${bundle}, not the threads bundle`);
  if (!(Number(smp) >= 1)) throw new Error("no SMP worker line in the diagnostics - the parallel run did not start");
  const minFrames = software ? 10 : 60;
  if (frame < minFrames) throw new Error(`only ${frame} frames in ${playMs} ms (need ${minFrames})`);
  if (errors.length) throw new Error(`${errors.length} page error(s)`);
  ok = true;
} catch (e) {
  console.error("FAIL:", e.message);
  await page.screenshot({ path: join(shotDir, "fail.png") }).catch(() => {});
  await writeFile(join(shotDir, "console.txt"), logs.join("\n")).catch(() => {});
} finally {
  await context.close();
  server?.close();
  await rm(work, { recursive: true, force: true }).catch(() => {});
}
console.log(ok ? "[smp-cube] PASS" : "[smp-cube] FAIL");
process.exit(ok ? 0 : 1);
