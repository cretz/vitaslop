// homebrew.mjs - run one authored conformance program (a velf from
// vitaslop-conformance-suite-vita) through the real player, and check what it PRINTED.
//
// The harness's native tests run these programs on wasmtime; this runs the same file through the
// import screen and Play, on whichever bundle the page picks (the threads bundle and its SMP
// workers on a cross-origin-isolated page), and reads the title's debug console out of the
// end-of-run diagnostics (`guest console`). So a guest-visible behaviour that depends on the
// browser's own scheduler - which the native tests never run - is checked against the same
// expected output.
//
// Env: VELF=<path to the .velf> (default: the thread-event program), EXPECT=<exact console text>
//      (default: that program's), HEADLESS=1, WAIT_MS (default 60000), SHOT_DIR
// Run: HEADLESS=1 node homebrew.mjs

import { mkdtemp, mkdir, copyFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";
import { startServer, webDir, here } from "./harness.mjs";

const suite = join(here, "..", "..", "vitaslop-conformance-suite-vita");
const velf = process.env.VELF || join(suite, "tevent-src", "tevent.velf");
const expect = process.env.EXPECT ?? "ax5sbx7sax5sax5scM4";
const waitMs = Number(process.env.WAIT_MS || 60000);
const shotDir = process.env.SHOT_DIR || join(here, "screenshots", "homebrew");

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

const work = await mkdtemp(join(tmpdir(), "vitaslop-homebrew-"));
const app = join(work, "app");
await mkdir(join(app, "sce_sys"), { recursive: true });
await copyFile(velf, join(app, "eboot.bin"));
await writeFile(join(app, "sce_sys", "param.sfo"), sfo({ TITLE_ID: "VSLP00002", TITLE: "vitaslop homebrew test" }));
await mkdir(shotDir, { recursive: true });

const server = await startServer(webDir);
const url = `http://127.0.0.1:${server.address().port}/`;
const context = await chromium.launchPersistentContext(join(work, "profile"), {
  channel: process.env.PWCHANNEL || "chrome",
  headless: !!process.env.HEADLESS,
  viewport: { width: 1100, height: 800 },
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--enable-gpu", "--use-angle=default", "--autoplay-policy=no-user-gesture-required"],
});
const page = await context.newPage();
const logs = [];
page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
let ok = false;
try {
  await page.goto(url, { waitUntil: "load" });
  await page.waitForSelector("#grid, .card.error", { state: "attached", timeout: 30000 });
  await page.goto(url + "#/import");
  await page.waitForSelector("#f-dir", { state: "attached" });
  await page.setInputFiles("#f-dir", app);
  await page.waitForFunction(() => location.hash.startsWith("#/title/") || document.querySelector(".card.error"), null, { timeout: 120000 });
  if (await page.$(".card.error")) throw new Error("import: " + (await page.$eval(".card.error", (e) => e.innerText)));
  await page.waitForSelector("#play", { timeout: 30000 });
  await page.click("#play");
  await page.waitForFunction(() => !document.getElementById("player").hidden, null, { timeout: 10000 });
  // The program prints its verdict and ends the process; the end-of-run report carries it.
  await page.click("#menubtn").catch(() => {});
  const t0 = Date.now();
  let diag = "";
  while (Date.now() - t0 < waitMs) {
    await page.waitForTimeout(1000);
    diag = await page.$eval("#m-diag", (e) => e.textContent).catch(() => "");
    if (/guest console \(/.test(diag) || /\bENDED\b/.test(diag)) break;
    await page.click("#menubtn").catch(() => {});
  }
  await writeFile(join(shotDir, "diag.txt"), diag);
  const bundle = /this run's bundle: ([\w-]+)/.exec(diag)?.[1];
  const status = /status: (.*)/.exec(diag)?.[1] || "";
  const consoleText = /guest console \([^)]*\):\n([\s\S]*?)\n\n/.exec(diag + "\n\n")?.[1];
  console.log(`bundle: ${bundle}\nstatus: ${status}\nconsole: ${JSON.stringify(consoleText)}`);
  if (consoleText === undefined) throw new Error("no guest console in the diagnostics");
  if (consoleText !== expect) throw new Error(`console ${JSON.stringify(consoleText)} != expected ${JSON.stringify(expect)}`);
  ok = true;
} catch (e) {
  console.error("FAIL:", e.message);
  await page.screenshot({ path: join(shotDir, "fail.png") }).catch(() => {});
  await writeFile(join(shotDir, "console.txt"), logs.join("\n")).catch(() => {});
} finally {
  await context.close();
  server.close();
  await rm(work, { recursive: true, force: true }).catch(() => {});
}
console.log(ok ? "[homebrew] PASS" : "[homebrew] FAIL");
process.exit(ok ? 0 : 1);
