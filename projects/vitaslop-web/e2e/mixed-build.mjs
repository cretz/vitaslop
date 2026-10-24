// A page that loads during a deploy can pair one build's glue JS with another build's wasm, and
// instantiating then fails with a LinkError. The page must refetch the bundles past the HTTP cache
// and reload ONCE (store.js `recoverFromMixedBuild`), ending up on a working library screen.
//
// Simulated here: the FIRST response for pkg/vitaslop_web.js has one import renamed, so the wasm
// asks for a function the glue no longer provides - the exact error a mixed load gives. Every
// later response is the real file. A second case serves the broken glue EVERY time and checks
// that the page tries one reload and then stops, instead of reloading forever.
//
// The same two cases then break ONLY pkg-threads/vitaslop_web.js, which the page never loads
// itself: the run worker does, on Play. The page's own `pkg` loads fine after the reload, and that
// must not clear the threads bundle's retry guard (one reload, then the error, not one per Play).
// That needs a title to play: the thread-event conformance program (VELF overrides it).
//
// Run: node mixed-build.mjs (after a web build)
import { requireBundle, startServer, webDir, here, launchChrome } from "./harness.mjs";
import { readFile, mkdtemp, mkdir, copyFile, writeFile, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";

requireBundle();
const server = await startServer(webDir);
const url = `http://127.0.0.1:${server.address().port}/`;
/// The glue at `rel` with one import renamed, so its wasm no longer links against it.
async function breakGlue(rel) {
  const glue = await readFile(join(webDir, rel), "utf8");
  const name = /__wbg_call_[0-9a-f]+/.exec(glue);
  if (!name) throw new Error(`no __wbg_call_ import in ${rel} to break`);
  return glue.split(name[0]).join(name[0] + "_stale");
}
const broken = await breakGlue("pkg/vitaslop_web.js");

let failures = 0;
const check = (ok, what) => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${what}`);
  if (!ok) failures += 1;
};

async function run(breakEvery) {
  const browser = await launchChrome();
  const page = await (await browser.newContext()).newPage();
  let served = 0;
  let loads = 0;
  page.on("load", () => (loads += 1));
  await page.route("**/pkg/vitaslop_web.js", async (route) => {
    served += 1;
    if (breakEvery || served === 1) {
      await route.fulfill({ status: 200, contentType: "text/javascript", body: broken });
    } else {
      await route.continue();
    }
  });
  await page.goto(url);
  await page.waitForTimeout(6000);
  const text = await page.locator("#view").innerText().catch(() => "");
  await browser.close();
  return { loads, served, error: /Something went wrong/.test(text), text: text.slice(0, 160).replace(/\s+/g, " ") };
}

const once = await run(false);
check(once.loads === 2 && !once.error, `a mixed load reloads once and recovers (loads ${once.loads}, glue served ${once.served}, ${once.error ? "error shown" : "no error"})`);
const always = await run(true);
check(always.loads === 2, `a load that stays broken tries one reload and does not loop (loads ${always.loads} in 6 s)`);

/// A minimal `param.sfo` carrying TITLE_ID and TITLE (the layout `ingest/sfo.rs` reads).
function sfo(fields) {
  let keyTable = Buffer.alloc(0);
  let dataTable = Buffer.alloc(0);
  const entries = [];
  for (const [k, v] of Object.entries(fields)) {
    const value = Buffer.from(v + "\0", "utf8");
    const max = (value.length + 3) & ~3;
    entries.push({ keyOff: keyTable.length, len: value.length, max, dataOff: dataTable.length });
    keyTable = Buffer.concat([keyTable, Buffer.from(k + "\0", "ascii")]);
    dataTable = Buffer.concat([dataTable, value, Buffer.alloc(max - value.length)]);
  }
  while (keyTable.length % 4) keyTable = Buffer.concat([keyTable, Buffer.alloc(1)]);
  const head = Buffer.alloc(20);
  head.write("\0PSF", 0, "binary");
  head.writeUInt32LE(0x0101, 4);
  head.writeUInt32LE(20 + entries.length * 16, 8);
  head.writeUInt32LE(20 + entries.length * 16 + keyTable.length, 12);
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

const velf = process.env.VELF || join(here, "..", "..", "vitaslop-conformance-suite-vita", "tevent-src", "tevent.velf");
const brokenThreads = await breakGlue("pkg-threads/vitaslop_web.js");
const fatalText = (page) => page.$eval("#fatal", (e) => (e.hidden ? "" : document.getElementById("fatal-text").textContent)).catch(() => "");

/// Import the program, press Play, and see what the threads glue did. `context.route`, not
/// `page.route`: the glue is requested by the run worker, not the page.
async function runThreads(breakEvery, app) {
  const work = await mkdtemp(join(tmpdir(), "vitaslop-mixed-"));
  const context = await chromium.launchPersistentContext(join(work, "profile"), {
    channel: process.env.PWCHANNEL || "chrome",
    headless: !process.env.HEADED,
    args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--use-angle=default", "--autoplay-policy=no-user-gesture-required"],
  });
  try {
    const page = await context.newPage();
    let served = 0;
    let loads = 0;
    page.on("load", () => (loads += 1));
    await context.route("**/pkg-threads/vitaslop_web.js", async (route) => {
      served += 1;
      if (breakEvery || served === 1) await route.fulfill({ status: 200, contentType: "text/javascript", body: brokenThreads });
      else await route.continue();
    });
    await page.goto(url, { waitUntil: "load" });
    await page.waitForSelector("#grid, .card.error", { state: "attached", timeout: 30000 });
    await page.goto(url + "#/import");
    await page.waitForSelector("#f-dir", { state: "attached" });
    await page.setInputFiles("#f-dir", app);
    await page.waitForFunction(() => location.hash.startsWith("#/title/") || document.querySelector(".card.error"), null, { timeout: 120000 });
    const play = async () => {
      await page.waitForSelector("#play:not([disabled])", { timeout: 30000 });
      await page.click("#play");
    };
    await play();
    // The mixed threads glue should send the page round one reload, back to the title page. Wait
    // for the LOAD, not the page: the title page is already showing before the reload starts.
    for (let t = 0; t < 600 && loads < 2; t++) await page.waitForTimeout(100);
    await page.waitForFunction(() => location.hash.startsWith("#/title/") && document.getElementById("play"), null, { timeout: 60000 }).catch(() => {});
    const loadsAfterFirst = loads;
    await play().catch(() => {});
    await page.waitForTimeout(10000);
    return { loads, loadsAfterFirst, served, fatal: (await fatalText(page)).slice(0, 200).replace(/\s+/g, " ") };
  } finally {
    await context.close();
    await rm(work, { recursive: true, force: true }).catch(() => {});
  }
}

if (!existsSync(velf)) {
  check(false, `threads-only cases need the program to play: ${velf} (or VELF=<path>)`);
} else {
  const work = await mkdtemp(join(tmpdir(), "vitaslop-mixed-app-"));
  const app = join(work, "app");
  await mkdir(join(app, "sce_sys"), { recursive: true });
  await copyFile(velf, join(app, "eboot.bin"));
  await writeFile(join(app, "sce_sys", "param.sfo"), sfo({ TITLE_ID: "VSLP00003", TITLE: "vitaslop mixed build test" }));
  const linkFatal = /COULD NOT START[\s\S]*(LinkError|import requires a callable)/;
  const t1 = await runThreads(false, app);
  check(t1.served > 0, `the play loaded pkg-threads (threads glue served ${t1.served}; 0 = the page did not pick the threads bundle)`);
  check(
    t1.loadsAfterFirst === 2 && t1.loads === 2 && !linkFatal.test(t1.fatal),
    `a mixed threads bundle reloads once and the next Play starts (loads ${t1.loadsAfterFirst} then ${t1.loads}, ${t1.fatal ? "fatal: " + t1.fatal : "no fatal"})`
  );
  const t2 = await runThreads(true, app);
  check(
    t2.loadsAfterFirst === 2 && t2.loads === 2 && linkFatal.test(t2.fatal),
    `a threads bundle that stays broken reloads once, then shows the error on the next Play (loads ${t2.loadsAfterFirst} then ${t2.loads}, ${t2.fatal ? "fatal: " + t2.fatal : "no fatal"})`
  );
  await rm(work, { recursive: true, force: true }).catch(() => {});
}

server.close();
console.log(failures ? `\nmixed-build: ${failures} FAILED` : "\nmixed-build: ALL PASS");
process.exit(failures ? 1 : 0);
