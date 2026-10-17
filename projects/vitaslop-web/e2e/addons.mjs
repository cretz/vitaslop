// Add-ons through the SHIPPED pages: a title's page lists none, its "Add add-ons" import adds
// each picked .pkg + work.bin, the list shows them, one is removed and added back, and the
// library's own "Add games" import refuses an add-on (filed under its title's id it would
// replace the game). Every step is a real click or file pick on index.html; nothing here
// writes storage behind the page's back except the library RECORD of the owning title, so
// a title imported by another harness (game-boot) has a page to open.
//
// Env:  TITLE_ID     the owning title, already imported into this profile's OPFS
//       ADDONS       ;-separated add-on folders, each holding one .pkg and its work.bin
//       PROFILE_DIR  the persistent browser profile (the same one game-boot uses)
//       PORT         the origin's port - OPFS is per origin, so it must match game-boot's
//       SHOT_DIR     where page screenshots land
//       HEADLESS=1   no window (the real GPU either way; this test draws no frames)
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, readdir, mkdir } from "node:fs/promises";
import { join, extname } from "node:path";
import { fileURLToPath } from "node:url";

const here = join(fileURLToPath(import.meta.url), "..");
const webDir = join(here, "..", "web");
const titleId = process.env.TITLE_ID || "";
const addonDirs = (process.env.ADDONS || "").split(";").filter(Boolean);
const profileDir = process.env.PROFILE_DIR || "";
const port = Number(process.env.PORT || 8791);
const shotDir = process.env.SHOT_DIR || join(here, "..", "..", "..", "target", "addons-e2e");
if (!titleId || !addonDirs.length || !profileDir) {
  console.error("[addons] TITLE_ID, ADDONS and PROFILE_DIR are required");
  process.exit(2);
}
const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml" };

/// The .pkg and work.bin in an add-on folder.
async function pickFiles(dir) {
  const names = await readdir(dir);
  const pkg = names.find((n) => n.toLowerCase().endsWith(".pkg"));
  const work = names.find((n) => n.toLowerCase() === "work.bin");
  if (!pkg || !work) throw new Error(`${dir} holds no .pkg + work.bin`);
  return { pkg: join(dir, pkg), work: join(dir, work) };
}

let failures = 0;
const check = (ok, what) => {
  console.log(`[addons] ${ok ? "PASS" : "FAIL"} ${what}`);
  if (!ok) failures += 1;
};

const server = createServer(async (req, res) => {
  const coi = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
  try {
    const url = decodeURIComponent(req.url.split("?")[0]);
    const file = join(webDir, url === "/" ? "index.html" : url);
    const body = await readFile(file);
    res.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream", ...coi });
    res.end(body);
  } catch {
    res.writeHead(404, coi);
    res.end("not found");
  }
});
await new Promise((resolve, reject) => {
  server.once("error", reject);
  server.listen(port, "127.0.0.1", resolve);
});
await mkdir(shotDir, { recursive: true });

const ctx = await chromium.launchPersistentContext(profileDir, {
  headless: !!process.env.HEADLESS,
  channel: "chrome",
  args: ["--enable-unsafe-webgpu", "--enable-gpu"],
  viewport: { width: 1100, height: 900 },
});
const page = ctx.pages()[0] || (await ctx.newPage());
page.on("console", (m) => m.type() === "error" && console.log(`[page error] ${m.text()}`));
page.on("dialog", (d) => d.accept());
const url = `http://127.0.0.1:${port}/`;
const shot = (name) => page.screenshot({ path: join(shotDir, `${name}.png`), fullPage: true });
const listed = () => page.locator("#ad-list li").count();

try {
  await page.goto(url, { waitUntil: "load" });
  // The owning title's library record, so its page exists. Only written when absent: a
  // title imported through the page already has one.
  await page.evaluate(async (id) => {
    const root = await navigator.storage.getDirectory();
    const lib = await root.getDirectoryHandle("library", { create: true });
    const dir = await lib.getDirectoryHandle(id, { create: true });
    try {
      await dir.getFileHandle("meta.json");
      return;
    } catch {}
    const fh = await dir.getFileHandle("meta.json", { create: true });
    const w = await fh.createWritable();
    await w.write(JSON.stringify({ titleId: id, title: id, contentId: "", appVersion: "", sourceKind: "dump", bytes: 0, files: 0, importedAt: Date.now(), lastPlayedAt: 0 }));
    await w.close();
  }, titleId);
  // Start clean: no add-ons from an earlier run of this test.
  await page.evaluate(async (id) => {
    const root = await navigator.storage.getDirectory();
    try {
      const top = await root.getDirectoryHandle("addons");
      await top.removeEntry(id, { recursive: true });
    } catch {}
  }, titleId);

  await page.goto(`${url}#/title/${titleId}`);
  await page.waitForSelector("#ad-list", { state: "attached" });
  await page.waitForFunction(() => !document.getElementById("ad-none").hidden);
  check((await listed()) === 0, "a title with no add-ons lists none");
  await shot("1-title-none");

  const files = [];
  for (const d of addonDirs) files.push(await pickFiles(d));
  for (const [i, f] of files.entries()) {
    await page.goto(`${url}#/title/${titleId}`);
    await page.click("text=Add add-ons");
    await page.waitForSelector("#f-pkg", { state: "attached" });
    check((await page.locator('#tabs .tab[data-mode="vpk"]').count()) === 0, "the add-on import offers no homebrew tab");
    await page.setInputFiles("#f-pkg", f.pkg);
    await page.setInputFiles("#f-work", f.work);
    if (i === 0) await shot("2-addon-import");
    await page.click("#go-pkg");
    await page.waitForURL(`**/#/title/${titleId}`, { timeout: 300000 });
    await page.waitForFunction((n) => document.querySelectorAll("#ad-list li").length === n, i + 1, { timeout: 30000 });
    check(true, `add-on ${i + 1} imported and listed (${f.pkg})`);
  }
  await shot("3-title-listed");
  const names = await page.locator("#ad-list li b").allTextContents();
  console.log(`[addons] listed: ${names.join(", ")}`);

  // Remove the first, then add it back.
  const firstDir = await page.locator("#ad-list button[data-addon]").first().getAttribute("data-addon");
  await page.locator("#ad-list button[data-addon]").first().click();
  await page.waitForFunction((n) => document.querySelectorAll("#ad-list li").length === n, files.length - 1);
  check(!(await page.locator(`#ad-list button[data-addon="${firstDir}"]`).count()), `removing ${firstDir} takes it off the list`);
  const stored = await page.evaluate(async ([id, dir]) => {
    const root = await navigator.storage.getDirectory();
    try {
      await (await (await root.getDirectoryHandle("addons")).getDirectoryHandle(id)).getDirectoryHandle(dir);
      return true;
    } catch {
      return false;
    }
  }, [titleId, firstDir]);
  check(!stored, `removing ${firstDir} deletes its stored files`);
  await page.click("text=Add add-ons");
  await page.waitForSelector("#f-pkg", { state: "attached" });
  const back = files.find((f) => f.pkg.includes(firstDir)) || files[0];
  await page.setInputFiles("#f-pkg", back.pkg);
  await page.setInputFiles("#f-work", back.work);
  await page.click("#go-pkg");
  await page.waitForURL(`**/#/title/${titleId}`, { timeout: 300000 });
  await page.waitForFunction((n) => document.querySelectorAll("#ad-list li").length === n, files.length, { timeout: 30000 });
  check(true, "a removed add-on can be added back");

  // The library's own import must refuse an add-on, and say where it goes instead.
  await page.goto(`${url}#/import`);
  await page.waitForSelector("#f-pkg", { state: "attached" });
  await page.setInputFiles("#f-pkg", files[0].pkg);
  await page.setInputFiles("#f-work", files[0].work);
  await page.click("#go-pkg");
  await page.waitForSelector("#result .card.error", { timeout: 120000 });
  const err = await page.locator("#result .card.error").textContent();
  check(/add-on for/.test(err) && /Add-ons/.test(err), `the library import refuses an add-on: ${err.replace(/\s+/g, " ").trim().slice(0, 160)}`);
  await shot("4-library-refuses");
  await page.goto(`${url}#/title/${titleId}`);
  await page.waitForFunction((n) => document.querySelectorAll("#ad-list li").length === n, files.length);
  check(true, "the refused import left the title's add-ons alone");
} catch (e) {
  console.log(`[addons] FAIL ${e.stack || e}`);
  failures += 1;
  await shot("error").catch(() => {});
} finally {
  await ctx.close();
  server.close();
}
console.log(failures ? `[addons] ${failures} FAILURE(S)` : "[addons] all checks passed");
process.exit(failures ? 1 : 0);
