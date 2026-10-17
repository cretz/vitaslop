// A rePatch overlay through the SHIPPED pages: a title's page shows none, "Add from a .zip" lays
// the picked zip over the title, the card lists what it replaces, the switch turns it off and on
// (the record says so; the files stay), "Replace with a folder" swaps it for a picked folder in a
// new generation (the old one's files go), and Remove deletes it - the title's own files are
// never touched. Every step is a real click or file pick on index.html.
//
// Env:  TITLE_ID     the title, already imported into this profile's OPFS
//       OVERLAY_ZIP  a .zip laid out as a player has it (`.../rePatch/<id>/...`)
//       OVERLAY_DIR  a folder laid out the same way
//       PROFILE_DIR  the persistent browser profile (the same one game-boot uses)
//       PORT         the origin's port - OPFS is per origin, so it must match game-boot's
//       SHOT_DIR     where page screenshots land
//       KEEP=1       leave the overlay installed at the end (on), for a game-boot run after
//       HEADLESS=1   no window
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import { join, extname } from "node:path";
import { fileURLToPath } from "node:url";

const here = join(fileURLToPath(import.meta.url), "..");
const webDir = join(here, "..", "web");
const titleId = process.env.TITLE_ID || "";
const zip = process.env.OVERLAY_ZIP || "";
const folder = process.env.OVERLAY_DIR || "";
const profileDir = process.env.PROFILE_DIR || "";
const port = Number(process.env.PORT || 8791);
const shotDir = process.env.SHOT_DIR || join(here, "..", "..", "..", "target", "repatch-e2e");
if (!titleId || !zip || !folder || !profileDir) {
  console.error("[repatch] TITLE_ID, OVERLAY_ZIP, OVERLAY_DIR and PROFILE_DIR are required");
  process.exit(2);
}
const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".css": "text/css", ".png": "image/png", ".svg": "image/svg+xml" };

let failures = 0;
const check = (ok, what) => {
  console.log(`[repatch] ${ok ? "PASS" : "FAIL"} ${what}`);
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
  viewport: { width: 1100, height: 1400 },
});
const page = ctx.pages()[0] || (await ctx.newPage());
page.on("console", (m) => (process.env.DEBUG || m.type() === "error") && console.log(`[page ${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => console.log(`[page exception] ${e.message}`));
page.on("dialog", (d) => d.accept());
const url = `http://127.0.0.1:${port}/`;
const shot = (name) => page.screenshot({ path: join(shotDir, `${name}.png`), fullPage: true });

/// What OPFS holds for the overlay: the record, its generations, and the stored paths of the
/// one the record names. Read straight from storage, so the page's own view is not the witness.
const stored = () =>
  page.evaluate(async (id) => {
    const root = await navigator.storage.getDirectory();
    let top;
    try {
      top = await (await root.getDirectoryHandle("repatch")).getDirectoryHandle(id);
    } catch {
      return null;
    }
    let rec = null;
    try {
      rec = JSON.parse(await (await (await top.getFileHandle("repatch.json")).getFile()).text());
    } catch {}
    const gens = [];
    for await (const [name, h] of top.entries()) if (h.kind === "directory") gens.push(name);
    const files = [];
    if (rec) for await (const [name] of (await top.getDirectoryHandle(rec.gen)).entries()) files.push(name.replace(/%2F/g, "/"));
    return { rec, gens, files: files.sort() };
  }, titleId);
const titleFiles = () =>
  page.evaluate(async (id) => {
    const root = await navigator.storage.getDirectory();
    let n = 0;
    for await (const _ of (await (await root.getDirectoryHandle("games")).getDirectoryHandle(id)).entries()) n += 1;
    return n;
  }, titleId);
const noteIs = (re) => page.waitForFunction((src) => new RegExp(src).test(document.getElementById("rp-note").textContent), re.source, { timeout: Number(process.env.NOTE_MS || 600000) });

try {
  await page.goto(url, { waitUntil: "load" });
  // The title's library record, so its page exists - only when absent (see addons.mjs).
  await page.evaluate(async (id) => {
    const root = await navigator.storage.getDirectory();
    const dir = await (await root.getDirectoryHandle("library", { create: true })).getDirectoryHandle(id, { create: true });
    try {
      await dir.getFileHandle("meta.json");
      return;
    } catch {}
    const w = await (await dir.getFileHandle("meta.json", { create: true })).createWritable();
    await w.write(JSON.stringify({ titleId: id, title: id, contentId: "", appVersion: "", sourceKind: "dump", bytes: 0, files: 0, importedAt: Date.now(), lastPlayedAt: 0 }));
    await w.close();
  }, titleId);
  // Start clean: no overlay from an earlier run.
  await page.evaluate(async (id) => {
    const root = await navigator.storage.getDirectory();
    try {
      await (await root.getDirectoryHandle("repatch")).removeEntry(id, { recursive: true });
    } catch {}
  }, titleId);
  const before = await titleFiles();

  await page.goto(`${url}#/title/${titleId}`);
  await page.waitForSelector("#rp-list", { state: "attached" });
  await page.waitForFunction(() => !document.getElementById("rp-none").hidden);
  check((await page.locator("#rp-list li").count()) === 0, "a title with no overlay lists none");
  await shot("1-none");

  // ---- add from a .zip ----
  await page.setInputFiles("#rp-zip-in", zip);
  await noteIs(/Overlay added|Could not/);
  const note = await page.locator("#rp-note").textContent();
  check(/Overlay added/.test(note), `the zip is laid over the title: ${note}`);
  await page.waitForSelector("#rp-list li");
  const row = await page.locator("#rp-list li").textContent();
  check(/replaces eboot\.bin/.test(row), `the card names the executable it replaces: ${row.replace(/\s+/g, " ").trim()}`);
  await shot("2-added-zip");
  let s = await stored();
  check(s && s.rec && s.rec.titleId === titleId && !s.rec.disabled && /^[0-9a-f]{16}$/.test(s.rec.codeHash), "the record is written, on, with a code hash");
  check(s && s.files.includes("app/eboot.bin") && s.files.some((f) => /^app\/movie_US\//i.test(f)), `the overlay's files are stored at app/...: ${s && s.files.join(", ")}`);
  check(s && !s.files.some((f) => /readme/i.test(f)), "a file outside rePatch/<id>/ is left out");
  check((await titleFiles()) === before, "the title's own stored files are untouched");

  // ---- the switch ----
  // The switch saves asynchronously; wait for the RECORD, not the checkbox.
  const recordSays = async (disabled) => {
    for (let i = 0; i < 100; i++) {
      const r = await stored();
      if (r && r.rec && r.rec.disabled === disabled) return r;
      await page.waitForTimeout(100);
    }
    return stored();
  };
  await page.uncheck("#rp-on");
  s = await recordSays(true);
  check(s && s.rec.disabled === true && s.files.length > 0, "off is recorded and keeps the files");
  await shot("3-off");
  await page.check("#rp-on");
  s = await recordSays(false);
  check(s && s.rec.disabled === false, "on again is recorded");

  // ---- replace with a folder: a new generation, the old one dropped ----
  const oldGen = s.rec.gen;
  await page.setInputFiles("#rp-dir-in", folder);
  // The zip's "added" note is still up (an import's outcome stays on the card), so the witness
  // is storage: the record naming a new generation - then the card's buttons free again.
  for (let i = 0; i < 6000; i++) {
    const r = await stored();
    if (r && r.rec && r.rec.gen !== oldGen) break;
    await page.waitForTimeout(100);
  }
  await page.waitForFunction(() => !document.getElementById("rp-dir").disabled, null, { timeout: 60000 });
  check(/Overlay added/.test(await page.locator("#rp-note").textContent()), "a folder replaces the zip's overlay");
  s = await stored();
  check(s && s.rec.gen !== oldGen && s.gens.length === 1 && s.gens[0] === s.rec.gen, `the replacement is a new generation and the old one is gone (${s && s.gens.join(",")})`);
  check(s && s.rec.source === folder.split(/[\\/]/).filter(Boolean).pop(), `the record names the folder: ${s && s.rec.source}`);
  await shot("4-replaced-folder");

  // ---- remove (or keep for a boot) ----
  if (!process.env.KEEP) {
    await page.click("#rp-rm");
    await page.waitForFunction(() => !document.getElementById("rp-none").hidden);
    check((await stored()) === null, "Remove deletes the overlay");
    check((await titleFiles()) === before, "and leaves the title's own files");
    await shot("5-removed");
  }
} catch (e) {
  console.log(`[repatch] FAIL ${e.stack || e}`);
  failures += 1;
  await shot("error").catch(() => {});
} finally {
  await ctx.close();
  server.close();
}
console.log(failures ? `[repatch] ${failures} FAILURE(S)` : "[repatch] all checks passed");
process.exit(failures ? 1 : 0);
