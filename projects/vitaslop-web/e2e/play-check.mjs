// Validate the USER-FACING browser play path end to end: load play.html (which auto-
// boots the emulator in a Web Worker with NO Chrome flags and live input), then drive
// it to the Tutorial level with REAL POINTER TAPS (not the scripted Rust recipe) - the
// exact thing a human does. Proves the whole chain: worker run + OffscreenCanvas render
// + page->worker input forwarding + touch reaching the guest. Screenshots the result.
//
// This is the interactive twin of game-boot.mjs (which uses the in-Rust recipe). It
// deliberately launches Chrome WITHOUT WebAssemblyUnlimitedSyncCompilation.
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, readdir, stat, mkdir } from "node:fs/promises";
import { join, extname, relative, sep, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = join(fileURLToPath(import.meta.url), "..");
const webDir = join(here, "..", "web");
const gameDir =
  process.env.GAME_DIR ||
  "C:/work/personal/vitaslop-work/working-area/games/olliolli/extracted/app/PCSE00341";
const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".css": "text/css" };

// The Tutorial tap sequence in SCREEN coords (960x544) = the recipe's panel coords / 2,
// each paired with the display frame to fire it at (the page reports "frame N" in
// #status, so taps sync to frames, not wall time).
const taps = [
  { frame: 12, x: 225, y: 337 },  // Offline Mode (dismiss connect dialog)
  { frame: 30, x: 115, y: 115 },  // main menu
  { frame: 80, x: 465, y: 337 },  // Play
  { frame: 112, x: 810, y: 188 }, // Tutorial / OK
  { frame: 128, x: 315, y: 435 }, // Pushing -> load level
];
const targetFrame = Number(process.env.TARGET_FRAME || 178);

async function walk(root, dir = root, out = []) {
  for (const name of await readdir(dir)) {
    const full = join(dir, name);
    const s = await stat(full);
    if (s.isDirectory()) await walk(root, full, out);
    else out.push(relative(root, full).split(sep).join("/"));
  }
  return out;
}

async function currentFrame(page) {
  const t = (await page.locator("#status").textContent().catch(() => "")) || "";
  const m = t.match(/frame (\d+)/);
  return m ? Number(m[1]) : -1;
}

async function main() {
  const manifest = await walk(gameDir);
  const server = createServer(async (req, res) => {
    const coi = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
    try {
      const url = decodeURIComponent(req.url.split("?")[0]);
      if (url === "/game-manifest.json") {
        res.writeHead(200, { "content-type": "application/json", ...coi });
        return res.end(JSON.stringify(manifest));
      }
      const file = url.startsWith("/game/") ? join(gameDir, url.slice("/game/".length)) : join(webDir, url === "/" ? "/play.html" : url);
      const body = await readFile(file);
      res.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream", ...coi });
      res.end(body);
    } catch {
      res.writeHead(404).end("not found");
    }
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const url = `http://127.0.0.1:${server.address().port}/`;
  console.log(`[play] serving at ${url}`);

  const browser = await chromium.launch({
    channel: process.env.PWCHANNEL || "chrome",
    headless: !process.env.HEADED,
    // NOTE: no WebAssemblyUnlimitedSyncCompilation - the worker path does not need it.
    args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--enable-unsafe-swiftshader", "--use-angle=default"],
  });
  const page = await browser.newPage();
  page.on("console", (m) => console.log(`[page:${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));

  let ok = false;
  try {
    await page.goto(url, { waitUntil: "load", timeout: 30000 });
    // Wait for the live loop to start reporting frames (boot done).
    await page.waitForFunction(() => /frame \d/.test(document.getElementById("status")?.textContent || ""), null, { timeout: 180000 });
    const box = await page.locator("#screen").boundingBox();
    const sx = box.width / 960, sy = box.height / 544;

    // Fire each tap when the live frame counter reaches its frame: move, press, hold a
    // few frames, release - the same press-then-release a real finger makes.
    for (const tap of taps) {
      await page.waitForFunction((f) => {
        const m = (document.getElementById("status")?.textContent || "").match(/frame (\d+)/);
        return m && Number(m[1]) >= f;
      }, tap.frame, { timeout: 60000 });
      const vx = box.x + tap.x * sx, vy = box.y + tap.y * sy;
      await page.mouse.move(vx, vy);
      await page.mouse.down();
      await page.waitForTimeout(150); // hold ~9 frames
      await page.mouse.up();
      console.log(`[play] tapped (${tap.x},${tap.y}) at frame >=${tap.frame}, now frame ${await currentFrame(page)}`);
    }

    await page.waitForFunction((f) => {
      const m = (document.getElementById("status")?.textContent || "").match(/frame (\d+)/);
      return m && Number(m[1]) >= f;
    }, targetFrame, { timeout: 60000 });
    await page.waitForTimeout(800);

    const fps = await page.locator("#fps").textContent().catch(() => "?");
    const perf = await page.locator("#perf").textContent().catch(() => "?");
    const shotDir = process.env.SHOT_DIR || join(here, "screenshots");
    await mkdir(shotDir, { recursive: true });
    await page.locator("#screen").screenshot({ path: join(shotDir, "play.png") });
    const frame = await currentFrame(page);
    console.log(`[play] ${fps} | ${perf} | frame ${frame} -> screenshot ${join(shotDir, "play.png")}`);
    ok = frame >= targetFrame;
  } catch (e) {
    console.error("[play] error:", e.message);
    console.error("[play] last status:", await page.locator("#status").textContent().catch(() => "?"));
  } finally {
    await browser.close();
    server.close();
  }
  console.log(ok ? "[play] PASS" : "[play] FAIL");
  process.exit(ok ? 0 : 1);
}

main();
