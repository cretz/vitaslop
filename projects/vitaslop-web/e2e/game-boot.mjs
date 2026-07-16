// Boot the REAL retail title in the browser through the JSPI preemptive
// scheduler, driven by Playwright. Serves the extracted container dir + the wasm
// bundle (cross-origin isolated for shared memory), lets the page fetch the files and
// call run_game, and reports the boot result.
//
// Env:  GAME_DIR   the extracted app dir (default: $VITASLOP_GAME_DIR)
//       MAX_FRAMES display flips to run to (default 3)
//       MAX_ROUNDS scheduler round cap (default 50_000_000)
//       HEADED=1   show the browser
import { chromium } from "playwright";
import { createServer } from "node:http";
import { readFile, readdir, stat } from "node:fs/promises";
import { join, extname, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = join(fileURLToPath(import.meta.url), "..");
const webDir = join(here, "..", "web");
const gameDir = process.env.GAME_DIR || process.env.VITASLOP_GAME_DIR || "";
// Live run: render frame-by-frame through the real GXM->WebGPU renderer, driven by a
// scripted recipe that navigates the touch front-end to the Tutorial level. Run past
// the frame the level appears (~175) so the screenshot lands on live gameplay.
const maxFrames = Number(process.env.MAX_FRAMES || 260);
const maxRounds = Number(process.env.MAX_ROUNDS || 50_000_000);
// The display flip by which the tutorial level is on screen; the harness waits for the
// live loop to pass it before screenshotting.
const targetFrame = Number(process.env.TARGET_FRAME || 178);
// The scripted-input recipe (frame-keyed touch taps). Override with RECIPE=<path>, or
// RECIPE="" for a live-input session. No default: pass a recipe from the
// vitaslop-gamerun-recipes crate for the title under test.
const recipePath = process.env.RECIPE ?? "";
const recipe = recipePath ? await readFile(recipePath, "utf8").catch(() => "") : "";

const MIME = { ".html": "text/html", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json" };

// Recursively list files under `root`, returning forward-slash relative paths.
async function walk(root, dir = root, out = []) {
  for (const name of await readdir(dir)) {
    const full = join(dir, name);
    const s = await stat(full);
    if (s.isDirectory()) await walk(root, full, out);
    else out.push(relative(root, full).split(sep).join("/"));
  }
  return out;
}

async function main() {
  const manifest = await walk(gameDir);
  const totalMB = (
    (await Promise.all(manifest.map((p) => stat(join(gameDir, p))))).reduce((a, s) => a + s.size, 0) / 1e6
  ).toFixed(0);
  console.log(`[game] ${manifest.length} files, ${totalMB} MB in ${gameDir}`);

  const server = createServer(async (req, res) => {
    const coi = {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    };
    try {
      const url = decodeURIComponent(req.url.split("?")[0]);
      if (url === "/game-manifest.json") {
        res.writeHead(200, { "content-type": "application/json", ...coi });
        return res.end(JSON.stringify(manifest));
      }
      const file = url.startsWith("/game/")
        ? join(gameDir, url.slice("/game/".length))
        : join(webDir, url === "/" ? "/game.html" : url);
      const body = await readFile(file);
      res.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream", ...coi });
      res.end(body);
    } catch {
      res.writeHead(404).end("not found");
    }
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const url = `http://127.0.0.1:${server.address().port}/`;
  console.log(`[game] serving at ${url}`);

  // WORKER=1 runs the emulator in a Web Worker (the production home). A worker allows
  // synchronous instantiation of the title's large transpiled module at any size, so it
  // needs NO WebAssemblyUnlimitedSyncCompilation flag - we deliberately omit it in worker
  // mode to prove that. On the main thread that flag is still required (large sync
  // instantiate mid-run is disallowed there).
  const useWorker = !!process.env.WORKER;
  const features = useWorker ? "Vulkan" : "Vulkan,WebAssemblyUnlimitedSyncCompilation";
  const browser = await chromium.launch({
    channel: process.env.PWCHANNEL || "chrome",
    headless: !process.env.HEADED,
    args: [
      "--enable-unsafe-webgpu",
      `--enable-features=${features}`,
      "--enable-unsafe-swiftshader",
      "--use-angle=default",
    ],
  });
  const page = await browser.newPage();
  page.on("console", (m) => console.log(`[page:${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));

  let ok = false;
  try {
    await page.goto(url + (useWorker ? "game-worker.html" : ""), { waitUntil: "load", timeout: 30000 });
    console.log(`[game] mode: ${useWorker ? "web worker (no sync-compile flag)" : "main thread"}`);
    console.log(`[game] recipe: ${recipe ? `${recipe.split("\n").length} lines` : "(none - live input)"}`);
    // __boot resolves after the one-time decrypt/link/transpile + WebGPU setup, then
    // the live loop runs on the event loop.
    const msg = await page.evaluate(
      ([recipe, f, r]) => window.__boot(recipe, f, r),
      [recipe, maxFrames, maxRounds]
    );
    console.log("[game] SETUP:", msg);
    // Wait for the live loop to render past the tutorial-appears frame.
    await page.waitForFunction(
      (target) => {
        const m = (document.getElementById("status")?.textContent || "").match(/frame (\d+)/);
        return m && Number(m[1]) >= target;
      },
      targetFrame,
      { timeout: 180000 }
    );
    // Let a few more live frames present so the FPS meter settles on a real cadence.
    await page.waitForFunction(() => /fps:\s*\d/.test(document.getElementById("fps")?.textContent || ""), { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(1500);
    const fps = await page.locator("#fps").textContent().catch(() => "?");
    const perf = await page.locator("#perf").textContent().catch(() => "?");
    const status = await page.locator("#status").textContent().catch(() => "?");
    const shotDir = process.env.SHOT_DIR || join(here, "screenshots");
    await import("node:fs/promises").then((fs) => fs.mkdir(shotDir, { recursive: true }));
    await page.locator("#screen").screenshot({ path: join(shotDir, "game.png") });
    console.log(`[game] live render ${fps} | ${perf} | ${status} -> screenshot ${join(shotDir, "game.png")}`);
    ok = /frame (\d+)/.test(status) && Number(status.match(/frame (\d+)/)[1]) >= targetFrame;
  } catch (e) {
    console.error("[game] error:", e.message);
    console.error("[game] last status:", await page.locator("#status").textContent().catch(() => "?"));
  } finally {
    await browser.close();
    server.close();
  }
  console.log(ok ? "[game] PASS" : "[game] FAIL");
  process.exit(ok ? 0 : 1);
}

main();
