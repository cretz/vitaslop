// Browser test: run the cube in Chrome via WebGPU, screenshot the canvas, and
// assert a cube was actually drawn.
//
// Run:  npm test                    (runs this + conformance.mjs)
//   or: node run.mjs
// Env:  SHOT_DIR=<dir>  where to write screenshots (default ./screenshots)
//       HEADED=1        show the browser window
//       PWCHANNEL=...   browser channel (default chrome)

import { PNG } from "pngjs";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { here, webDir, requireBundle, startServer, launchChrome } from "./harness.mjs";

const shotDir = process.env.SHOT_DIR || join(here, "screenshots");

// Count pixels in a screenshot PNG clearly brighter than the dark clear color -
// i.e. the rendered cube. A WebGPU canvas is not readable via in-page 2D
// drawImage/getImageData (the drawing buffer is not preserved), so we analyze
// Playwright's composited screenshot instead.
function cubeCoverage(pngBuffer) {
  const img = PNG.sync.read(pngBuffer);
  let bright = 0;
  const total = img.width * img.height;
  for (let i = 0; i < img.data.length; i += 4) {
    if (Math.max(img.data[i], img.data[i + 1], img.data[i + 2]) > 50) bright++;
  }
  return { bright, total, frac: bright / total, w: img.width, h: img.height };
}

async function main() {
  requireBundle();
  mkdirSync(shotDir, { recursive: true });

  const server = await startServer(webDir);
  const { port } = server.address();
  const url = `http://127.0.0.1:${port}/`;
  console.log(`[cube] serving ${webDir} at ${url}`);

  const browser = await launchChrome();
  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));

  let ok = false;
  try {
    await page.goto(url, { waitUntil: "load", timeout: 30000 });
    await page.waitForFunction(
      () => /cube ran|Error|no WebGPU/i.test(document.getElementById("status")?.textContent || ""),
      { timeout: 120000 }
    );
    const status = await page.locator("#status").textContent();
    // Let the render loop run long enough for the live FPS meter (500ms window) to
    // publish a couple of readings before we sample it.
    await page.waitForFunction(
      () => /fps:\s*\d/.test(document.getElementById("fps")?.textContent || ""),
      { timeout: 10000 }
    );
    await page.waitForTimeout(1500);
    const fps = await page.locator("#fps").textContent();
    const buf = await page.locator("#screen").screenshot({ path: join(shotDir, "cube.png") });
    const cov = cubeCoverage(buf);

    console.log("[cube] status:", status);
    console.log("[cube] live", fps);
    console.log("[cube] coverage:", JSON.stringify(cov));
    ok = /cube ran/i.test(status) && cov.frac > 0.05 && cov.frac < 0.9;
  } catch (e) {
    console.error("[cube] harness error:", e.message);
  } finally {
    if (!ok) {
      console.log("--- browser console ---");
      for (const l of logs) console.log(l);
    }
    await browser.close();
    server.close();
  }

  console.log(ok ? "[cube] PASS\n" : "[cube] FAIL\n");
  process.exit(ok ? 0 : 1);
}

main();
