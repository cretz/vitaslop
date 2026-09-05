// Drive the PRODUCT front end end to end on the real GPU: open the library, import a
// title from the files a person would have (a pkg + work.bin, or a dump folder),
// open its page, play it, and screenshot the running game.
//
// Env:  GAME_SRC     a directory holding the container (pkg + work.bin, a PFS dump, or a
//                    vitaslop dump tree). Every file under it is picked, as a folder pick.
//       PROFILE_DIR  persistent Chrome profile (required; the import lives in its OPFS).
//       SHOT_DIR     where screenshots land (default: e2e/screenshots/ui).
//       PLAY_MS      how long to run the game before the screenshot (default 20000).
//       HEADLESS=1   no window (still the real GPU).
//       SKIP_IMPORT=1 the title is already in this profile's library; go straight to it.
import { chromium } from "playwright";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { startServer, webDir, requireBundle } from "./harness.mjs";

const src = process.env.GAME_SRC;
const profileDir = process.env.PROFILE_DIR;
const shotDir = process.env.SHOT_DIR || join(webDir, "..", "e2e", "screenshots", "ui");
const playMs = Number(process.env.PLAY_MS || 20000);
if (!profileDir) throw new Error("PROFILE_DIR is required");
requireBundle();
await mkdir(shotDir, { recursive: true });

// A FIXED port: the origin is the storage key, and a random port is a fresh, empty
// library every run.
const server = await startServer(webDir, Number(process.env.PORT || 8765));
const url = `http://127.0.0.1:${server.address().port}/`;
const context = await chromium.launchPersistentContext(profileDir, {
  channel: process.env.PWCHANNEL || "chrome",
  headless: !!process.env.HEADLESS,
  viewport: { width: 1100, height: 800 },
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--enable-gpu", "--use-angle=default", "--autoplay-policy=no-user-gesture-required"],
});
const page = await context.newPage();
const logs = [];
page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));
const fail = async (why) => {
  await page.screenshot({ path: join(shotDir, "fail.png") }).catch(() => {});
  await writeFile(join(shotDir, "console.txt"), logs.join("\n"));
  console.error("FAIL:", why);
  await context.close();
  server.close();
  process.exit(1);
};

try {
  await page.goto(url, { waitUntil: "load" });
  await page.waitForSelector("#grid, .card.error", { state: "attached", timeout: 30000 });
  await page.screenshot({ path: join(shotDir, "01-library.png") });
  const blocked = await page.$(".card.error");
  if (blocked) await fail("the browser checks block play: " + (await blocked.innerText()));
  console.log("library rendered; checks pass");

  let titleId;
  if (!process.env.SKIP_IMPORT) {
    if (!src) throw new Error("GAME_SRC is required unless SKIP_IMPORT=1");
    await page.goto(url + "#/import");
    await page.waitForSelector("#f-dir", { state: "attached" });
    await page.setInputFiles("#f-dir", src);
    await page.waitForSelector("#do-import, .card.error", { timeout: 120000 });
    if (await page.$(".card.error")) await fail("probe: " + (await page.$eval(".card.error", (e) => e.innerText)));
    await page.screenshot({ path: join(shotDir, "02-confirm.png") });
    const confirm = await page.$eval(".confirm", (e) => e.innerText);
    console.log("probe:", confirm.split("\n").slice(0, 2).join(" | "));
    const t0 = Date.now();
    await page.click("#do-import");
    await page.waitForFunction(() => location.hash.startsWith("#/title/") || document.querySelector(".card.error"), null, { timeout: 15 * 60 * 1000 });
    if (await page.$(".card.error")) await fail("import: " + (await page.$eval(".card.error", (e) => e.innerText)));
    console.log(`imported in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
    titleId = (await page.evaluate(() => location.hash)).split("/").pop();
  } else {
    await page.waitForSelector(".tile");
    titleId = await page.$eval(".tile", (e) => e.dataset.id);
    await page.goto(url + `#/title/${titleId}`);
  }
  await page.waitForSelector("#play", { timeout: 30000 });
  await page.waitForFunction(() => document.querySelector("#gd-info").textContent !== "checking...");
  await page.screenshot({ path: join(shotDir, "03-title.png") });
  console.log("title page:", titleId);

  await page.goto(url + "#/settings");
  await page.waitForSelector("#sf");
  await page.screenshot({ path: join(shotDir, "04-settings.png"), fullPage: true });

  await page.goto(url + `#/title/${titleId}`);
  await page.waitForSelector("#play");
  await page.click("#play");
  await page.waitForFunction(() => !document.getElementById("player").hidden, null, { timeout: 10000 });
  // The loading screen goes away on the first frame-rate report.
  await page.waitForFunction(() => document.getElementById("loading").hidden || !document.getElementById("fatal").hidden, null, { timeout: 5 * 60 * 1000 });
  if (!(await page.evaluate(() => document.getElementById("fatal").hidden))) {
    await fail("fatal: " + (await page.$eval("#fatal-text", (e) => e.innerText)));
  }
  console.log("running; playing for", playMs, "ms");
  await page.waitForTimeout(playMs);
  await page.screenshot({ path: join(shotDir, "05-play.png") });
  await page.click("#menubtn");
  await page.waitForFunction(() => !document.getElementById("menu").hidden);
  await page.click("#menu details:last-of-type summary");
  await page.screenshot({ path: join(shotDir, "06-menu.png") });
  const diag = await page.$eval("#m-diag", (e) => e.innerText);
  await writeFile(join(shotDir, "diag.txt"), diag);
  console.log(diag.split("\n").filter((l) => /^(fps|status|adapter)/.test(l)).join("\n"));
  await page.click("#m-quit");
  await page.waitForFunction(() => document.getElementById("player").hidden);
  await page.waitForSelector("#play");
  console.log("quit back to the title page");
  await writeFile(join(shotDir, "console.txt"), logs.join("\n"));
  console.log("PASS");
} catch (e) {
  await fail(e && e.stack ? e.stack : String(e));
}
await context.close();
server.close();
