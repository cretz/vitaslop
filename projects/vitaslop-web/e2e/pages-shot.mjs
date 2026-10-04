// Screenshot the site's own pages - the library, a title page, the settings, the import and the
// about page - for a side-by-side look against the desktop shell's (`VITASLOP_SHELL_SHOT`).
//
// Env:  PROFILE_DIR  persistent Chrome profile (its OPFS holds the library; required).
//       SHOT_DIR     where the screenshots land (required).
//       TITLE_ID     also shoot `#/title/<id>` and `#/settings/<id>`.
//       PORT         fixed port - the origin is the storage key (default 8765, as ui.mjs).
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";
import { startServer, webDir, requireBundle } from "./harness.mjs";

const profileDir = process.env.PROFILE_DIR;
const shotDir = process.env.SHOT_DIR;
if (!profileDir || !shotDir) throw new Error("PROFILE_DIR and SHOT_DIR are required");
requireBundle();
await mkdir(shotDir, { recursive: true });
const server = await startServer(webDir, Number(process.env.PORT || 8765));
const url = `http://127.0.0.1:${server.address().port}/`;
const context = await chromium.launchPersistentContext(profileDir, {
  channel: process.env.PWCHANNEL || "chrome",
  headless: true,
  viewport: { width: 1100, height: 800 },
  args: ["--enable-unsafe-webgpu", "--enable-gpu"],
});
const page = await context.newPage();
const pages = [["library", "#/"], ["settings", "#/settings"], ["import", "#/import"], ["about", "#/about"]];
const id = process.env.TITLE_ID;
if (id) pages.push(["title", `#/title/${id}`], ["title-settings", `#/settings/${id}`]);
try {
  for (const [name, hash] of pages) {
    await page.goto(url + hash);
    await page.waitForTimeout(1500);
    await page.screenshot({ path: join(shotDir, `${name}.png`) });
    console.log(`${name} -> ${join(shotDir, name + ".png")}`);
  }
} finally {
  await context.close();
  server.close();
}
