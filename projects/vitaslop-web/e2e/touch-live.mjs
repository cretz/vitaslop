// touch-live.mjs - a person's mouse on the player's canvas reaches the guest's touch panels, with
// the opt-in "Rear touch panel" setting OFF and then ON.
//
// Runs the authored touchlive program (vitaslop-conformance-suite-vita/touchlive-src) through the
// real import screen and Play, presses the canvas with the mouse, and reads what the program
// printed (`F`/`B` + a 4x4 panel cell per press, `f`/`b` per lift, `E` at the end) out of the
// end-of-run diagnostics, as homebrew.mjs does.
//
//   OFF: every button is the FRONT panel, exactly as before the setting existed - the right
//        click lands on the front too.
//   ON:  the right button is the REAR panel; the left stays on the front.
//   ON, by TOUCH (a phone): fingers are the front panel until the on-screen REAR toggle is lit,
//        then the rear, and the front again once it is put out.
//   OFF, turned ON from the in-game MENU (by touch): the title's own settings say off, so no REAR
//        toggle; the menu's "Rear touch panel" adds it mid-run and the presses read as the ON arm.
//        The change is kept in the TITLE's settings (which override the global record, so a
//        global save would not stick) and the global record is left alone.
//
// Env: HEADLESS=1, ALLOW_SOFTWARE=1 (a software WebGPU adapter is acceptable, as on CI; with
//      FORCE_SWIFTSHADER=1 it is forced), BOOT_MS (default 8000: the wait for the program to start sampling before the
//      first press), WAIT_MS (default 60000), SHOT_DIR
// Run: HEADLESS=1 node touch-live.mjs

import { mkdtemp, mkdir, copyFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";
import { startServer, webDir, here, sfo, requireBundle } from "./harness.mjs";

const velf = join(here, "..", "..", "vitaslop-conformance-suite-vita", "touchlive-src", "touchlive.velf");
const bootMs = Number(process.env.BOOT_MS || 8000);
const waitMs = Number(process.env.WAIT_MS || 60000);
const shotDir = process.env.SHOT_DIR || join(here, "screenshots", "touch-live");

// The presses, in the guest's 960x544 screen space, and what each arm must read back.
// (600,340) is front cell 2,2; (480,200) is front cell 2,1 and rear cell 2,1 (the rear panel's
// rows are 0..889); (120,68) is front cell 0,0, which ends the program. All three are clear of
// the on-screen controls the touch arm puts over the game.
const PRESSES = [
  { x: 600, y: 340, button: "left" },
  { x: 480, y: 200, button: "right" },
  { x: 120, y: 68, button: "left" },
];
const ARMS = [
  { rearTouch: false, expect: "F22fF21fF00E" },
  { rearTouch: true, expect: "F22fB21bF00E" },
  { rearTouch: true, touch: true, expect: "F22fB21bF00E" },
  { rearTouch: false, touch: true, viaMenu: true, expect: "F22fB21bF00E" },
];
const TITLE_ID = "VSLP00004";

requireBundle();
const work = await mkdtemp(join(tmpdir(), "vitaslop-touchlive-"));
const app = join(work, "app");
await mkdir(join(app, "sce_sys"), { recursive: true });
await copyFile(velf, join(app, "eboot.bin"));
await writeFile(join(app, "sce_sys", "param.sfo"), sfo({ TITLE_ID, TITLE: "vitaslop touch test" }));
await mkdir(shotDir, { recursive: true });

const server = await startServer(webDir);
const software = !!process.env.ALLOW_SOFTWARE;
const url = `http://127.0.0.1:${server.address().port}/` + (software ? "?knobs=VITASLOP_ALLOW_SOFTWARE_GPU=1" : "");
let failures = 0;
// One browser per kind of input. Touch emulation (`hasTouch`) is the touch arm's alone: under
// it Chrome delivered no right-button press at all, which would make the mouse arms test the
// emulator of a touch screen rather than a mouse.
for (const touch of [false, true]) {
  const arms = ARMS.filter((a) => !!a.touch === touch);
  const context = await chromium.launchPersistentContext(join(work, touch ? "profile-touch" : "profile"), {
    channel: process.env.PWCHANNEL || "chrome",
    headless: !!process.env.HEADLESS,
    viewport: { width: 1100, height: 800 },
    hasTouch: touch,
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

  /// The page coordinates of guest screen point (x, y): the canvas's box, less the letterbox that
  /// `object-fit: contain` leaves - the same arithmetic worker-input.js inverts.
  const toPage = (x, y) =>
    page.$eval(
      "#screen",
      (c, [x, y]) => {
        const r = c.getBoundingClientRect();
        const k = Math.min(r.width / 960, r.height / 544);
        return { px: r.left + (r.width - 960 * k) / 2 + x * k, py: r.top + (r.height - 544 * k) / 2 + y * k };
      },
      [x, y]
    );

  /// A finger down for 200 ms at page point (x, y) - a real touch, `pointerType: "touch"`.
  /// Playwright's `tap` lifts in the same task, before the guest can sample it.
  const cdp = await context.newCDPSession(page);
  const touchAt = async (x, y) => {
    await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: [{ x, y }] });
    await page.waitForTimeout(200);
    await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    await page.waitForTimeout(300);
  };
  const toggleRear = async () => {
    const b = await page.$eval("[data-toggle=rear]", (e) => {
      const r = e.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    });
    await touchAt(b.x, b.y);
  };

  try {
    await page.goto(url, { waitUntil: "load" });
    await page.waitForSelector("#grid, .card.error", { state: "attached", timeout: 30000 });
    await page.goto(url + "#/import");
    await page.waitForSelector("#f-dir", { state: "attached" });
    await page.setInputFiles("#f-dir", app);
    await page.waitForFunction(() => location.hash.startsWith("#/title/") || document.querySelector(".card.error"), null, { timeout: 120000 });
    if (await page.$(".card.error")) throw new Error("import: " + (await page.$eval(".card.error", (e) => e.innerText)));
    const titlePage = page.url();

    for (const arm of arms) {
      const tag = `${arm.rearTouch ? "on" : "off"}${arm.touch ? "-touch" : ""}${arm.viaMenu ? "-menu" : ""}`;
      const name = `rear ${arm.rearTouch ? "ON" : "OFF"}${arm.touch ? " by touch" : ""}${arm.viaMenu ? ", turned on from the menu" : ""}`;
      // The touch arm puts the on-screen controls over the game (a desktop hides them by default).
      // The menu arm keeps rear touch in the TITLE's patch, the place the menu must save it back to.
      const stored = arm.viaMenu ? { pad: { mode: "overlay" } } : arm.touch ? { rearTouch: arm.rearTouch, pad: { mode: "overlay" } } : { rearTouch: arm.rearTouch };
      const titlePatch = arm.viaMenu ? { rearTouch: arm.rearTouch } : null;
      await page.evaluate(
        ([s, t, key]) => {
          localStorage.setItem("vitaslop.settings", JSON.stringify(s));
          if (t) localStorage.setItem(key, JSON.stringify(t));
          else localStorage.removeItem(key);
        },
        [stored, titlePatch, `vitaslop.title.${TITLE_ID}`]
      );
      await page.goto("about:blank");
      await page.goto(titlePage);
      await page.waitForSelector("#play", { timeout: 30000 });
      await page.click("#play");
      await page.waitForFunction(() => !document.getElementById("player").hidden, null, { timeout: 10000 });
      await page.waitForTimeout(bootMs);
      let menuOk = true;
      if (arm.viaMenu) {
        const rearShown = () => page.$eval("[data-toggle=rear]", (e) => !e.hidden && e.offsetParent !== null);
        const before = await rearShown();
        await page.click("#menubtn");
        await page.waitForSelector("#m-rear", { state: "visible" });
        const unticked = !(await page.$eval("#m-rear", (e) => e.checked));
        await page.click("#m-rear");
        await page.click("#m-resume");
        await page.waitForTimeout(500);
        const after = await rearShown();
        menuOk = !before && unticked && after;
        console.log(`[touch-live] menu: REAR before ${before}, box unticked ${unticked}, REAR after ${after} - ${menuOk ? "ok" : "WRONG"}`);
      }
      for (const p of PRESSES) {
        const { px, py } = await toPage(p.x, p.y);
        if (arm.touch) {
          // The press the mouse makes with its right button, a finger makes with REAR lit.
          if (p.button === "right") await toggleRear();
          await touchAt(px, py);
          if (p.button === "right") await toggleRear();
          continue;
        }
        await page.mouse.move(px, py);
        await page.mouse.down({ button: p.button });
        await page.waitForTimeout(200);
        await page.mouse.up({ button: p.button });
        await page.waitForTimeout(300);
      }
      await page.click("#menubtn").catch(() => {});
      const t0 = Date.now();
      let diag = "";
      while (Date.now() - t0 < waitMs) {
        await page.waitForTimeout(1000);
        diag = await page.$eval("#m-diag", (e) => e.textContent).catch(() => "");
        if (/guest console \(/.test(diag) || /\bENDED\b/.test(diag)) break;
        await page.click("#menubtn").catch(() => {});
      }
      await writeFile(join(shotDir, `diag-rear-${tag}.txt`), diag);
      const consoleText = /guest console \([^)]*\):\n([\s\S]*?)\n\n/.exec(diag + "\n\n")?.[1];
      let ok = consoleText === arm.expect && menuOk;
      if (arm.viaMenu) {
        const [t, g] = await page.evaluate((key) => [localStorage.getItem(key), localStorage.getItem("vitaslop.settings")], `vitaslop.title.${TITLE_ID}`);
        const kept = JSON.parse(t || "{}").rearTouch === true && !("rearTouch" in JSON.parse(g || "{}"));
        console.log(`[touch-live] menu change kept: title ${t}, global ${g} - ${kept ? "ok" : "WRONG"}`);
        ok = ok && kept;
      }
      if (!ok) failures++;
      console.log(`[touch-live] ${name}: read ${JSON.stringify(consoleText)}, expected ${JSON.stringify(arm.expect)} - ${ok ? "ok" : "WRONG"}`);
    }
  } catch (e) {
    failures++;
    console.error("FAIL:", e.message);
    await page.screenshot({ path: join(shotDir, `fail${touch ? "-touch" : ""}.png`) }).catch(() => {});
  } finally {
    await writeFile(join(shotDir, `console${touch ? "-touch" : ""}.txt`), logs.join("\n")).catch(() => {});
    await context.close();
  }
}
server.close();
await rm(work, { recursive: true, force: true }).catch(() => {});
console.log(failures ? "[touch-live] FAIL" : "[touch-live] PASS");
process.exit(failures ? 1 : 0);
