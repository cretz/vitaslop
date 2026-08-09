// Load the PRODUCT page the way a phone does - through serve.mjs, not through the e2e
// harness - and print every console message, page error and worker error with its stack.
//
// The harness loads `game-worker.html`. The device loads `live.html`. Testing only the
// former is how a break in the latter reaches a user first.
import { chromium } from "playwright";

const url = process.env.URL || "https://localhost:8443/";
const browser = await chromium.launch({
  // The installed Chrome, the same one `game-boot.mjs` drives - the bundled headless shell
  // is not present here and is not what a device runs anyway.
  channel: process.env.PWCHANNEL || "chrome",
  headless: !process.env.HEADED,
  args: [
    "--ignore-certificate-errors",
    "--enable-unsafe-webgpu",
    "--autoplay-policy=no-user-gesture-required",
  ],
});
const ctx = await browser.newContext({ ignoreHTTPSErrors: true });
const page = await ctx.newPage();

page.on("console", (m) => console.log(`[${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}\n${e.stack || ""}`));
page.on("worker", (w) => {
  console.log(`[worker started] ${w.url()}`);
});
page.on("requestfailed", (r) => console.log(`[requestfailed] ${r.url()} ${r.failure()?.errorText}`));

await page.goto(url, { waitUntil: "load", timeout: 60000 });
console.log("--- loaded, title:", await page.title());

// Click through to the title the device would pick, if the page offers a picker.
// The page lists titles as cards, each with its own recipe <select> and PLAY button, keyed
// by TITLE_ID. Drive it the way a person does: choose the recipe, then press PLAY.
const titleId = process.env.TITLE_ID || "PCSA00015";
const recipe = process.env.RECIPE_NAME || "";
await page.waitForSelector(`#p-${titleId}`, { timeout: 30000 });
if (recipe) {
  const opts = await page.locator(`#r-${titleId} option`).allTextContents();
  console.log(`--- recipes offered: ${JSON.stringify(opts)}`);
  const match = opts.find((o) => o.toLowerCase().includes(recipe.toLowerCase()));
  if (match) {
    await page.selectOption(`#r-${titleId}`, { label: match });
    console.log(`--- selected recipe: ${match}`);
  } else {
    console.log(`--- NO recipe matching ${recipe}; leaving the default`);
  }
}
console.log(`--- pressing PLAY for ${titleId}`);
await page.click(`#p-${titleId}`);

// Report the page's own status line as it changes - it is where the worker error lands.
let lastStatus = "";
const statusTimer = setInterval(async () => {
  const s = await page.textContent("#status").catch(() => null);
  if (s && s !== lastStatus) {
    lastStatus = s;
    console.log(`[status] ${s}`);
  }
}, 1000);

await page.waitForTimeout(Number(process.env.WAIT_MS || 45000));
clearInterval(statusTimer);
const shot = process.env.SHOT;
if (shot) {
  await page.locator("#screen").screenshot({ path: shot });
  console.log(`--- wrote ${shot}`);
}
console.log("--- diagnostics panel ---");
console.log((await page.textContent("#diag").catch(() => "")) || "(empty)");
console.log("--- done");
await browser.close();
