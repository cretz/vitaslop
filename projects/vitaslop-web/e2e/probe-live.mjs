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

// Every line carries milliseconds since the probe started. Without it there is no way to
// price the fast-forward - the one part of a live run that does a FIXED amount of guest
// work in both arms of an A/B, and therefore the only part of a diverging race that can
// be compared like for like.
const t0 = Date.now();
const stamp = () => String(Date.now() - t0).padStart(7);
page.on("console", (m) => console.log(`${stamp()} [${m.type()}] ${m.text()}`));
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
// Extra knobs, appended to the card's own textarea exactly as a person would type them -
// `KNOBS="VITASLOP_LOG=info\nVITASLOP_DIRTY_PAGES=1"`. Driving the real control rather than
// reaching past it into the worker keeps this probe a test of the page that ships.
if (process.env.KNOBS) {
  // The textarea lives inside a collapsed <details>, so it has to be opened first - a
  // person taps "knobs" before typing in it.
  const box = page.locator(`#k-${titleId}`);
  await box.evaluate((el) => {
    const d = el.closest("details");
    if (d) d.open = true;
  });
  const existing = await box.inputValue();
  await box.fill(existing.replace(/\s*$/, "") + "\n" + process.env.KNOBS);
  console.log(`--- knobs appended: ${JSON.stringify(process.env.KNOBS)}`);
}
// DEBUG=1 ticks the card's "capture debug" box, which times every host call and counts them
// by NID so the panel can say where the guest CPU goes. It roughly DOUBLES the frame cost, so
// the frame times it reports are not real - the RATIOS are what it is for.
if (process.env.DEBUG) {
  await page.check(`#dbg-${titleId}`);
  console.log("--- debug capture ON (frame times inflated, ratios valid)");
}
console.log(`--- pressing PLAY for ${titleId}`);
await page.click(`#p-${titleId}`);

// Report the page's own status line as it changes - it is where the worker error lands.
let lastStatus = "";
const statusTimer = setInterval(async () => {
  const s = await page.textContent("#status").catch(() => null);
  if (s && s !== lastStatus) {
    lastStatus = s;
    console.log(`${stamp()} [status] ${s}`);
  }
}, 1000);

await page.waitForTimeout(Number(process.env.WAIT_MS || 45000));
clearInterval(statusTimer);
const shot = process.env.SHOT;
if (shot) {
  await page.locator("#screen").screenshot({ path: shot });
  console.log(`--- wrote ${shot}`);
}
// The fatal box first: a Rust panic, a dead worker or a failed boot lands there, and it is the
// one thing worth reading before the counters. On a phone this box is the whole crash report.
const fatalText = (await page.textContent("#fatal").catch(() => "")) || "";
console.log("--- fatal box ---");
console.log(fatalText || "(empty - nothing fatal reported)");
console.log("--- diagnostics panel ---");
console.log((await page.textContent("#diag").catch(() => "")) || "(empty)");
console.log("--- done");
await browser.close();
