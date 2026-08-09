// Price a host-call crossing in a real browser: run `web/jspi-cost.html` and print its
// table. See that page's header for what each arm isolates and why the page exists.
//
// This is the DESKTOP arm, and a desktop crossing is cheap enough to mis-rank anything
// (`vitaslop-rank-host-calls-by-phone-count`) - so treat what this prints as a shape, and
// get the real numbers by opening the same page on the phone. Same code, both machines.
//
//   node e2e/jspi-cost.mjs
import { launchChrome, startServer, webDir } from "./harness.mjs";

const server = await startServer(webDir);
const port = server.address().port;
const browser = await launchChrome();
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("pageerror:", e.message));

await page.goto(`http://127.0.0.1:${port}/jspi-cost.html`);
console.log(await page.textContent("#env"));

await page.click("#go");
// The whole set is a few seconds on a desktop and rather more on a phone; the page
// publishes `window.__jspiCost` when it is done, so wait on that rather than on a sleep.
await page.waitForFunction(() => window.__jspiCost !== undefined, null, { timeout: 300_000 });
const { results, text } = await page.evaluate(() => window.__jspiCost);

console.log(text);

await browser.close();
server.close();

// A failed arm is a failed run: the page silently degrading to four arms would leave the
// two differences this exists to compute reading as "-", which looks like a small number.
const failed = results.filter((r) => r.error);
if (failed.length) {
  console.error(`FAIL: ${failed.length} arm(s) errored`);
  process.exit(1);
}
