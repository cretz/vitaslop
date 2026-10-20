// A page that loads during a deploy can pair one build's glue JS with another build's wasm, and
// instantiating then fails with a LinkError. The page must refetch the bundles past the HTTP cache
// and reload ONCE (store.js `recoverFromMixedBuild`), ending up on a working library screen.
//
// Simulated here: the FIRST response for pkg/vitaslop_web.js has one import renamed, so the wasm
// asks for a function the glue no longer provides - the exact error a mixed load gives. Every
// later response is the real file. A second case serves the broken glue EVERY time and checks
// that the page tries one reload and then stops, instead of reloading forever.
import { requireBundle, startServer, webDir, launchChrome } from "./harness.mjs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

requireBundle();
const server = await startServer(webDir);
const url = `http://127.0.0.1:${server.address().port}/`;
const glue = await readFile(join(webDir, "pkg", "vitaslop_web.js"), "utf8");
const name = /__wbg_call_[0-9a-f]+/.exec(glue);
if (!name) throw new Error("no __wbg_call_ import in the glue to break");
const broken = glue.split(name[0]).join(name[0] + "_stale");

let failures = 0;
const check = (ok, what) => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${what}`);
  if (!ok) failures += 1;
};

async function run(breakEvery) {
  const browser = await launchChrome();
  const page = await (await browser.newContext()).newPage();
  let served = 0;
  let loads = 0;
  page.on("load", () => (loads += 1));
  await page.route("**/pkg/vitaslop_web.js", async (route) => {
    served += 1;
    if (breakEvery || served === 1) {
      await route.fulfill({ status: 200, contentType: "text/javascript", body: broken });
    } else {
      await route.continue();
    }
  });
  await page.goto(url);
  await page.waitForTimeout(6000);
  const text = await page.locator("#view").innerText().catch(() => "");
  await browser.close();
  return { loads, served, error: /Something went wrong/.test(text), text: text.slice(0, 160).replace(/\s+/g, " ") };
}

const once = await run(false);
check(once.loads === 2 && !once.error, `a mixed load reloads once and recovers (loads ${once.loads}, glue served ${once.served}, ${once.error ? "error shown" : "no error"})`);
const always = await run(true);
check(always.loads === 2, `a load that stays broken tries one reload and does not loop (loads ${always.loads} in 6 s)`);
server.close();
console.log(failures ? `\nmixed-build: ${failures} FAILED` : "\nmixed-build: ALL PASS");
process.exit(failures ? 1 : 0);
