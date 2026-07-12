// Browser test: run the entire ARM conformance corpus on Chrome's WebAssembly
// engine (through WebVm) and assert every case passes - proving the transpiler's
// output behaves identically on the browser engine and on native wasmtime.
//
// Run:  node conformance.mjs   (also part of `npm test`)
// Env:  HEADED=1, PWCHANNEL=... (see harness.mjs)

import { webDir, requireBundle, startServer, launchChrome } from "./harness.mjs";

async function main() {
  requireBundle();

  const server = await startServer(webDir);
  const { port } = server.address();
  const url = `http://127.0.0.1:${port}/conformance.html`;
  console.log(`[conformance] ${url}`);

  const browser = await launchChrome();
  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));

  let ok = false;
  let data = null;
  try {
    await page.goto(url, { waitUntil: "load", timeout: 30000 });
    // The corpus runs in one synchronous wasm call; wait for the global it sets.
    await page.waitForFunction(() => window.__CONFORMANCE__ !== undefined, { timeout: 120000 });
    data = await page.evaluate(() => window.__CONFORMANCE__);

    if (data.error) {
      console.error("[conformance] runner error:", data.error);
    } else {
      console.log(`[conformance] ${data.passed} / ${data.total} passed`);
      for (const c of data.cases) {
        console.log(`  ${c.pass ? "PASS" : "FAIL"}  ${c.name}${c.detail ? "  -> " + c.detail : ""}`);
      }
      ok = data.total > 0 && data.passed === data.total;
    }
  } catch (e) {
    console.error("[conformance] harness error:", e.message);
  } finally {
    if (!ok) {
      console.log("--- browser console ---");
      for (const l of logs) console.log(l);
    }
    await browser.close();
    server.close();
  }

  console.log(ok ? "[conformance] PASS\n" : "[conformance] FAIL\n");
  process.exit(ok ? 0 : 1);
}

main();
