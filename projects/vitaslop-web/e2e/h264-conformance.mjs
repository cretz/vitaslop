// Browser test: run the H.264 crate's own conformance suite against the browser's
// WebCodecs decoder, and fail if any check does.
//
// This exists because `cargo test` cannot reach that backend at all. Until it did, the
// WebCodecs path shipped having never executed, and its first three runs in a browser found
// three separate defects - a RefCell re-entered from a decoder callback, a `copyTo` that
// refuses an explicit format, and a frame layout that has to be read back rather than
// requested. Each was invisible on a desktop and cost a round trip through a person holding
// a phone.
//
// Run:  node h264-conformance.mjs   (also part of `npm test`)
// Env:  HEADED=1, PWCHANNEL=... (see harness.mjs)
import { webDir, requireBundle, startServer, launchChrome } from "./harness.mjs";

async function main() {
  requireBundle();

  const server = await startServer(webDir);
  const { port } = server.address();
  const url = `http://127.0.0.1:${port}/h264-conformance.html`;
  console.log(`[h264] ${url}`);

  const browser = await launchChrome();
  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));

  let result = null;
  try {
    await page.goto(url, { waitUntil: "load", timeout: 60000 });
    await page.waitForFunction(() => window.__H264__ !== undefined, null, { timeout: 120000 });
    result = await page.evaluate(() => window.__H264__);
  } finally {
    await browser.close();
    server.close();
  }

  if (logs.length) console.log(logs.join("\n"));
  console.log(result?.text ?? "(no report)");
  const ok = Boolean(result?.ok);
  console.log(ok ? "[h264] PASS" : "[h264] FAIL");
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
