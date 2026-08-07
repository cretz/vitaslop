// Tests the browser's game storage: the OPFS read path the emulator serves guest file
// reads from.
//
// # What this covers that nothing else can
// The emulator's own boot only ever reads WHOLE files out of storage (the dump manifest
// and the loadable modules). The guest reads at OFFSETS, in small pieces, for the whole
// run - a completely different path through the same code. When that path is wrong
// nothing reports it: the guest is handed the wrong bytes and traps somewhere in its own
// code tens of thousands of host calls later, with nothing anywhere pointing at storage.
//
// It cannot be a cargo test. The code under test is the wasm/JS boundary to
// `FileSystemSyncAccessHandle`, and there is no OPFS in a native test process. So it runs
// here, against a synthetic fixture, in seconds - rather than being re-proven at every
// boot of a gigabyte-sized title.
//
// Run:  node e2e/opfs.mjs
import { requireBundle, startServer, webDir, launchChrome } from "./harness.mjs";

requireBundle();
const server = await startServer(webDir);
const url = `http://127.0.0.1:${server.address().port}/`;
const browser = await launchChrome();
const page = await browser.newPage();
page.on("console", (m) => console.log(`[page:${m.type()}] ${m.text()}`));
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` - ${detail}` : ""}`);
  if (!ok) failures++;
};

try {
  await page.goto(url + "opfs-test.html", { waitUntil: "load", timeout: 30000 });
  const result = await page.evaluate(() => window.__opfsTest());

  // Sizes with awkward remainders against the 997-byte verify chunk, so an off-by-one in
  // the last partial read shows up rather than landing exactly on a boundary.
  check("import + sync handles opened", result.opened === result.written, `${result.opened} files`);
  check("whole and chunked reads agree", result.verify.ok === true, result.verify.detail);
  check(
    "a read past end of file is SHORT, not padded",
    result.shortRead.got === result.shortRead.expected,
    `got ${result.shortRead.got}, expected ${result.shortRead.expected}`
  );
  check(
    "an offset read returns the bytes at that offset",
    result.offsetRead.ok === true,
    result.offsetRead.detail
  );
  check(
    "a second import of the same title is REUSED, not rewritten",
    result.reused === true,
    result.reused === true ? "" : "importTitle re-wrote an already-complete title"
  );
} catch (e) {
  check("opfs test ran", false, e.message);
} finally {
  await browser.close();
  server.close();
}

console.log(failures === 0 ? "\nopfs: ALL PASS" : `\nopfs: ${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
