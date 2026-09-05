// Tests the storage half of persistence: the guest's saved state in this browser, and the
// promise that it lives somewhere the imported title does not.
//
// # What this covers that a cargo test cannot
// The Rust side is covered where it belongs (`cargo test -p vitaslop-runtime`: what may be
// inside a container, what a restore does, and that a container cannot name a path outside
// the guest's own saved state). What that cannot reach is the browser: two OPFS trees, a
// write that replaces rather than appends, and a "clear my save" that must not cost a
// gigabyte-long re-import. There is no OPFS in a native test process, so this runs here.
//
// Run:  node e2e/gamedata.mjs
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
  await page.goto(url + "debug/gamedata-test.html", { waitUntil: "load", timeout: 30000 });
  const r = await page.evaluate(() => window.__gameDataTest());

  check("a title with no save reports nothing stored", r.emptyBefore === true);
  check("the fixture title imported", r.titleImported === true);
  check("a container written is the container read back", r.roundTrip === true);
  check("the stored container reports its size and a write time", r.info.bytes === 1024 && r.info.hasModified, JSON.stringify(r.info));
  check(
    "a SMALLER container replaces the previous one whole",
    r.overwrite === true,
    r.overwrite === true ? "" : "bytes of the previous save survived the write"
  );
  // The two below are the user-visible promise of the whole design.
  check("saving game data leaves the imported title alone", r.titleSurvivedSave === true);
  check("CLEARING game data leaves the imported title alone", r.titleSurvivedClear === true);
  check("clearing really removes it", r.cleared === true);
  check(
    "game data and titles are separate top-level directories",
    r.topLevel.includes("games") && r.topLevel.includes("gamedata"),
    r.topLevel.join(", ")
  );
} catch (e) {
  check("game data test ran", false, e.message);
} finally {
  await browser.close();
  server.close();
}

console.log(failures === 0 ? "\ngamedata: ALL PASS" : `\ngamedata: ${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
