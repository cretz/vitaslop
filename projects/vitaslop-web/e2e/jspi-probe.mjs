// Probe: does the Chrome we drive expose JSPI (WebAssembly.promising / Suspending)?
// Also runs a tiny suspend/resume proof if present. This gates whether the browser
// preemptive scheduler can be built+tested on JSPI here.
import { launchChrome } from "./harness.mjs";

const browser = await launchChrome();
const page = await browser.newPage();
const logs = [];
page.on("console", (m) => logs.push(m.text()));
page.on("pageerror", (e) => logs.push("pageerror: " + e.message));

const result = await page.evaluate(async () => {
  const out = { ua: navigator.userAgent, promising: typeof WebAssembly.promising, suspending: typeof WebAssembly.Suspending };
  if (out.promising !== "function" || out.suspending !== "function") return out;
  // Minimal JSPI proof: a wasm module that imports one function `env.f` and calls it,
  // returning its result. Wrap `f` as Suspending (returns a Promise -> guest suspends),
  // and the exported `run` as promising, so we prove a suspend+resume round-trips.
  // (module (import "env" "f" (func $f (result i32)))
  //  (func (export "run") (result i32) call $f))
  const bytes = new Uint8Array([
    0x00,0x61,0x73,0x6d,0x01,0x00,0x00,0x00,
    0x01,0x09,0x02,0x60,0x00,0x01,0x7f,0x60,0x00,0x01,0x7f, // types: ()->i32 twice
    0x02,0x09,0x01,0x03,0x65,0x6e,0x76,0x01,0x66,0x00,0x00, // import env.f : type0
    0x03,0x02,0x01,0x01,                                     // func: type1
    0x07,0x07,0x01,0x03,0x72,0x75,0x6e,0x00,0x01,            // export "run" func1
    0x0a,0x06,0x01,0x04,0x00,0x10,0x00,0x0b,                 // body: call $f
  ]);
  try {
    const mod = await WebAssembly.compile(bytes);
    const f = new WebAssembly.Suspending(async () => { await Promise.resolve(); return 42; });
    const inst = await WebAssembly.instantiate(mod, { env: { f } });
    const run = WebAssembly.promising(inst.exports.run);
    const v = await run();
    out.roundtrip = v;
  } catch (e) {
    out.error = String(e);
  }
  return out;
});

console.log(JSON.stringify(result, null, 2));
for (const l of logs) console.log("[page]", l);
await browser.close();
