// tintcheck.mjs - hand a DIRECTORY of WGSL modules to Chrome's own compiler and report every
// message it produces.  node tintcheck.mjs <dir>
//
// >>> WHY THIS EXISTS. The corpus tests validate emitted WGSL with naga, and naga is not the
// compiler the product uses: Tint refuses things naga accepts, and the one that matters here is
// `dpdx`/`dpdy` outside uniform control flow - which compiles on the desktop and kills the
// browser's run worker with an invalid pipeline. Until now the only Tint in reach was a full
// browser replay of the title that binds the pair, which cannot check a shader no recipe reaches.
// `corpus.rs::write_every_linked_pair_wgsl` writes every module a corpus can produce; this
// compiles all of them in the real browser in one go.
//
// A module is compiled for its own entry points only (no pipeline, no bindings), which is exactly
// the stage that enforces the uniformity rules. Anything a PIPELINE would reject (a binding layout
// mismatch, a missing target) is not this script's question.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
// The project's own launcher and static server: WebGPU is a SECURE-CONTEXT api and Chrome does
// not expose `navigator.gpu` on `about:blank`, so the page has to come over http://127.0.0.1.
// (That cost one "no WebGPU adapter in this Chrome" that was nothing to do with the GPU.)
import { launchChrome, startServer, webDir } from "./harness.mjs";

const dir = process.argv[2];
if (!dir) {
  console.error("usage: node tintcheck.mjs <dir of .wgsl>");
  process.exit(2);
}
const files = readdirSync(dir).filter((f) => f.endsWith(".wgsl")).sort();
if (files.length === 0) {
  console.error(`no .wgsl in ${dir}`);
  process.exit(2);
}

const server = await startServer(webDir);
const port = server.address().port;
const browser = await launchChrome();
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "load", timeout: 60000 });

const have = await page.evaluate(async () => !!(navigator.gpu && (await navigator.gpu.requestAdapter())));
if (!have) {
  console.error("no WebGPU adapter in this Chrome - nothing was checked");
  await browser.close();
  process.exit(3);
}

// ONE DEVICE for the whole run. A device per module is 595 D3D12 command queues and the 400th
// one fails with E_OUTOFMEMORY - which reads as a shader failure and is not one.
const features = await page.evaluate(async () => {
  const adapter = await navigator.gpu.requestAdapter();
  // Ask for every feature the emitter may have used; a module using a feature the device was not
  // given fails for a reason that is about the REQUEST and not about the shader.
  const want = ["dual-source-blending", "shader-f16", "float32-filterable"].filter((x) =>
    adapter.features.has(x),
  );
  globalThis.__dev = await adapter.requestDevice({ requiredFeatures: want });
  return want;
});
console.log(`device features: ${features.join(", ") || "(none of the three asked for)"}`);

let failed = 0;
let warned = 0;
for (const f of files) {
  const src = readFileSync(join(dir, f), "utf8");
  const res = await page.evaluate(async (src) => {
    const device = globalThis.__dev;
    device.pushErrorScope("validation");
    const mod = device.createShaderModule({ code: src });
    const info = await mod.getCompilationInfo();
    const err = await device.popErrorScope();
    return {
      messages: info.messages.map((m) => `${m.type} ${m.lineNum}:${m.linePos} ${m.message}`),
      validation: err ? err.message : null,
    };
  }, src);
  const errs = res.messages.filter((m) => m.startsWith("error"));
  const warns = res.messages.filter((m) => m.startsWith("warning"));
  if (errs.length || res.validation) {
    failed++;
    console.log(`FAIL ${f}`);
    for (const m of errs) console.log(`  ${m}`);
    if (res.validation) console.log(`  validation: ${res.validation.split("\n").slice(0, 6).join(" | ")}`);
  } else if (warns.length) {
    warned++;
    console.log(`warn ${f}: ${warns.join(" | ")}`);
  }
}
console.log(`\n${files.length} modules compiled by Chrome: ${failed} FAILED, ${warned} with warnings`);
await browser.close();
process.exit(failed === 0 ? 0 : 1);
