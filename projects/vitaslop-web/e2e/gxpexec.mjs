// gxpexec.mjs - RUN every emitted shader on the real GPU through the real browser compiler and
// check the numbers against the CPU reference.  node gxpexec.mjs <dir of cases> [--limit N]
//
// >>> WHY THIS EXISTS. `tintcheck.mjs` proves a module COMPILES. It cannot prove the module
// COMPUTES THE RIGHT THING, and every graphics defect this project has chased one title at a
// time was a wrong number in a module that compiled perfectly: a write mask that dropped a lane,
// a load offset biased by one register, an index scale of 2 where the hardware uses 1. Each was
// found because a title drew something visibly wrong, long after the blob was captured.
//
// `execcases.rs` writes, per corpus blob, a compute module wrapping the SAME emitted body the
// renderer ships plus the register file the crate's own USSE interpreter computes for the same
// inputs. This runs the GPU half and diffs. The emitter and the interpreter share the decoder
// and nothing else, so a disagreement is a defect in one of them - visible with no title, no
// recipe and no frame.
//
// >>> IT IS BUILT FOR CI, WHICH MEANS IT IS BUILT AROUND ONE COST. The expensive part is Tint
// compiling each module; everything else must not add to it. So:
//   * ONE GPUDevice for the whole run (a device per case is thousands of command queues and the
//     ~400th fails with E_OUTOFMEMORY, which reads as a shader failure and is not one);
//   * the whole batch is encoded into ONE command buffer and submitted ONCE, rather than a
//     submit-and-wait per case;
//   * the inputs are REGENERATED in the page from the case's seed instead of being shipped, and
//     the comparison happens in the page, so the only thing crossing the CDP boundary is the
//     list of divergences. A round trip per case is what made the naive version minutes long.
import { readdirSync, readFileSync } from "node:fs";
import { join, basename } from "node:path";
import { launchChrome, startServer, webDir } from "./harness.mjs";
import { installLaneValue } from "./caseinputs.mjs";

const dir = process.argv[2];
if (!dir) {
  console.error("usage: node gxpexec.mjs <dir of .wgsl + .json cases> [--limit N]");
  process.exit(2);
}
const limitArg = process.argv.indexOf("--limit");
const limit = limitArg > 0 ? Number(process.argv[limitArg + 1]) : Infinity;

// >>> THE BLOBS THAT NEVER BECAME CASES, so the pass rate below cannot read as coverage.
// The writer leaves this beside the cases because it is the only thing that knows: a blob
// excluded for a screen-space derivative, a texture gather, fragment pipeline state or a
// global register is a program NOTHING here checked, and "1,029 cases, 18 diverged" says
// nothing about it [[vitaslop-a-drop-count-needs-its-draw-count]].
let coverage = null;
try {
  coverage = JSON.parse(readFileSync(join(dir, "coverage.summary"), "utf8"));
} catch {
  // An older case directory has no summary. Saying so is the honest report; assuming
  // "nothing was skipped" is the failure this exists to prevent.
}

const names = readdirSync(dir)
  .filter((f) => f.endsWith(".json"))
  .sort()
  .slice(0, limit === limit ? limit : Infinity)
  .map((f) => basename(f, ".json"));
if (names.length === 0) {
  console.error(`no .json cases in ${dir}`);
  process.exit(2);
}

const cases = names.map((n) => ({
  ...JSON.parse(readFileSync(join(dir, `${n}.json`), "utf8")),
  src: readFileSync(join(dir, `${n}.wgsl`), "utf8"),
}));

// >>> THE RENDER RIG'S CASES, AND THE TEXELS THEY SAMPLE.
//
// A case whose program gathers, differences across the quad, discards or writes a depth cannot
// run in a compute dispatch at all, so the writer emits it as a RENDER module (`stage:
// "fragment"`) and it needs a real texture bound. The texels come from the case directory as
// BYTES rather than from a generator reimplemented here: a second copy of the avalanche would
// be a second thing to keep in step with the reference, and a drift would hand the GPU
// different texels and be reported as a translation defect.
// >>> ONLY A RENDER CASE THAT DECLARES A SAMPLER UNIT NEEDS THE TEXELS. The AUTHORED
// fragment-pipeline cases - a discard and a written depth - run on the render rig because a
// compute dispatch has neither, and they sample nothing at all; demanding a texture file from
// them refused a whole directory over a binding no case in it declares.
const renderCases = cases.filter((c) => c.stage === "fragment" && (c.units?.length ?? 0) > 0).length;
let texB64 = null;
if (renderCases > 0) {
  try {
    texB64 = readFileSync(join(dir, "casetex.bin")).toString("base64");
  } catch {
    console.error(
      `${renderCases} case(s) need the render rig's textures and ${join(dir, "casetex.bin")} is ` +
        "not there - those cases would sample nothing. Re-write the cases.",
    );
    process.exit(2);
  }
}
// The texture geometry the writer used. Stated once here and asserted against the file's
// length, so a rig whose texture size changed cannot quietly upload a differently-shaped set.
const TEX_SIZE = 64;
const TEX_UNITS = 16;
// Seven planes per unit: the flat texture, then the cube's six faces. The file is the FLAT
// block for every unit followed by the CUBE block, six faces contiguous per unit.
const TEX_LAYERS = 7;
if (texB64 !== null) {
  const want = TEX_UNITS * TEX_SIZE * TEX_SIZE * 4 * TEX_LAYERS;
  const got = Buffer.from(texB64, "base64").length;
  if (got !== want) {
    console.error(
      `casetex.bin is ${got} bytes, expected ${want} ` +
        `(${TEX_UNITS} units x ${TEX_LAYERS} planes x ${TEX_SIZE}^2 RGBA8)`,
    );
    process.exit(2);
  }
}

const t0 = Date.now();
const server = await startServer(webDir);
const port = server.address().port;
const browser = await launchChrome();
const page = await browser.newPage();
// The seeded-input generator, installed as a script rather than built with `new Function`:
// the page's CSP forbids eval. See `caseinputs.mjs`.
await installLaneValue(page);
await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "load", timeout: 60000 });

const have = await page.evaluate(async () => !!(navigator.gpu && (await navigator.gpu.requestAdapter())));
if (!have) {
  console.error("no WebGPU adapter in this Chrome - nothing was checked");
  await browser.close();
  process.exit(3);
}

const features = await page.evaluate(async () => {
  const adapter = await navigator.gpu.requestAdapter();
  const want = ["shader-f16"].filter((x) => adapter.features.has(x));
  globalThis.__dev = await adapter.requestDevice({ requiredFeatures: want });
  globalThis.__dev.onuncapturederror = (e) => {
    globalThis.__uncaptured = (globalThis.__uncaptured || []).concat(String(e.error.message));
  };
  return want;
});
console.log(
  `device features: ${features.join(", ") || "(none)"}; ${cases.length} cases` +
    (renderCases ? ` (${renderCases} on the RENDER rig)` : ""),
);

// The stand-in texture bytes reach the page ONCE, before any batch runs, so the per-batch
// round trip stays the list of divergences it was.
if (texB64 !== null) {
  await page.evaluate(
    ({ b64, size, units }) => {
      globalThis.__texb64 = b64;
      globalThis.__texSize = size;
      globalThis.__texUnits = units;
    },
    { b64: texB64, size: TEX_SIZE, units: TEX_UNITS },
  );
}

// One batch is one command buffer. Sized so the resident buffer set stays small (a case is
// 4 KB in + 6 KB out) while the number of submits stays in the tens rather than the thousands.
const BATCH = 96;
let failed = 0;
let diverged = 0;
let exact = 0;
let close = 0;
// Lanes the case writer could not assign a view, graded in both and agreeing if either does.
let ambiguousLanes = 0;
// Lanes where one side held a subnormal and the other zero - a flush WGSL permits. See
// `judgeAt`; counted because it is weaker coverage, not because it is a failure.
let flushedLanes = 0;
const report = [];

const batchTimes = [];
for (let i = 0; i < cases.length; i += BATCH) {
  const batch = cases.slice(i, i + BATCH);
  const tb = Date.now();
  // The batch itself is web/runner/gxpbatch.js - shared with the device runner.
  const res = await page.evaluate(async ({ batch }) => (await import("/runner/gxpbatch.js")).runBatch({ batch }), { batch });

  if (res.invalid) {
    console.error(`\nRIG FAILURE - the batch's command buffer never ran, so ${res.invalid.length} cases were NOT checked:`);
    console.error(`  ${res.error}`);
    console.error("  No verdict is possible from this run.");
    await browser.close();
    process.exit(4);
  }
  for (const r of res) {
    ambiguousLanes += r.ambiguous ?? 0;
    flushedLanes += r.flushed ?? 0;
    if (r.status === "compile") {
      failed++;
      report.push(r);
    } else if (r.status === "diverged") {
      diverged++;
      report.push(r);
    } else if (r.status === "close") close++;
    else exact++;
  }
  batchTimes.push([i, Date.now() - tb]);
  process.stdout.write(`\r  ${Math.min(i + BATCH, cases.length)}/${cases.length} ...`);
}
// >>> A NEWLINE, NOT A BARE CARRIAGE RETURN. `\r` rewrites the line on a terminal and does
// NOTHING in a redirected file, so the first report line below was CONCATENATED onto the
// progress line - and `grep "^DIVERGED"` over the log then found 17 of 18. The count and the
// list disagreed by one and the missing one was `worst Infinity ULP`, the loudest kind there is.
process.stdout.write("\r" + " ".repeat(40) + "\r\n");

// >>> ATTRIBUTE A DIVERGENCE BEFORE REPORTING IT. The split by half-precision content is kept
// because it is the axis that has mattered: the reference once held one f32 per lane and so had
// no authority at all over the 65% of the corpus that packs two F16s into a word (490 of 670
// cases disagreed). It now models the packed register file, and the split is reported so the
// claim stays checkable - if half and full programs diverge at very different rates, the
// half-precision model is still the place to look.
const halfDiv = report.filter((r) => r.status === "diverged" && r.half > 0).length;
const fullDiv = report.filter((r) => r.status === "diverged" && !r.half).length;

for (const r of report) {
  if (r.status === "compile") {
    console.log(`COMPILE-FAIL ${r.name}`);
    for (const m of r.messages) console.log(`    ${m}`);
  } else {
    console.log(
      `DIVERGED ${r.name} (${r.kind}, ${r.instrs} instrs, ${r.half} half-precision) worst ${r.worst} ULP`,
    );
    for (const b of r.bad) {
      const hex = (v) => (v === undefined ? "?" : `0x${(v >>> 0).toString(16).padStart(8, "0")}`);
      const where = b.reg === "" ? b.bank : `${b.bank}[${b.reg}]`;
      console.log(
        `    ${where} ${b.view}: reference ${b.want}  gpu ${b.got}  ${b.ulps} ULP` +
          `  raw ${hex(b.rawWant)} vs ${hex(b.rawGot)}`,
      );
    }
  }
}

// Per-batch wall time, so a run that grows slower says WHERE rather than only that it did.
// (Opening the texture tranche took a 9 s run to 171 s, and the shape of this list is what
// says whether that is a few pathological modules or a cost that rises with every batch.)
const slowest = [...batchTimes].sort((a, b) => b[1] - a[1]).slice(0, 5);
console.log(`\nslowest batches (case index, ms): ${slowest.map(([i, ms]) => `${i}:${ms}`).join("  ")}`);
console.log(`batch times in order (ms): ${batchTimes.map(([, ms]) => ms).join(" ")}`);
const secs = ((Date.now() - t0) / 1000).toFixed(1);
const halfCases = cases.filter((c) => c.half > 0).length;
const fullCases = cases.length - halfCases;
console.log(
  `\n${cases.length} cases in ${secs}s: ${exact} exact, ${close} within tolerance, ` +
    `${diverged} DIVERGED, ${failed} failed to compile`,
);
console.log(
  `  of the divergences: ${fullDiv} of ${fullCases} full-precision programs, ` +
    `${halfDiv} of ${halfCases} carrying half-precision instructions`,
);
if (flushedLanes) {
  // >>> A FLUSHED LANE IS WEAKER COVERAGE AND HAS TO SAY SO, for the same reason a lenient one
  // does. WGSL permits an implementation to flush subnormals to zero, so a subnormal against a
  // zero is conformance rather than a defect - but inside that window a DROPPED WRITE looks the
  // same, and so does any other value the expectation could have held.
  console.log(
    `  ${flushedLanes} differing lane(s) were a permitted SUBNORMAL FLUSH: one side held a ` +
      `subnormal and the other an exact zero, which WGSL allows and every backend here does`,
  );
}
if (ambiguousLanes) {
  // >>> A LENIENT LANE IS WEAKER COVERAGE AND HAS TO SAY SO. These are lanes whose last writer
  // sits behind a predicate or a branch and disagrees with an earlier one about whether the word
  // holds one f32 or two halves, so they are graded in BOTH views and pass if either does. Left
  // uncounted, a rise here would read as the differential getting cleaner.
  console.log(
    `  ${ambiguousLanes} differing lane(s) were graded LENIENTLY: their view is not statically ` +
      `determined (a conditional writer disagrees), so both readings were tried`,
  );
}
if (coverage) {
  const reasons = Object.entries(coverage.reasons ?? {})
    .sort((a, b) => b[1] - a[1])
    .map(([why, n]) => `${n} ${why}`)
    .join(", ");
  console.log(
    `  COVERAGE: ${coverage.written} cases from ${coverage.blobs} blobs; ` +
      `${coverage.skipped} blobs were SKIPPED and are checked by nothing above`,
  );
  if (reasons) console.log(`    skipped for: ${reasons}`);
  if (coverage.rendered) {
    console.log(
      `    ${coverage.rendered} of those cases run on the RENDER rig (a real fragment stage: ` +
        `gather, derivatives, kill, written depth); ${coverage.written - coverage.rendered} ` +
        "on the compute rig",
    );
  }
  if (coverage.trivial) {
    // >>> AN ALL-ZERO EXPECTATION IS A WEAK CASE, NOT AN EMPTY ONE. The reference produced no
    // nonzero register for these, so they still catch a GPU that writes something - but they
    // cannot check a VALUE, and counting them beside the ones that can overstates what the
    // pass rate means. Why so many is its own question: a program whose seeded inputs drive it
    // to zero is a program the seed failed to exercise.
    const real = coverage.written - coverage.trivial;
    console.log(
      `    of those cases, ${coverage.trivial} expect an ALL-ZERO register file: they catch a ` +
        `spurious write but check no value, so ${real} cases carry the verdict`,
    );
  }
} else {
  console.log(
    "  COVERAGE: unknown - this case directory carries no `coverage.summary`, so how many " +
      "blobs were skipped rather than checked is not recoverable from these numbers.",
  );
}
const uncaptured = await page.evaluate(() => globalThis.__uncaptured || []);
if (uncaptured.length) console.log(`uncaptured device errors: ${uncaptured.length}\n  ${uncaptured.slice(0, 5).join("\n  ")}`);
await browser.close();
process.exit(diverged === 0 && failed === 0 ? 0 : 1);
