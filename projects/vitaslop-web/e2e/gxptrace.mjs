// gxptrace.mjs - WHERE DOES A DIVERGING PROGRAM FIRST GO WRONG?
//
// >>> WHY THIS EXISTS. `gxpexec` grades a whole register file at the END of a program. When it
// says a 183-instruction vertex program disagrees, that is true and useless: the remaining rows
// in the corpus remainder are 63, 115 and 183 instructions long, and reading them by eye is the
// guess-and-check this project keeps paying for.
//
// `write_one_blob_as_a_trace_case` records a checksum of the whole register file at every
// TOP-LEVEL instruction boundary on the reference side; the module it writes records the same
// checksums on the GPU. The first index whose incoming state differs names the instruction that
// ran just before it, and that one is the culprit.
//
// Both sides key by INSTRUCTION INDEX rather than by a step counter. The reference follows
// branches, so its execution order is not the emitted order and a counter would put the two
// traces out of phase at the first taken branch; "the state on last arriving at instruction k"
// is the same statement on both sides and needs no agreement about control flow.
//
// Run: node gxptrace.mjs <dir written by the test>
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { launchChrome, startServer, webDir } from "./harness.mjs";
import { installLaneValue } from "./caseinputs.mjs";

const dir = process.argv[2];
if (!dir) {
  console.error("usage: node gxptrace.mjs <dir of .trace.json + .trace.wgsl>");
  process.exit(2);
}
const names = readdirSync(dir).filter((f) => f.endsWith(".trace.json"));
if (!names.length) {
  console.error(`no .trace.json in ${dir}`);
  process.exit(2);
}

const server = await startServer(webDir);
const browser = await launchChrome();
const page = await browser.newPage();
// The seeded-input generator, installed as a script rather than built with `new Function`:
// the page's CSP forbids eval. See `caseinputs.mjs`.
await installLaneValue(page);
await page.goto(`http://127.0.0.1:${server.address().port}/`, { waitUntil: "load", timeout: 60000 });

let failed = 0;
for (const file of names) {
  const c = JSON.parse(readFileSync(join(dir, file), "utf8"));
  const src = readFileSync(join(dir, file.replace(".json", ".wgsl")), "utf8");

  // The GPU half is web/runner/gxptrace-run.js - shared with the device runner.
  const got = await page.evaluate(async ({ src, c }) => (await import("/runner/gxptrace-run.js")).runTrace({ src, c }), { src, c });

  console.log(`\n=== ${c.name}: ${c.instrs.length} instructions, ${c.checkpoints.length} checkpoints`);
  if (got.error) {
    console.log(`  RIG FAILURE - the trace module did not run: ${got.error}`);
    failed++;
    continue;
  }
  // Only the CHECKPOINTED indices mean anything: every other slot is zero on both sides
  // because nothing writes it, and reporting those as agreement would flatter the instrument.
  const checked = c.checkpoints;
  let first = null;
  let agreed = 0;
  for (const k of checked) {
    if (c.trace[k] === got.trace[k]) {
      agreed++;
      continue;
    }
    if (first === null) first = k;
  }
  if (first === null) {
    console.log(
      `  the two sides agree at ALL ${agreed} checkpoints, INCLUDING the final one - so this ` +
        `program does not diverge, or it diverges only inside a conditional or a loop body, ` +
        `where there are no top-level boundaries to check.`,
    );
    continue;
  }
  const before = checked.filter((k) => k < first);
  const culprit = before.length ? before[before.length - 1] : null;
  // >>> THE TRACE IS AN EXACT COMPARISON, and the differential is not. A checksum fires on any
  // difference at all, including the last-bit ones `gxpexec` grades as "within tolerance" - so
  // a first difference here is "where the two sides first stopped being identical", which is
  // where to LOOK, not a verdict that the program is wrong. A program the differential passes
  // can still have a first difference, and that is the instrument working.
  console.log(`  agreed at ${agreed} of ${checked.length} checkpoints`);
  console.log(`  FIRST DIFFERENCE arriving at instruction ${first}`);
  console.log(
    `    reference 0x${(c.trace[first] >>> 0).toString(16).padStart(8, "0")}  ` +
      `gpu 0x${(got.trace[first] >>> 0).toString(16).padStart(8, "0")}`,
  );
  if (culprit === null) {
    console.log("    nothing ran before it - the INPUTS differ, which is a rig fault, not a shader one");
  } else {
    console.log(`  so the culprit is the run between checkpoint ${culprit} and ${first}:`);
    for (let k = culprit; k < first; k++) {
      if (c.instrs[k]) console.log(`      ${c.instrs[k]}`);
    }
  }
  failed++;
}

await browser.close();
server.close();
process.exit(failed ? 1 : 0);
