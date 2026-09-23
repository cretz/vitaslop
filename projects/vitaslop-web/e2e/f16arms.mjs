// f16arms.mjs - DO THE TWO ROUND-TO-NEAREST f16 ARMS COMPUTE THE SAME NUMBERS?
//
// >>> WHY THIS IS A SEPARATE INSTRUMENT. Which arm a run uses is chosen from the DEVICE's
// features: an adapter with `shader-f16` gets the language's own conversion, one without gets
// portable bit arithmetic. If the two disagree, the same title renders differently on two
// phones - which is the exact defect the rounding fix exists to remove, reintroduced by the
// fix's own implementation.
//
// `gxpexec` cannot answer this. It grades each arm against the CPU REFERENCE, and two arms can
// both pass while differing from each other: a case that is "exact" under one and "within
// tolerance" under the other has moved, and the run that measured 861/150/18 against 848/163/18
// says exactly that much and no more. This compares the two arms' REGISTER FILES to each
// other, per case, with the reference out of the picture.
//
// It reads the case directories `f16ab.sh` already wrote, so the modules are the ones that were
// graded - not re-emitted ones that would have to be trusted to match.
//
// Run: node f16arms.mjs <portable case dir> <native case dir>
import { readdirSync, readFileSync } from "node:fs";
import { join, basename } from "node:path";
import { launchChrome, startServer, webDir } from "./harness.mjs";
import { installLaneValue } from "./caseinputs.mjs";

const [portDir, natDir] = process.argv.slice(2);
if (!portDir || !natDir) {
  console.error("usage: node f16arms.mjs <portable case dir> <native case dir>");
  process.exit(2);
}

const names = readdirSync(portDir)
  .filter((f) => f.endsWith(".json"))
  .sort()
  .map((f) => basename(f, ".json"));
if (!names.length) {
  console.error(`no .json cases in ${portDir}`);
  process.exit(2);
}

const cases = [];
let onlyPortable = 0;
for (const n of names) {
  let nat;
  try {
    nat = readFileSync(join(natDir, `${n}.wgsl`), "utf8");
  } catch {
    onlyPortable++;
    continue;
  }
  const meta = JSON.parse(readFileSync(join(portDir, `${n}.json`), "utf8"));
  cases.push({
    name: n,
    seed: meta.seed,
    lanes: meta.lanes,
    half: meta.half,
    mem: meta.mem,
    a: readFileSync(join(portDir, `${n}.wgsl`), "utf8"),
    b: nat,
  });
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

const features = await page.evaluate(async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return null;
  const want = ["shader-f16"].filter((x) => adapter.features.has(x));
  globalThis.__dev = await adapter.requestDevice({ requiredFeatures: want });
  return want;
});
if (!features) {
  console.error("no WebGPU adapter in this Chrome - nothing was checked");
  await browser.close();
  process.exit(3);
}
if (!features.includes("shader-f16")) {
  // The native arm's modules carry `enable f16;` and will not compile here. Saying so is the
  // answer; reporting "0 cases differ" would be a rig that cannot see its subject.
  console.error("this adapter has no `shader-f16`, so the NATIVE arm cannot run - nothing compared");
  await browser.close();
  process.exit(3);
}
console.log(`device features: ${features.join(", ")}; ${cases.length} cases (${onlyPortable} only in the portable set)`);

const BATCH = 48;
const report = [];
const nanOnly = [];
let same = 0;
let failed = 0;

for (let i = 0; i < cases.length; i += BATCH) {
  const batch = cases.slice(i, i + BATCH);
  const res = await page.evaluate(async ({ batch }) => {
    const dev = globalThis.__dev;
    // The seeded inputs, from `caseinputs.mjs` - the SAME generator `gxpexec` uses, so both
    // arms are handed the bytes the reference was graded against and not a second twin's.
    const laneValue = globalThis.__laneValue;
    const out = [];
    const retired = [];
    for (const c of batch) {
      const lanes = c.lanes;
      const input = new Float32Array(lanes * 2);
      for (let n = 0; n < lanes; n++) {
        input[n] = laneValue(c.seed, n);
        input[lanes + n] = laneValue(c.seed, lanes + n);
      }
      // The same per-program overrides the case carries - see `caseinputs.mjs`. Both arms are
      // fed the identical file, so this is about comparing them on the inputs the case is FOR
      // rather than about either arm's correctness.
      globalThis.__applyCaseInputs(c, input);
      const results = [];
      let bad = null;
      for (const src of [c.a, c.b]) {
        dev.pushErrorScope("validation");
        const mod = dev.createShaderModule({ code: src });
        const info = await mod.getCompilationInfo();
        const errs = info.messages.filter((m) => m.type === "error").map((m) => m.message);
        if (errs.length) {
          bad = errs.join("; ");
          await dev.popErrorScope();
          break;
        }
        const pipe = dev.createComputePipeline({
          layout: "auto",
          compute: { module: mod, entryPoint: "cs_main" },
        });
        const inBuf = dev.createBuffer({
          size: input.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        dev.queue.writeBuffer(inBuf, 0, input);
        const outBytes = lanes * 4 * 4; // r, o, i, pa - see `gxpexec`
        const outBuf = dev.createBuffer({
          size: outBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const read = dev.createBuffer({
          size: outBytes,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const entries = [
          { binding: 0, resource: { buffer: inBuf } },
          { binding: 1, resource: { buffer: outBuf } },
        ];
        // A case whose program loads guest memory declares a third binding; the bytes are the
        // case's own and are the same for both arms.
        let memBuf = null;
        if (c.mem && c.mem.length) {
          const words = new Uint32Array(c.mem);
          memBuf = dev.createBuffer({
            size: words.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          dev.queue.writeBuffer(memBuf, 0, words);
          entries.push({ binding: 2, resource: { buffer: memBuf } });
        }
        const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
        const enc = dev.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipe);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(1);
        pass.end();
        enc.copyBufferToBuffer(outBuf, 0, read, 0, outBytes);
        dev.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        results.push(Array.from(new Uint32Array(read.getMappedRange().slice(0))));
        read.unmap();
        const err = await dev.popErrorScope();
        if (err) bad = err.message;
        retired.push(inBuf, outBuf, read);
        if (memBuf) retired.push(memBuf);
        if (bad) break;
      }
      if (bad || results.length < 2) {
        out.push({ name: c.name, status: "failed", why: bad ?? "one arm did not run" });
        continue;
      }
      const [pa, na] = results;
      let differing = 0;
      let worstUlp = 0;
      let firstLane = -1;
      // >>> A DIFFERENCE THAT IS ONLY A NaN PAYLOAD IS NOT A DIFFERENCE IN A NUMBER, and the
      // two are not the same finding. Both arms narrow a NaN to a NaN; WHICH one is not
      // something this translation decides, and `gxpexec` already grades it that way ("both
      // non-finite is agreement"). Counting a payload as a numeric divergence would report a
      // portability defect no picture can contain - and it would report it as exactly the
      // constant 511 ULP every time, because 0x7e00 (what the portable arm writes) and 0x7c01
      // (what the device's own conversion produces) are 511 apart as ordinals. A finding whose
      // magnitude is the same constant on every case is a clue that it is not a magnitude.
      let nanOnly = 0;
      const samples = [];
      const ord = (bits, signBit) => {
        const neg = (bits & signBit) !== 0;
        const mag = bits & (signBit - 1);
        return neg ? -mag : mag;
      };
      const isNanH = (h) => (h & 0x7c00) === 0x7c00 && (h & 0x03ff) !== 0;
      const isNanF = (w) => (w & 0x7f800000) === 0x7f800000 && (w & 0x007fffff) !== 0;
      for (let k = 0; k < pa.length; k++) {
        if (pa[k] === na[k]) continue;
        differing++;
        if (firstLane < 0) firstLane = k;
        let laneWorst = 0;
        let laneNumeric = false;
        // Read the lane the way the program stores it: a 16-bit program's word is two halves,
        // and comparing it as one f32 turns a last-bit difference in the high half into a
        // difference of thousands in a number neither arm computed.
        if (c.half > 0) {
          for (const sh of [0, 16]) {
            const x = (pa[k] >>> sh) & 0xffff;
            const y = (na[k] >>> sh) & 0xffff;
            if (x === y) continue;
            if (isNanH(x) && isNanH(y)) continue; // two NaNs; the payload is not ours
            laneNumeric = true;
            laneWorst = Math.max(laneWorst, Math.abs(ord(x, 0x8000) - ord(y, 0x8000)));
          }
        } else if (!(isNanF(pa[k]) && isNanF(na[k]))) {
          laneNumeric = true;
          laneWorst = Math.abs(ord(pa[k], 0x80000000) - ord(na[k], 0x80000000));
        }
        if (!laneNumeric) nanOnly++;
        else {
          worstUlp = Math.max(worstUlp, laneWorst);
          if (samples.length < 3) {
            samples.push({ lane: k, port: pa[k], nat: na[k], ulp: laneWorst });
          }
        }
      }
      out.push(
        differing
          ? {
              name: c.name,
              status: differing === nanOnly ? "nan" : "differ",
              differing,
              nanOnly,
              worstUlp,
              firstLane,
              half: c.half,
              samples,
            }
          : { name: c.name, status: "same" },
      );
    }
    for (const b of retired) b.destroy();
    return out;
  }, { batch });

  for (const r of res) {
    if (r.status === "same") same++;
    else if (r.status === "failed") {
      failed++;
      report.push(r);
    } else if (r.status === "nan") nanOnly.push(r);
    else report.push(r);
  }
  process.stdout.write(`\r  ${Math.min(i + BATCH, cases.length)}/${cases.length} ...`);
}
process.stdout.write("\r");

const differ = report.filter((r) => r.status === "differ");
differ.sort((a, b) => b.worstUlp - a.worstUlp);
for (const r of report.filter((r) => r.status === "failed")) {
  console.log(`FAILED ${r.name}: ${r.why}`);
}
for (const r of differ.slice(0, 40)) {
  console.log(
    `DIFFER ${r.name} (${r.half} half-precision instrs): ${r.differing} lanes ` +
      `(${r.nanOnly} of them NaN-payload only), worst ${r.worstUlp} ULP ` +
      `in the ${r.half > 0 ? "f16" : "f32"} view, first lane ${r.firstLane}`,
  );
  for (const sp of r.samples) {
    console.log(
      `    lane ${sp.lane}: portable 0x${sp.port.toString(16).padStart(8, "0")}  ` +
        `native 0x${sp.nat.toString(16).padStart(8, "0")}  ${sp.ulp} ULP`,
    );
  }
}
const secs = ((Date.now() - t0) / 1000).toFixed(1);
console.log(
  `\n${cases.length} cases in ${secs}s: ${same} BIT-IDENTICAL between the arms, ` +
    `${nanOnly.length} differ ONLY in a NaN payload, ${differ.length} differ NUMERICALLY, ` +
    `${failed} failed to run`,
);
if (nanOnly.length) {
  console.log(
    "  A NaN payload is not a number. Both arms narrow a NaN to a NaN; which bit pattern is " +
      "the unit's business and not this translation's - the same rule `gxpexec` grades by.",
  );
}
if (differ.length) {
  const buckets = { "1 ULP": 0, "2-4": 0, "5-16": 0, ">16": 0 };
  for (const r of differ) {
    const u = r.worstUlp;
    buckets[u <= 1 ? "1 ULP" : u <= 4 ? "2-4" : u <= 16 ? "5-16" : ">16"]++;
  }
  console.log(`  worst-lane ULP between the arms: ${JSON.stringify(buckets)}`);
  console.log(
    "  A LAST-BIT spread is the shader compiler optimising two different modules two ways, " +
      "which is what any two drivers already do. Anything above a few ULP is the ARMS " +
      "disagreeing, and that would be a defect in the fix itself.",
  );
}
await browser.close();
server.close();
process.exit(failed ? 1 : 0);
