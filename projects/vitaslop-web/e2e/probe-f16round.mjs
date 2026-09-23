// probe-f16round.mjs - WHICH WAY DOES THIS DEVICE ROUND f32 -> f16, AND DOES THE TEXT WE SHIP
// ROUND THE WAY THE GUEST'S HARDWARE DOES?
//
// >>> WHY THIS EXISTS. Every half-precision store the recompiler emits used to go through
// WGSL's `pack2x16float`, which lowers to SPIR-V `PackHalf2x16` - and the rounding mode of that
// conversion is IMPLEMENTATION-DEFINED, not specified by the language. The CPU reference rounds
// to nearest even (checked exhaustively against a third-party oracle: 674,872 values, zero
// disagreements), and so does the guest's hardware. If the device truncates instead, every f16
// store disagrees by up to one ULP in a direction that BIASES, and a chain of them drifts -
// which is exactly the shape the corpus residue had: 237 of 278 diverging lanes with the GPU
// nearer zero than the reference.
//
// A bias is not a proof, so this MEASURES the mode instead of inferring it, and it now measures
// four things rather than two:
//
//   pack2x16float    the old store instruction    - the defect
//   native f16()     the language's conversion    - the fast arm's rounding
//   gxp_f16b PACK    our control arm              - must reproduce the old behaviour exactly
//   gxp_f16b PORTABLE / NATIVE   what we ship     - must be round-to-nearest-even, saturating
//
// >>> THE LAST TWO READ THE SHIPPED TEXT OFF DISK. `src/f16rounding/*.wgsl` is `include_str!`d
// by the recompiler and read here, so this probe cannot pass while the emitter ships something
// else [[vitaslop-probe-the-shader-dont-simulate-it]]. A probe carrying its own copy of the
// code under test measures the copy.
//
// Run: node probe-f16round.mjs
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { launchChrome, startServer, webDir, here } from "./harness.mjs";

const F16DIR = join(here, "..", "..", "vitaslop-gxp-shader", "src", "f16rounding");
const armText = (name) => readFileSync(join(F16DIR, `${name}.wgsl`), "utf8");

// >>> THE INPUTS ARE CHOSEN, NOT RANDOM. Each is built from an exactly-representable f16 value
// `v` and its successor `w`, so "half way" and "just past" are exact statements rather than
// artifacts of decimal printing. A random sweep would mostly land on values every mode agrees
// on, and would say nothing.
const LANES = 64;

const preamble = (arm) =>
  (arm === "native" ? "enable f16;\n" : "") + armText(arm) + armText("common");

// The four modules. The first two ask what the LANGUAGE's two conversions do; the last two run
// the helper the emitter actually calls, exactly as a shipped module would define it.
const BUILTIN_PACK = `
struct Out { bits: array<u32, ${LANES}> };
@group(0) @binding(0) var<storage, read> inp: array<f32, ${LANES}>;
@group(0) @binding(1) var<storage, read_write> outp: Out;
@compute @workgroup_size(1)
fn main() {
  for (var i = 0u; i < ${LANES}u; i = i + 1u) {
    // The low half carries the conversion; the high lane is zeroed so the pattern is clean.
    outp.bits[i] = pack2x16float(vec2<f32>(inp[i], 0.0)) & 0xffffu;
  }
}`;

const BUILTIN_NATIVE = `
enable f16;
struct Out { bits: array<u32, ${LANES}> };
@group(0) @binding(0) var<storage, read> inp: array<f32, ${LANES}>;
@group(0) @binding(1) var<storage, read_write> outp: Out;
@compute @workgroup_size(1)
fn main() {
  for (var i = 0u; i < ${LANES}u; i = i + 1u) {
    // The VALUE CONVERSION f32 -> f16, not the pack builtin: a different operation in the
    // language, and the one the native arm's rounding comes from.
    outp.bits[i] = pack2x16float(vec2<f32>(f32(f16(inp[i])), 0.0)) & 0xffffu;
  }
}`;

// >>> THE SHIPPED ARM, AND IT IS EXERCISED THROUGH `gxp_hpk` RATHER THAN `gxp_f16b` ALONE.
// The pair helper is what a folded four-channel store calls - the commonest f16 store there is -
// and in the native arm it is a DIFFERENT expression from two single narrowings. Reading both
// halves out of one pair call checks the placement (low half, high half) at the same time.
const shipped = (arm) => `${preamble(arm)}
struct Out { bits: array<u32, ${LANES}> };
@group(0) @binding(0) var<storage, read> inp: array<f32, ${LANES}>;
@group(0) @binding(1) var<storage, read_write> outp: Out;
@compute @workgroup_size(1)
fn main() {
  for (var i = 0u; i < ${LANES}u; i = i + 1u) {
    // The value in BOTH halves of one pair store, so the low and high placements are read from
    // the same call the emitter emits.
    let both = gxp_hpk(inp[i], inp[i]);
    outp.bits[i] = (both & 0xffffu) | select(0x10000u, 0u, (both >> 16u) == (both & 0xffffu));
  }
}`;

// --- the CPU side of the question: the candidate modes, and the f16 helpers -----------------
const f16ToF32 = (h) => {
  const s = h >> 15 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * Math.pow(2, -24);
  if (e === 0x1f) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * Math.pow(2, e - 15);
};
// Round-to-nearest-even and round-toward-zero, both by exhaustive search over the f16 pattern
// space rather than by bit surgery - slower and obviously right, which is what a probe wants.
const f16Candidates = [];
for (let h = 0; h < 0x10000; h++) {
  const v = f16ToF32(h);
  if (Number.isFinite(v)) f16Candidates.push({ h, v });
}
f16Candidates.sort((a, b) => a.v - b.v);
function neighbours(x) {
  // The representable values bracketing x.
  let lo = null, hi = null;
  for (const c of f16Candidates) {
    if (c.v <= x && (lo === null || c.v > lo.v)) lo = c;
    if (c.v >= x && (hi === null || c.v < hi.v)) hi = c;
  }
  return { lo, hi };
}
// >>> A ZERO CARRIES A SIGN AND THE CANDIDATE LIST DOES NOT. Both 0x0000 and 0x8000 decode to
// the number 0, so sorting by VALUE cannot tell them apart and `neighbours` returns whichever
// it met first. A negative input that rounds to zero rounds to NEGATIVE zero, and without this
// the oracle reports the device wrong on exactly the inputs that test the underflow tie - which
// it did, on -2^-25, while the shader was right.
const signed = (h, x) => (h !== null && (h & 0x7fff) === 0 && (x < 0 || Object.is(x, -0)) ? h | 0x8000 : h);
const modeRTE = (x) => {
  const { lo, hi } = neighbours(x);
  if (!lo || !hi) return null;
  if (lo.v === x) return signed(lo.h, x);
  const dl = x - lo.v, dh = hi.v - x;
  if (dl < dh) return signed(lo.h, x);
  if (dh < dl) return signed(hi.h, x);
  // A tie goes to the even SIGNIFICAND, which is the low bit of the pattern.
  return signed(lo.h % 2 === 0 ? lo.h : hi.h, x);
};
const modeRTZ = (x) => {
  const { lo, hi } = neighbours(x);
  if (!lo || !hi) return null;
  return signed(x >= 0 ? lo.h : hi.h, x);
};

// --- build the inputs ------------------------------------------------------------------------
// Three groups, because they answer three different questions:
//   * the TIE and quarter points around chosen halves - which mode is this?
//   * the SUBNORMAL floor and the zero tie - does the low end round or flush?
//   * the OVERFLOW boundary - does a finite overflow SATURATE to 65504 the way the guest's
//     hardware does, or become an infinity? That one is not a rounding mode at all, and
//     getting it wrong poisons everything downstream: an infinity minus an infinity is a NaN.
const inputs = [];
const note = [];
const push = (x, label) => { inputs.push(x); note.push(label); };
const pick = [0x3c00, 0x3c01, 0x4900, 0x5640, 0x1400, 0x0002, 0x7bfe, 0x3555];
for (const h of pick) {
  const v = f16ToF32(h), w = f16ToF32(h + 1);
  for (const [frac, label] of [[0.25, "quarter"], [0.5, "TIE"], [0.75, "three-quarter"]]) {
    for (const sign of [1, -1]) {
      push(sign * (v + (w - v) * frac), `${sign < 0 ? "-" : "+"}0x${h.toString(16)}+${label}`);
    }
  }
}
// The zero tie (2^-25 rounds to an EVEN zero) and just past it, both signs.
for (const sign of [1, -1]) {
  push(sign * Math.pow(2, -25), `${sign < 0 ? "-" : "+"}2^-25 (zero TIE)`);
  push(sign * Math.pow(2, -25) * 1.5, `${sign < 0 ? "-" : "+"}1.5*2^-25`);
}
const SATURATE = [];
for (const sign of [1, -1]) {
  const s = sign < 0 ? "-" : "+";
  SATURATE.push(inputs.length); push(sign * 65520, `${s}65520 (overflow TIE)`);
  SATURATE.push(inputs.length); push(sign * 1e30, `${s}1e30 (far overflow)`);
}
while (inputs.length < LANES) { inputs.push(0); note.push("(pad)"); }
if (inputs.length > LANES) { console.error(`too many inputs: ${inputs.length} > ${LANES}`); process.exit(2); }

const server = await startServer(webDir);
const port = server.address().port;
const browser = await launchChrome();
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "load", timeout: 60000 });

const runArm = async (src) => await page.evaluate(async ({ src, inputs, LANES }) => {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return { error: "no adapter" };
  // `enable f16;` is only legal when the DEVICE was created with the feature - asking the
  // adapter is not enough, and without it an f16 arm reports "not allowed in the current
  // environment" and reads as an unsupported device rather than an unrequested feature.
  const wantF16 = adapter.features.has("shader-f16");
  const device = await adapter.requestDevice(wantF16 ? { requiredFeatures: ["shader-f16"] } : {});
  device.pushErrorScope("validation");
  const mod = device.createShaderModule({ code: src });
  const info = await mod.getCompilationInfo();
  const msgs = info.messages.filter((m) => m.type === "error").map((m) => m.message);
  if (msgs.length) return { error: msgs.join("; ") };
  const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
  const bytes = LANES * 4;
  const inBuf = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, new Float32Array(inputs));
  const outBuf = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }],
  });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(1); pass.end();
  enc.copyBufferToBuffer(outBuf, 0, read, 0, bytes);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const bits = Array.from(new Uint32Array(read.getMappedRange().slice(0)));
  read.unmap();
  const err = await device.popErrorScope();
  return { bits, adapter: adapter.info?.description ?? "(unnamed)", hasF16: wantF16, err: err?.message ?? null };
}, { src, inputs, LANES });

// The largest finite half, which is what a FINITE overflow saturates to on the guest's hardware.
const SAT = 0x7bff;
const want = (x) => {
  if (Math.abs(x) > 65504 && Number.isFinite(x)) return SAT | (x < 0 ? 0x8000 : 0);
  return modeRTE(x);
};

/// Score one arm's readback against each candidate mode, and against what we REQUIRE.
function score(bits) {
  let rte = 0, rtz = 0, checked = 0, wrong = [], mismatchedHalves = 0;
  for (let i = 0; i < inputs.length; i++) {
    if (note[i] === "(pad)") continue;
    const x = inputs[i];
    if (bits[i] & 0x10000) mismatchedHalves++;
    const g = bits[i] & 0xffff;
    const r = modeRTE(x), z = modeRTZ(x), w = want(x);
    checked++;
    if (g === r) rte++;
    if (g === z) rtz++;
    if (w !== null && g !== w) wrong.push({ note: note[i], x, g, w });
  }
  return { rte, rtz, checked, wrong, mismatchedHalves };
}

const results = [];
for (const [label, src, shippedArm] of [
  ["pack2x16float  (the OLD store instruction)", BUILTIN_PACK, false],
  ["native f16()   conversion", BUILTIN_NATIVE, false],
  ["SHIPPED gxp_hpk - control arm (pack)", shipped("pack"), true],
  ["SHIPPED gxp_hpk - PORTABLE arm", shipped("portable"), true],
  ["SHIPPED gxp_hpk - NATIVE arm", shipped("native"), true],
]) {
  const got = await runArm(src);
  results.push({ label, got, shippedArm });
}

const adapterName = results.find((r) => r.got.adapter)?.got.adapter ?? "(unknown)";
const hasF16 = results.find((r) => r.got.hasF16 !== undefined)?.got.hasF16;
console.log(`adapter: ${adapterName}`);
console.log(`shader-f16: ${hasF16 ? "available" : "NOT available"}`);
console.log("");
console.log("  store instruction                            RTE      RTZ     what we require");
let failed = 0;
for (const { label, got, shippedArm } of results) {
  if (got.error) {
    const skippable = !hasF16 && label.includes("NATIVE") ;
    console.log(`  ${label.padEnd(44)} ${skippable ? "(skipped: no shader-f16)" : `FAILED: ${got.error}`}`);
    if (!skippable) failed++;
    continue;
  }
  const s = score(got.bits);
  const verdict = shippedArm
    ? (label.includes("control") ? "(control: the old mode)" : s.wrong.length === 0 ? "PASS" : `FAIL (${s.wrong.length})`)
    : "";
  console.log(
    `  ${label.padEnd(44)} ${String(s.rte + "/" + s.checked).padStart(7)}  ${String(s.rtz + "/" + s.checked).padStart(7)}  ${verdict}`,
  );
  if (shippedArm && !label.includes("control")) {
    if (s.wrong.length) {
      failed++;
      for (const d of s.wrong.slice(0, 12)) {
        console.log(
          `      ${d.note.padEnd(22)} x=${String(d.x.toPrecision(9)).padStart(14)}  device=0x${d.g.toString(16).padStart(4, "0")}  required=0x${d.w.toString(16).padStart(4, "0")}`,
        );
      }
    }
    if (s.mismatchedHalves) {
      failed++;
      console.log(`      the two halves of the PAIR store disagree on ${s.mismatchedHalves} inputs`);
    }
  }
}

// >>> AND THE TWO SHIPPED ROUNDING ARMS MUST AGREE BIT FOR BIT. Which one a run uses is chosen
// from the device's features, so a difference between them is a difference between two phones.
const portable = results.find((r) => r.label.includes("PORTABLE"))?.got;
const native = results.find((r) => r.label.includes("NATIVE"))?.got;
console.log("");
if (portable?.bits && native?.bits) {
  const differ = portable.bits.filter((b, i) => b !== native.bits[i]).length;
  console.log(
    differ === 0
      ? "the PORTABLE and NATIVE arms agree on every input - which is what lets the device choose"
      : `ARMS DISAGREE on ${differ} inputs - a title would render differently on two devices`,
  );
  if (differ) {
    failed++;
    for (let i = 0; i < inputs.length; i++) {
      if (portable.bits[i] === native.bits[i] || note[i] === "(pad)") continue;
      console.log(
        `  ${note[i].padEnd(22)} x=${String(inputs[i].toPrecision(9)).padStart(14)}  portable=0x${(portable.bits[i] & 0xffff).toString(16)}  native=0x${(native.bits[i] & 0xffff).toString(16)}`,
      );
    }
  }
} else if (!hasF16) {
  console.log("the NATIVE arm was not measured (no shader-f16 here) - the portable arm is what ships");
} else {
  failed++;
  console.log("one of the two shipped arms did not run, so they were NOT compared");
}

await browser.close();
server.close();
process.exit(failed ? 1 : 0);
