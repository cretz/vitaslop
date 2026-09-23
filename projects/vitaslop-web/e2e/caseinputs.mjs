// caseinputs.mjs - THE SEEDED CASE INPUTS, IN ONE PLACE.
//
// >>> WHY THIS IS A FILE AND NOT A COPY IN EACH RUNNER. The Rust side's note says it already:
// `lane_value` must have exactly ONE copy, because the runner REGENERATES the inputs in the
// page from the case's seed rather than shipping them, and a second generator that drifted
// would feed the GPU different numbers and report it as a translation defect. The same
// argument applies on this side the moment there is more than one runner - and there are three
// (`gxpexec` grades an arm against the CPU reference, `f16arms` compares two arms against each
// other, `gxptrace` locates a divergence), so the generator moved here.
//
// >>> AND IT IS INJECTED WITH `addInitScript`, NOT BUILT WITH `new Function`.
//
// The page ships a strict Content-Security-Policy (`script-src 'self' 'wasm-unsafe-eval'`), so
// `new Function` is an EvalError there:
//
//   Evaluating a string as JavaScript violates the following Content Security Policy directive
//   because 'unsafe-eval' is not an allowed source of script
//
// Playwright's `addInitScript` installs the source as a script before the page's own code runs,
// which is not eval and is not subject to that directive. It also means an in-page callback
// simply USES `globalThis.__laneValue` instead of rebuilding it per batch.
export const LANE_VALUE_SRC = `
// The SAME avalanche \`execcases.rs::lane_bits\` uses. \`Math.imul\` is the only 32-bit multiply
// in JS that matches Rust's wrapping one; \`>>> 0\` keeps every step unsigned.
globalThis.__laneBits = (seed, lane) => {
  let h = (seed ^ Math.imul(lane, 0x9e3779b9)) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  h = Math.imul(h, 0x7feb352d) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  h = Math.imul(h, 0x846ca68b) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h;
};
// ... and the same mapping onto [-4, 4). Computed in f64 then stored into a Float32Array,
// which rounds once - exactly as the Rust side's f32 arithmetic does.
globalThis.__laneValue = (seed, lane) => ((globalThis.__laneBits(seed, lane) >>> 8) / 16777216) * 8 - 4;

// >>> AND THE PER-PROGRAM OVERRIDES, WHICH ARE DATA AND NOT A GENERATOR.
//
// A lane a program reads as a COUNT, an INDEX or a POINTER cannot hold a float in [-4, 4): read
// as an integer that is 3,229,614,080, and an address computed from it misses every bound window
// while a loop bounded by it never ends. The Rust case writer decides which lanes those are by a
// dataflow walk over the IR and CHOOSES their values by running the reference - neither of which
// this side can reproduce, and neither of which it should try to: the whole point of the case
// format is that there is one generator and no second copy to drift.
//
// So the chosen lanes travel in the case as \`inputs\`, a sparse \`[lane, bits]\` list over the flat
// \`pa\`-then-\`sa\` numbering, and this applies them on top of the seeded fill. A case without the
// field is a program that needed none, and is filled exactly as it always was.
//
// The bits are written through a Uint32 VIEW of the same buffer, because they are a bit pattern
// and not a number: an integer index stored as an f32 would be a denormal, and storing it as a
// float would round it to something else again.
globalThis.__applyCaseInputs = (c, floatInput) => {
  if (!c.inputs || c.inputs.length === 0) return;
  const bits = new Uint32Array(floatInput.buffer, floatInput.byteOffset, floatInput.length);
  for (const [lane, v] of c.inputs) bits[lane] = v >>> 0;
};
`;

/// Install the generator into `page` so the in-page callbacks can use `globalThis.__laneValue`.
///
/// Called BEFORE `page.goto`, because an init script runs at document creation.
export async function installLaneValue(page) {
  await page.addInitScript({ content: LANE_VALUE_SRC });
}

/// The same generator for NODE-side use, derived from the same source rather than written twice:
/// a runner that ships a handful of cases (rather than thousands) can build the input array here
/// and hand it over, which is cheaper than shipping a generator and simpler than regenerating.
const nodeScope = {};
new Function("globalThis", LANE_VALUE_SRC).call(null, nodeScope);
export const laneBits = nodeScope.__laneBits;
export const laneValue = nodeScope.__laneValue;
export const applyCaseInputs = nodeScope.__applyCaseInputs;
