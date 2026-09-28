// Which JPEG decoder is faster on THIS device: the one this build carries, or the
// browser's own.
//
// SceJpeg is a synchronous guest call and every browser JPEG entry point is a promise -
// but that does not settle anything, because a host call CAN wait on a promise here: it is
// what every blocking kernel primitive already does through JSPI. So the browser's decoder
// is reachable, and the question is the ordinary one, which is faster END TO END.
//
// "End to end" is the whole point of this file. `createImageBitmap` alone hands back an
// opaque bitmap; the guest needs PIXELS in its own memory, so the browser path has to draw
// that bitmap to an OffscreenCanvas and read it back. That readback is often the larger
// half, and comparing a decode against a decode would flatter it. Three numbers are timed:
//
//   ours          this build's decoder, through the same functions `vita::jpeg` calls
//   bitmap        createImageBitmap alone - the decode, without the pixels
//   bitmap+read   createImageBitmap, drawn to a canvas, read back as RGBA - the real path
//
// In:  { bytes, iterations }   the JPEG's bytes and how many times to decode
// Out: { type: "log", line } ... { type: "done" }

import init, { jpeg_bench } from "../pkg/vitaslop_web.js";

const say = (line) => self.postMessage({ type: "log", line });

/// Time `f` over `n` runs, in milliseconds per run. One warm run first: the first touches
/// cold code and cold GPU/driver paths and would otherwise be most of a small average.
async function timeAsync(n, f) {
  await f();
  const t0 = performance.now();
  for (let i = 0; i < n; i++) await f();
  return (performance.now() - t0) / n;
}

self.onmessage = async (e) => {
  const { bytes, iterations } = e.data;
  const n = iterations || 20;
  try {
    await init();
  } catch (err) {
    say(`wasm init failed: ${err}`);
    self.postMessage({ type: "done" });
    return;
  }

  say(`image: ${bytes.byteLength} bytes, ${n} decodes per arm`);

  // --- ours -------------------------------------------------------------------------
  const r = jpeg_bench(new Uint8Array(bytes), n);
  if (r.error) {
    say(`ours: ${r.error}`);
    self.postMessage({ type: "done" });
    return;
  }
  const px = r.width * r.height;
  say(`image is ${r.width}x${r.height} = ${(px / 1e6).toFixed(2)} Mpixel`);
  say(`ours          ${r.msPerDecode.toFixed(2)} ms  (${r.megapixelsPerSecond.toFixed(1)} Mpixel/s)`);

  // --- the browser's own ------------------------------------------------------------
  const blob = new Blob([bytes], { type: "image/jpeg" });
  let bitmapMs = 0;
  try {
    bitmapMs = await timeAsync(n, async () => {
      const bm = await createImageBitmap(blob);
      bm.close();
    });
    say(`bitmap        ${bitmapMs.toFixed(2)} ms  (${(px / (bitmapMs / 1000) / 1e6).toFixed(1)} Mpixel/s) - decode only, no pixels`);
  } catch (err) {
    say(`bitmap        unavailable: ${err}`);
  }

  // The path the guest would actually need: pixels back in memory.
  try {
    const canvas = new OffscreenCanvas(r.width, r.height);
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const readMs = await timeAsync(n, async () => {
      const bm = await createImageBitmap(blob);
      ctx.drawImage(bm, 0, 0);
      bm.close();
      // Touch the data so nothing here can be elided.
      const d = ctx.getImageData(0, 0, r.width, r.height);
      if (d.data[0] === 12345 && d.data[1] === 54321) say("(unreachable)");
    });
    say(`bitmap+read   ${readMs.toFixed(2)} ms  (${(px / (readMs / 1000) / 1e6).toFixed(1)} Mpixel/s) - THIS is the comparable number`);
    const verdict =
      readMs < r.msPerDecode
        ? `the browser is ${(r.msPerDecode / readMs).toFixed(2)}x faster end to end`
        : `ours is ${(readMs / r.msPerDecode).toFixed(2)}x faster end to end`;
    say(`verdict: ${verdict}`);
  } catch (err) {
    say(`bitmap+read   unavailable: ${err}`);
  }

  self.postMessage({ type: "done" });
};
