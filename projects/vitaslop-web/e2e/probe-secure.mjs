// Does the origin the phone's flag whitelists actually get a SECURE, cross-origin-isolated
// context - i.e. WebGPU and SharedArrayBuffer, which is what audio needs?
//
// Launched with the SAME flag the device carries, so this answers the device's question and
// not this machine's.
import { chromium } from "playwright";
const url = process.env.URL;
const origin = new URL(url).origin;
const browser = await chromium.launch({
  channel: "chrome",
  headless: true,
  args: [`--unsafely-treat-insecure-origin-as-secure=${origin}`, "--disable-features=IsolateOrigins"],
});
const page = await browser.newPage();
await page.goto(url, { waitUntil: "load", timeout: 60000 });
console.log(url, JSON.stringify(await page.evaluate(async () => {
  const out = {
    isSecureContext,
    crossOriginIsolated,
    hasSharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    hasGpu: !!navigator.gpu,
  };
  if (navigator.gpu) {
    const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    out.adapterNull = a === null;
    if (a) out.vendor = a.info?.vendor;
  }
  return out;
})));
await browser.close();
