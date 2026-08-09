// Ask ONE question at a given origin: is WebGPU actually available there?
//
// `localhost` is a trusted secure origin in Chrome whatever its certificate says. A LAN IP
// with a self-signed certificate is not - the user clicks through a warning and the origin
// carries a certificate error, which is grounds for Chrome to withhold powerful features.
// Testing on localhost and reporting it as "the same URL" is how that difference stays
// invisible until a device hits it.
import { chromium } from "playwright";
const url = process.env.URL;
const browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--ignore-certificate-errors"] });
const ctx = await browser.newContext({ ignoreHTTPSErrors: true });
const page = await ctx.newPage();
await page.goto(url, { waitUntil: "load", timeout: 60000 });
const r = await page.evaluate(async () => {
  const out = { secureContext: isSecureContext, hasGpu: !!navigator.gpu };
  if (navigator.gpu) {
    try {
      const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      out.adapterNull = a === null;
      if (a) out.info = { vendor: a.info?.vendor, arch: a.info?.architecture };
    } catch (e) {
      out.requestAdapterThrew = String(e);
    }
  }
  return out;
});
console.log(url, JSON.stringify(r));
await browser.close();
