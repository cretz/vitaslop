// Does a CLICKED-THROUGH certificate cost you WebGPU?
//
// The earlier version of this test passed --ignore-certificate-errors, which makes Chrome treat
// the certificate as VALID and so removes the exact condition under test. `ignoreHTTPSErrors` at
// the context level is the honest equivalent of a user tapping through the interstitial: the
// page loads and the origin keeps its certificate error.
import { chromium } from "playwright";
const url = process.env.URL;
const bypassProperly = !!process.env.IGNORE_CERT_FLAG;
const browser = await chromium.launch({
  channel: "chrome",
  headless: true,
  args: bypassProperly ? ["--ignore-certificate-errors"] : [],
});
const ctx = await browser.newContext({ ignoreHTTPSErrors: true });
const page = await ctx.newPage();
console.log("chrome:", browser.version(), bypassProperly ? "(cert forced VALID)" : "(cert error BYPASSED, as a user does)");
await page.goto(url, { waitUntil: "load", timeout: 60000 });
console.log(JSON.stringify(await page.evaluate(async () => {
  const out = { isSecureContext, crossOriginIsolated, hasGpu: !!navigator.gpu };
  if (navigator.gpu) {
    const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    out.adapterNull = a === null;
    if (a) out.vendor = a.info?.vendor;
  }
  return out;
})));
await browser.close();
