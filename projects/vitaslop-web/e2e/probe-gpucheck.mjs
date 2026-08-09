import { chromium } from "playwright";
const b = await chromium.launch({ channel: "chrome", headless: true });
const c = await b.newContext({ ignoreHTTPSErrors: true });
const p = await c.newPage();
await p.goto(process.env.URL, { waitUntil: "load", timeout: 60000 });
await p.waitForTimeout(8000);
console.log(await p.innerText("#out"));
await b.close();
