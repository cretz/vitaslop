import { chromium } from "playwright";
const b = await chromium.launch({ channel: "chrome", headless: true });
const p = await b.newPage();
await p.goto(process.env.URL, { waitUntil: "networkidle", timeout: 90000 }).catch(() => {});
await p.waitForTimeout(6000);
const text = await p.evaluate(() => document.body.innerText);
const i = text.indexOf("COMMENTS") >= 0 ? text.indexOf("COMMENTS") : text.indexOf("#2");
console.log(text.slice(i > 0 ? i : 2000, (i > 0 ? i : 2000) + 6000));
await b.close();
