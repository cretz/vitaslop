import { chromium } from "playwright";
const b = await chromium.launch({ channel: "chrome", headless: true });
const p = await b.newPage();
await p.goto(process.env.URL, { waitUntil: "networkidle", timeout: 90000 }).catch(() => {});
await p.waitForTimeout(6000);
const text = await p.evaluate(() => document.body.innerText);
// Print the tail: the comments, where the resolution lives.
console.log(text.slice(-Number(process.env.TAIL || 5000)));
await b.close();
