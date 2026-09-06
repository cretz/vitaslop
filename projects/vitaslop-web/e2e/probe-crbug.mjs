// The Chromium tracker is a JS app: a plain fetch gets a sign-in shell, a real browser gets the
// issue. Read it with the browser that is already here.
import { chromium } from "playwright";
const b = await chromium.launch({ channel: "chrome", headless: true });
const p = await b.newPage();
await p.goto(process.env.URL, { waitUntil: "networkidle", timeout: 90000 }).catch(() => {});
await p.waitForTimeout(5000);
const text = await p.evaluate(() => document.body.innerText);
console.log(text.slice(0, Number(process.env.MAX || 4000)));
await b.close();
