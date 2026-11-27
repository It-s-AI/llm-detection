// Render humanizer/docs/overview/index.html to overview.pdf: one tall page, exactly as the browser shows it
// (light theme, screen styles, fonts loaded). Usage: node render_pdf.mjs <index.html> <out.pdf>
import { chromium } from "playwright";
import { pathToFileURL } from "node:url";
const [src, out] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 1 });
await page.emulateMedia({ media: "screen", colorScheme: "light" });
await page.goto(pathToFileURL(src).href, { waitUntil: "networkidle" });
await page.evaluate(() => document.documentElement.setAttribute("data-theme", "light"));
await page.evaluate(() => document.fonts.ready);
// reveal anything that animates in on scroll, then measure
await page.addStyleTag({ content: "*{animation:none!important;transition:none!important} .reveal,[data-reveal]{opacity:1!important;transform:none!important}" });
const height = await page.evaluate(() => Math.ceil(document.documentElement.scrollHeight));
await page.pdf({ path: out, width: "1200px", height: `${height + 2}px`, printBackground: true,
                 margin: { top: "0", right: "0", bottom: "0", left: "0" }, pageRanges: "1" });
await browser.close();
console.log(`${out}: 1 page, 1200 x ${height} px`);
