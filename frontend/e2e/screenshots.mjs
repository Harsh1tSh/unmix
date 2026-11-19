// Dev helper (not a test): capture screenshots of a running dev setup.
//
//   node e2e/screenshots.mjs <out-dir> [job-id]
//
// Needs the frontend on http://127.0.0.1:5173 and, for the board, a finished job id.
import { chromium } from "@playwright/test";
import path from "node:path";

const [outDir = ".", jobId] = process.argv.slice(2);
const base = process.env.UNMIX_URL ?? "http://127.0.0.1:5173";

const browser = await chromium.launch({ args: ["--autoplay-policy=no-user-gesture-required"] });
for (const [label, viewport] of [
  ["desktop", { width: 1440, height: 900 }],
  ["mobile", { width: 375, height: 740 }],
]) {
  const page = await browser.newPage({ viewport, deviceScaleFactor: label === "mobile" ? 2 : 1 });
  await page.goto(base);
  await page.waitForSelector('[data-testid="health"]');
  await page.waitForTimeout(800);
  await page.screenshot({ path: path.join(outDir, `landing-${label}.png`) });

  if (jobId) {
    await page.goto(`${base}/#job=${jobId}`);
    await page.reload();
    await page.waitForSelector('[data-testid="stem-grid"][data-ready="true"]', { timeout: 120_000 });
    await page.evaluate(() => window.__unmix.seek(window.__unmix.getSnapshot().duration * 0.4));
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(outDir, `board-${label}.png`), fullPage: true });
    if (label === "desktop") {
      await page.locator('[data-testid="stem-card-bass"]').hover();
      await page.waitForTimeout(900);
      await page.screenshot({ path: path.join(outDir, "board-hover-bass.png") });
    }
  }
  await page.close();
}
await browser.close();
console.log("screenshots written to", path.resolve(outDir));
