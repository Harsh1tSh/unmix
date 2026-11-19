import { expect, test, type Page } from "@playwright/test";
import { makeTestWav } from "./fixtures";

/** Subset of StemEngine reachable through window.__unmix (see StemBoard.tsx). */
interface EngineHandle {
  getSnapshot(): { status: string; playing: boolean; preview: boolean; hover: string | null; duration: number; gains: Record<string, number> };
  position(): number;
  outputLevel(): number;
  liveGains(): Record<string, number>;
  contextState(): string;
}

const engineState = (page: Page) =>
  page.evaluate(() => {
    const e = (window as unknown as { __unmix: EngineHandle }).__unmix;
    const s = e.getSnapshot();
    return { ...s, position: e.position(), level: e.outputLevel(), live: e.liveGains(), ctx: e.contextState() };
  });

async function uploadAndWaitForBoard(page: Page) {
  await page.goto("/");
  await expect(page.getByTestId("health")).toHaveAttribute("data-status", "ok");
  await page.getByTestId("file-input").setInputFiles({ name: "test-song.wav", mimeType: "audio/wav", buffer: makeTestWav(4) });

  // Progress screen with real stage names streamed over SSE.
  await expect(page.getByTestId("progress-screen")).toBeVisible();
  await expect(page.getByTestId("progress-screen")).toContainText("test-song.wav");
  const stages = page.getByTestId("stage-list");
  await expect(stages.locator('[data-state="active"]')).toHaveCount(1);
  const seen = new Set<string>();
  for (let i = 0; i < 100 && !(await page.getByTestId("stem-board").isVisible()); i++) {
    const active = await stages.locator('[data-state="active"]').getAttribute("data-stage", { timeout: 500 }).catch(() => null);
    if (active) seen.add(active);
    await page.waitForTimeout(100);
  }
  expect([...seen].some((s) => s === "voices" || s === "music"), `stages seen: ${[...seen].join(",")}`).toBe(true);

  await expect(page.getByTestId("stem-board")).toBeVisible();
  await expect(page.getByTestId("stem-grid")).toHaveAttribute("data-ready", "true");
}

test("upload -> progress -> cards render -> hover plays that stem alone", async ({ page }) => {
  await uploadAndWaitForBoard(page);

  // All seven cards, in tree order, each with its visuals.
  for (const name of ["voices", "music", "drums", "bass", "guitar", "piano", "other"]) {
    const card = page.getByTestId(`stem-card-${name}`);
    await expect(card).toBeVisible();
    await expect(card.getByTestId("waveform")).toBeVisible();
    await expect(card.getByTestId("band-meter")).toBeVisible();
    await expect(card.getByTestId("energy-share")).toHaveText(/%$/);
    const img = card.locator("img");
    await expect(img).toHaveJSProperty("complete", true);
    expect(await img.evaluate((el: HTMLImageElement) => el.naturalWidth)).toBeGreaterThan(10);
  }
  await expect(page).toHaveURL(/#job=[0-9a-f]{32}$/);

  // Nothing plays until the user does something.
  let s = await engineState(page);
  expect(s.playing).toBe(false);
  expect(Object.keys(s.gains).sort()).toEqual(["bass", "drums", "guitar", "other", "piano", "voices"]);

  // HOVER = LISTEN: hovering the bass card starts playback with only bass audible.
  await page.getByTestId("stem-card-bass").hover();
  await expect.poll(async () => (await engineState(page)).playing).toBe(true);
  await page.waitForTimeout(500); // > 150 ms crossfade
  s = await engineState(page);
  expect(s.ctx).toBe("running");
  expect(s.hover).toBe("bass");
  expect(s.preview).toBe(true);
  expect(s.live.bass).toBeCloseTo(1, 2);
  for (const other of ["voices", "drums", "guitar", "piano", "other"]) expect(s.live[other], other).toBeCloseTo(0, 2);
  expect(s.level, "audio must actually reach the output").toBeGreaterThan(0.01);
  const p1 = s.position;
  await page.waitForTimeout(400);
  expect((await engineState(page)).position).toBeGreaterThan(p1 + 0.2);
  await expect(page.getByTestId("stem-card-bass")).toHaveAttribute("data-hovered", "true");
  await expect(page.getByTestId("stem-card-voices")).toHaveAttribute("data-audible", "false");

  // Hovering the Music parent plays all of its children, but not the voices.
  await page.getByTestId("stem-card-music").locator("h3").hover();
  await page.waitForTimeout(400);
  s = await engineState(page);
  expect(s.hover).toBe("music");
  expect(s.live.voices).toBeCloseTo(0, 2);
  for (const child of ["drums", "bass", "guitar", "piano", "other"]) expect(s.live[child], child).toBeCloseTo(1, 2);

  // Leaving the cards fades back to the full mix and the preview stops.
  await page.mouse.move(2, 2);
  await expect.poll(async () => (await engineState(page)).playing).toBe(false);
  s = await engineState(page);
  expect(s.hover).toBeNull();
  expect(Object.values(s.gains).every((g) => g === 1)).toBe(true);
});

test("transport, solo, mute and seek keep every stem on one clock", async ({ page }) => {
  await uploadAndWaitForBoard(page);
  await page.mouse.move(2, 2);

  await page.getByTestId("play-toggle").click();
  await expect.poll(async () => (await engineState(page)).playing).toBe(true);
  let s = await engineState(page);
  expect(s.preview).toBe(false);
  expect(Object.values(s.gains).every((g) => g === 1)).toBe(true);
  await expect(page.getByTestId("time-total")).toHaveText("0:04");

  // Solo drums -> only drums; mute is applied on top of solo.
  await page.getByTestId("stem-card-drums").getByTestId("solo").click();
  await page.mouse.move(2, 2);
  s = await engineState(page);
  expect(s.gains).toMatchObject({ drums: 1, bass: 0, voices: 0, guitar: 0, piano: 0, other: 0 });
  await page.getByTestId("stem-card-drums").getByTestId("solo").click();
  await page.getByTestId("stem-card-voices").getByTestId("mute").click();
  await page.mouse.move(2, 2);
  s = await engineState(page);
  expect(s.gains).toMatchObject({ voices: 0, drums: 1, bass: 1 });

  // Seeking while playing restarts all sources together at the new offset.
  await page.evaluate(() => (window as unknown as { __unmix: { seek(t: number): void } }).__unmix.seek(2.5));
  await page.waitForTimeout(250);
  s = await engineState(page);
  expect(s.playing).toBe(true);
  expect(s.position).toBeGreaterThan(2.5);
  expect(s.position).toBeLessThan(3.4);

  await page.getByTestId("play-toggle").click();
  await expect.poll(async () => (await engineState(page)).playing).toBe(false);
  const paused = (await engineState(page)).position;
  await page.waitForTimeout(300);
  expect((await engineState(page)).position).toBe(paused);

  // Collapsing the Music card hides the instrument cards.
  await page.getByTestId("expand-music").click();
  await expect(page.getByTestId("stem-card-drums")).toHaveCount(0);
  await page.getByTestId("expand-music").click();
  await expect(page.getByTestId("stem-card-drums")).toBeVisible();
});

test("a finished job is restored from the URL after a reload", async ({ page }) => {
  await uploadAndWaitForBoard(page);
  await page.reload();
  await expect(page.getByTestId("stem-board")).toBeVisible();
  await expect(page.getByTestId("stem-grid")).toHaveAttribute("data-ready", "true");
  await expect(page.getByTestId("stem-card-voices")).toBeVisible();
});

test("a progress stream that is held back (tunnel) falls back to polling", async ({ page }) => {
  // The Cloudflare quick tunnel used for a Colab backend can buffer SSE.
  // Simulate a stream that never delivers anything: the job must still finish.
  await page.route("**/api/jobs/*/events", () => {
    /* never answered */
  });
  await page.goto("/");
  await expect(page.getByTestId("health")).toHaveAttribute("data-status", "ok");
  await expect(page.getByTestId("health")).toContainText("Ready · CPU");
  await page.getByTestId("file-input").setInputFiles({ name: "held-back.wav", mimeType: "audio/wav", buffer: makeTestWav(3) });
  await expect(page.getByTestId("progress-screen")).toBeVisible();
  await expect(page.getByTestId("stem-grid")).toHaveAttribute("data-ready", "true", { timeout: 45_000 });
});

test("an undecodable file shows the server's error message", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("file-input").setInputFiles({ name: "broken.wav", mimeType: "audio/wav", buffer: Buffer.from("not audio at all ".repeat(200)) });
  await expect(page.getByTestId("upload-error")).toContainText(/not look like a supported audio file/i);
  await expect(page.getByTestId("dropzone")).toBeVisible();
});

test("layout holds at 375 px wide", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 740 });
  await uploadAndWaitForBoard(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow, "no horizontal scrolling").toBeLessThanOrEqual(0);
  await expect(page.getByTestId("play-toggle")).toBeInViewport();
});
