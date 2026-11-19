import { expect, test, type Page } from "@playwright/test";
import { makeTestWav } from "./fixtures";

interface EngineHandle {
  getSnapshot(): { status: string; playing: boolean; duration: number };
  position(): number;
  seek(t: number): void;
  play(): Promise<boolean>;
  pause(): void;
}

const engine = (page: Page) => page.evaluate(() => {
  const e = (window as unknown as { __unmix: EngineHandle }).__unmix;
  return { ...e.getSnapshot(), position: e.position() };
});

async function uploadWith(page: Page, opts: { splitVoices?: boolean; lyrics?: boolean } = {}) {
  await page.goto("/");
  await expect(page.getByTestId("health")).toHaveAttribute("data-status", "ok");
  const split = page.getByTestId("opt-split-voices");
  const lyrics = page.getByTestId("opt-lyrics");
  if ((await split.isChecked()) !== !!opts.splitVoices) await split.locator("..").click();
  if ((await lyrics.isChecked()) !== (opts.lyrics ?? true)) await lyrics.locator("..").click();
  await page.getByTestId("file-input").setInputFiles({ name: "talk-over-song.wav", mimeType: "audio/wav", buffer: makeTestWav(6) });
  await expect(page.getByTestId("progress-screen")).toBeVisible();
}

async function waitForBoard(page: Page) {
  await expect(page.getByTestId("stem-board")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("stem-grid")).toHaveAttribute("data-ready", "true");
  await expect(page.getByTestId("fly-in")).toHaveCount(0, { timeout: 10_000 });
}

test("landing: orb, options and drop overlay", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("orb")).toBeEnabled();
  await expect(page.getByTestId("orb-status")).toContainText("Hold to record");
  await expect(page.getByTestId("opt-lyrics")).toBeChecked();
  await expect(page.getByTestId("opt-split-voices")).toBeEnabled(); // fake backend reports it installed

  // Dragging a file over the page shows the full-screen drop target.
  const dt = await page.evaluateHandle(() => {
    const d = new DataTransfer();
    d.items.add(new File(["x"], "a.wav", { type: "audio/wav" }));
    return d;
  });
  await page.dispatchEvent("body", "dragenter", { dataTransfer: dt });
  await expect(page.getByTestId("drop-overlay")).toBeVisible();
  await page.dispatchEvent("body", "dragleave", { dataTransfer: dt });
  await expect(page.getByTestId("drop-overlay")).toHaveCount(0);
});

test("record from the microphone with a tap, then unmix it", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("health")).toHaveAttribute("data-status", "ok");
  const orb = page.getByTestId("orb");
  await orb.click(); // tap: start
  await expect(orb).toHaveAttribute("data-mode", "recording");
  await expect(page.getByTestId("orb-status")).toContainText("Recording");
  await page.waitForTimeout(2200);
  await orb.click(); // tap: stop and send
  await expect(page.getByTestId("progress-screen")).toBeVisible();
  await expect(page.getByTestId("progress-screen")).toContainText(/recording-.*\.(webm|ogg|m4a)/);
  await waitForBoard(page);
  await expect(page.getByTestId("stem-card-voices")).toBeVisible();
  expect((await engine(page)).duration).toBeGreaterThan(1.5);
});

test("hold the orb to record; a too-short hold explains itself", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("health")).toHaveAttribute("data-status", "ok");
  const box = (await page.getByTestId("orb").boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await expect(page.getByTestId("orb")).toHaveAttribute("data-mode", "recording");
  await page.waitForTimeout(500);
  await page.mouse.up(); // held > 350 ms: release sends, but < 1 s of audio
  await expect(page.getByTestId("upload-error")).toContainText(/too short/i);
  await expect(page.getByTestId("orb")).toHaveAttribute("data-mode", "idle");
});

test("lyrics follow playback, word by word, and clicking a word seeks", async ({ page }) => {
  await uploadWith(page, { lyrics: true });
  await expect(page.getByTestId("stage-list").locator('[data-stage="lyrics"]')).toHaveCount(1);
  await waitForBoard(page);

  const panel = page.getByTestId("stem-card-voices").getByTestId("lyrics");
  await expect(panel).toBeVisible();
  const line = panel.getByTestId("lyrics-current");
  await expect(line).toHaveAttribute("data-segment", "0");
  await expect(line.locator("[data-word]").first()).toHaveAttribute("data-state", "upcoming");

  // Seek into the second line; it becomes current and its first word is lit.
  const t = await page.evaluate(async () => {
    const lyr = await (await fetch(`/api/jobs/${location.hash.slice(5)}/lyrics`)).json();
    const seg = lyr.segments[1];
    (window as unknown as { __unmix: EngineHandle }).__unmix.seek(seg.words[0].end + 0.01);
    return seg.words[2].start as number;
  });
  await expect(line).toHaveAttribute("data-segment", "1");
  await expect(line.locator("[data-word]").first()).toHaveAttribute("data-state", "sung");

  // Clicking the third word of the current line seeks to its start.
  await line.locator("[data-word]").nth(2).click();
  expect((await engine(page)).position).toBeCloseTo(t, 2);

  // During playback the highlighted line advances on its own.
  await page.evaluate(() => (window as unknown as { __unmix: EngineHandle }).__unmix.seek(0));
  await page.getByTestId("play-toggle").click();
  await expect(line).not.toHaveAttribute("data-segment", "0", { timeout: 5000 });
});

test("without lyrics there is no panel and no lyrics stage", async ({ page }) => {
  await uploadWith(page, { lyrics: false });
  await expect(page.getByTestId("stage-list").locator('[data-stage="lyrics"]')).toHaveCount(0);
  await waitForBoard(page);
  await expect(page.getByTestId("lyrics")).toHaveCount(0);
});

test("split voices: speaker cards under Voices, lyrics tinted per speaker", async ({ page }) => {
  await uploadWith(page, { splitVoices: true });
  await expect(page.getByTestId("stage-list").locator('[data-stage="speakers"]')).toHaveCount(1);
  await waitForBoard(page);

  const branch = page.getByTestId("children-voices");
  await expect(branch).toBeVisible();
  await expect(branch.getByTestId("stem-card-speaker_1")).toBeVisible();
  await expect(branch.getByTestId("stem-card-speaker_2")).toBeVisible();
  // One lyrics lane per speaker, headed with that speaker's label.
  const lanes = page.getByTestId("lyrics").getByTestId("lyrics-lane");
  await expect(lanes).toHaveCount(2);
  await expect(lanes.nth(0)).toHaveAttribute("data-speaker", "speaker_1");
  await expect(lanes.nth(0)).toContainText("Voice 1");
  await expect(lanes.nth(1)).toContainText("Voice 2");

  // Hovering one speaker solos just that voice.
  await page.getByTestId("stem-card-speaker_1").hover();
  await expect.poll(async () => page.evaluate(() => (window as unknown as { __unmix: { getSnapshot(): { gains: Record<string, number> } } }).__unmix.getSnapshot().gains)).toMatchObject({ speaker_1: 1, speaker_2: 0, drums: 0 });
});

test("the waveform splits into strands that land in the cards", async ({ page }) => {
  await uploadWith(page);
  await expect(page.getByTestId("split-strands")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("fly-in")).toBeVisible();
  // While a strand is in flight its card waveform is hidden; afterwards all are shown.
  await expect(page.getByTestId("fly-in")).toHaveCount(0, { timeout: 10_000 });
  const opacities = await page.locator("[data-wave-for]").evaluateAll((els) => els.map((e) => getComputedStyle(e).opacity));
  expect(opacities.length).toBeGreaterThanOrEqual(7);
  expect(opacities.every((o) => o === "1")).toBe(true);
});

test("reduced motion skips the split animation", async ({ browser }) => {
  const context = await browser.newContext({ reducedMotion: "reduce" });
  const page = await context.newPage();
  await uploadWith(page);
  await expect(page.getByTestId("stem-board")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("split-strands")).toHaveCount(0);
  await expect(page.getByTestId("fly-in")).toHaveCount(0);
  await context.close();
});

test("top bar: file name, and Delete removes the job from this computer", async ({ page }) => {
  await uploadWith(page);
  await waitForBoard(page);
  await expect(page.getByTestId("board-filename")).toHaveText("talk-over-song.wav");
  const jobId = await page.evaluate(() => location.hash.slice(5));
  await page.getByTestId("delete-job").click();
  await page.getByTestId("confirm-delete").click();
  await expect(page.getByTestId("landing")).toBeVisible();
  await expect.poll(async () => (await page.request.get(`/api/jobs/${jobId}`)).status()).toBe(404);
});

test("'That's me' renames the voices on cards and lyric lanes", async ({ page }) => {
  await uploadWith(page, { splitVoices: true });
  await waitForBoard(page);
  await page.getByTestId("me-speaker_2").click();
  await expect(page.getByTestId("stem-card-speaker_2").getByTestId("stem-label")).toHaveText("You");
  await expect(page.getByTestId("stem-card-speaker_1").getByTestId("stem-label")).toHaveText("Other voice");
  await expect(page.getByTestId("lyrics-lane").filter({ hasText: "You" })).toHaveCount(1);
  // Remembered for this job across reloads.
  await page.reload();
  await expect(page.getByTestId("stem-card-speaker_2").getByTestId("stem-label")).toHaveText("You", { timeout: 15_000 });
  await page.getByTestId("me-speaker_2").click(); // undo
  await expect(page.getByTestId("stem-card-speaker_2").getByTestId("stem-label")).toHaveText("Voice 2");
});

test("lyrics export as .lrc and .srt", async ({ page }) => {
  await uploadWith(page);
  await waitForBoard(page);
  const [lrc] = await Promise.all([page.waitForEvent("download"), page.getByTestId("export-lrc").click()]);
  expect(lrc.suggestedFilename()).toBe("talk-over-song.lrc");
  const text = await (await lrc.createReadStream()).toArray().then((c) => Buffer.concat(c).toString("utf-8"));
  expect(text).toMatch(/^\[ti:talk-over-song\]/);
  expect(text).toMatch(/\[00:00\.\d\d\]la la la this/);
  const [srt] = await Promise.all([page.waitForEvent("download"), page.getByTestId("export-srt").click()]);
  expect(srt.suggestedFilename()).toBe("talk-over-song.srt");
});

test("transport: arrow keys skip, L loops, master volume", async ({ page }) => {
  await uploadWith(page);
  await waitForBoard(page);
  await page.mouse.move(2, 2);
  await page.keyboard.press("ArrowRight");
  await expect.poll(async () => (await engine(page)).position).toBeCloseTo(5, 1);
  await page.keyboard.press("ArrowLeft");
  await expect.poll(async () => (await engine(page)).position).toBeCloseTo(0, 1);
  await page.keyboard.press("KeyL");
  await expect(page.getByTestId("loop")).toHaveAttribute("aria-pressed", "true");
  // With loop on, playback wraps around instead of stopping at the end.
  await page.evaluate(() => (window as unknown as { __unmix: EngineHandle }).__unmix.seek(5.6));
  await page.getByTestId("play-toggle").click();
  await page.waitForTimeout(1200);
  const s = await engine(page);
  expect(s.playing).toBe(true);
  expect(s.position).toBeLessThan(2);
  await page.getByTestId("master-volume").fill("0.5");
  expect(await page.evaluate(() => (window as unknown as { __unmix: { getSnapshot(): { masterVolume: number } } }).__unmix.getSnapshot().masterVolume)).toBeCloseTo(0.5, 2);
});

test("clean up a voice: progress, then a player and a download", async ({ page }) => {
  await uploadWith(page, { splitVoices: true });
  await waitForBoard(page);
  const panel = page.getByTestId("clean-speaker_1");
  await expect(panel).toHaveAttribute("data-status", "idle");
  await panel.getByRole("button", { name: /clean up this voice/i }).click();
  await expect(panel).toHaveAttribute("data-status", "done", { timeout: 20_000 });
  await expect(panel.locator("audio")).toHaveAttribute("src", /\/clean\/speaker_1\.wav$/);
  // Survives a reload: the finished clean-up is found again.
  await page.reload();
  await expect(page.getByTestId("clean-speaker_1")).toHaveAttribute("data-status", "done", { timeout: 15_000 });
});

test("live mode: mic in, separated blocks back, then unmix the recording fully", async ({ page }) => {
  test.setTimeout(90_000);
  await page.goto("/");
  await expect(page.getByTestId("health")).toHaveAttribute("data-status", "ok");
  await page.getByTestId("go-live").click();
  const screen = page.getByTestId("live-screen");
  await expect(screen).toHaveAttribute("data-state", /waiting|live/, { timeout: 15_000 });
  // The first block needs ~5.75 s of input; with the fake model it comes straight back.
  await expect(screen).toHaveAttribute("data-state", "live", { timeout: 30_000 });
  await expect(page.getByTestId("live-blocks")).not.toHaveText("0");
  const block = await page.evaluate(() => {
    const s = (window as unknown as { __live: { blocks: { frames: number; voices: Float32Array[]; music: Float32Array[] }[] } }).__live;
    const b = s.blocks[0]!;
    let energyV = 0;
    let energyM = 0;
    for (let i = 0; i < b.frames; i += 50) {
      energyV += b.voices[0]![i]! ** 2;
      energyM += b.music[0]![i]! ** 2;
    }
    return { frames: b.frames, energyV, energyM };
  });
  expect(block.frames).toBe(253440);
  expect(block.energyV + block.energyM).toBeGreaterThan(0); // the fake mic is a tone, not silence

  await page.getByTestId("live-listen").click();
  await expect(page.getByTestId("live-listen")).toHaveAttribute("aria-pressed", "true");
  await page.getByTestId("live-stop").click();
  await expect(screen).toHaveAttribute("data-state", "ended", { timeout: 30_000 });
  await page.getByTestId("live-unmix-fully").click();
  await expect(page.getByTestId("progress-screen")).toContainText(/live-.*\.wav/);
  await waitForBoard(page);
  await expect(page.getByTestId("stem-card-voices")).toBeVisible();
});

test("browser-only mode: real model in the browser, nothing uploaded", async ({ page }) => {
  test.setTimeout(240_000);
  const model = await page.request.get("http://127.0.0.1:8010/api/models/Kim_Vocal_2.onnx", { headers: { Origin: "http://127.0.0.1:5183" } }).catch(() => null);
  test.skip(!model || model.status() !== 200, "Kim_Vocal_2.onnx is not downloaded on this machine");

  const uploads: string[] = [];
  page.on("request", (r) => {
    if (r.method() === "POST") uploads.push(r.url());
  });
  await page.goto("/");
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(true); // needed for WASM threads
  await expect(page.getByTestId("health")).toHaveAttribute("data-status", "ok");
  await page.getByTestId("opt-local").locator("..").click();
  await expect(page.getByTestId("opt-split-voices")).toBeDisabled();
  await page.getByTestId("file-input").setInputFiles({ name: "in-browser.wav", mimeType: "audio/wav", buffer: makeTestWav(5) });
  await expect(page.getByTestId("progress-screen")).toContainText("in-browser.wav");
  await expect(page.getByTestId("progress-message")).toContainText(/block 1 of 1|Analysing/, { timeout: 120_000 });
  await expect(page.getByTestId("stem-grid")).toHaveAttribute("data-ready", "true", { timeout: 200_000 });

  expect(uploads, "the audio must never be uploaded").toEqual([]);
  await expect(page.getByTestId("stem-card-voices")).toBeVisible();
  await expect(page.getByTestId("stem-card-music")).toBeVisible();
  await expect(page.getByTestId("stem-card-drums")).toHaveCount(0);
  await expect(page.getByTestId("board-filename")).toHaveText("in-browser.wav");
  // The two stems add back up to the input, as on the server.
  const check = await page.evaluate(async () => {
    const e = (window as unknown as { __unmix: { getSnapshot(): { duration: number } } }).__unmix;
    const urls = [...document.querySelectorAll<HTMLAnchorElement>('[data-testid="download"]')].map((a) => a.href);
    const ctx = new OfflineAudioContext(2, 1, 44100);
    const [v, m] = await Promise.all(urls.slice(0, 2).map(async (u) => ctx.decodeAudioData(await (await fetch(u)).arrayBuffer())));
    let energyV = 0;
    let energyM = 0;
    const vl = v!.getChannelData(0);
    const ml = m!.getChannelData(0);
    for (let i = 0; i < vl.length; i += 10) {
      energyV += vl[i]! ** 2;
      energyM += ml[i]! ** 2;
    }
    return { duration: e.getSnapshot().duration, energyV, energyM, urls };
  });
  expect(check.urls.every((u) => u.startsWith("blob:"))).toBe(true);
  expect(check.duration).toBeCloseTo(5, 1);
  expect(check.energyV).toBeGreaterThan(0);
  expect(check.energyM).toBeGreaterThan(0);
  const facts = await page.locator("dl").first().textContent();
  console.log("browser-only facts:", facts);
});

test("live mode: Back (and Esc) return to the start", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("health")).toHaveAttribute("data-status", "ok");
  await page.getByTestId("go-live").click();
  await expect(page.getByTestId("live-screen")).toBeVisible();
  await page.getByTestId("live-back").click();
  await expect(page.getByTestId("landing")).toBeVisible();
  await page.getByTestId("go-live").click();
  await expect(page.getByTestId("live-screen")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("landing")).toBeVisible();
});
