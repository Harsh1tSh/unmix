import { defineConfig, devices } from "@playwright/test";
import os from "node:os";
import path from "node:path";

/**
 * End-to-end tests run against a real backend process using the FAKE models
 * (UNMIX_FAKE_MODELS=1: DSP filters, no weights, finishes in ~2 s) so they
 * exercise the whole upload -> SSE -> stems -> Web Audio path quickly.
 * Ports differ from the dev defaults so a running dev setup is not disturbed.
 */
const BACKEND_PORT = 8010;
const FRONTEND_PORT = 5183;
const dataDir = path.join(os.tmpdir(), "unmix-e2e-data");

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${FRONTEND_PORT}`,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          // Let the AudioContext start without a click, so "hover plays audio" is testable.
          args: [
            "--autoplay-policy=no-user-gesture-required",
            // A fake microphone (a test tone) with auto-granted permission, for the orb tests.
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
          ],
        },
      },
    },
  ],
  webServer: [
    {
      command: `uv run uvicorn app.server:app --host 127.0.0.1 --port ${BACKEND_PORT}`,
      cwd: "../backend",
      url: `http://127.0.0.1:${BACKEND_PORT}/api/health`,
      reuseExistingServer: false,
      timeout: 120_000,
      // Fake separation models, but the real model folder: the browser-only test
      // runs the real Kim Vocal 2 ONNX file in the browser (skipped if absent).
      env: { UNMIX_FAKE_MODELS: "1", UNMIX_FAKE_DELAY_S: "2", UNMIX_DATA_DIR: dataDir, UNMIX_MODELS_DIR: path.resolve("../backend/data/models") },
    },
    {
      command: `npm run dev -- --host 127.0.0.1 --port ${FRONTEND_PORT} --strictPort`,
      url: `http://127.0.0.1:${FRONTEND_PORT}`,
      reuseExistingServer: false,
      timeout: 60_000,
      env: { UNMIX_BACKEND_URL: `http://127.0.0.1:${BACKEND_PORT}` },
    },
  ],
});
