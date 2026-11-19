import { defineConfig } from "vitest/config";

// Unit tests cover pure logic only (no DOM), so none of the app's Vite plugins are needed.
export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
    environment: "node",
  },
});
