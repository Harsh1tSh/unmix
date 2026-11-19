import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

// The backend URL only matters to the dev/preview proxy: app code always calls
// relative /api/... paths (see src/api/client.ts).
const backend = process.env.UNMIX_BACKEND_URL ?? "http://127.0.0.1:8000";
// Remote GPU mode (start.ps1 -Remote): the access key is added here, by the
// local proxy, so it never reaches the browser (or any page script).
const backendKey = process.env.UNMIX_BACKEND_KEY ?? "";

const proxy = {
  "/api": {
    target: backend,
    changeOrigin: true,
    secure: true, // verify the remote's TLS certificate
    ws: true, // live mode streams over a WebSocket at /api/live
    ...(backendKey ? { headers: { "X-Unmix-Key": backendKey } } : {}),
  },
};

/**
 * Content-Security-Policy for the production build (`npm run build` / preview).
 * Not used in dev: Vite's dev server injects inline scripts for hot reload.
 * Everything is same-origin: no CDN, no fonts from the network, no inline
 * scripts. 'unsafe-inline' for styles is needed by React style props set from
 * markup and by the motion library; it does not allow script execution.
 */
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval'", // browser-only mode compiles onnxruntime's WebAssembly
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "media-src 'self' blob:",
  "font-src 'self'",
  "connect-src 'self' blob:", // blob: = stems produced in the browser
  "worker-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
].join("; ");

function contentSecurityPolicy(): Plugin {
  return {
    name: "unmix-csp",
    apply: "build",
    transformIndexHtml: () => [{ tag: "meta", attrs: { "http-equiv": "Content-Security-Policy", content: CSP }, injectTo: "head-prepend" }],
  };
}

// Cross-origin isolation: lets onnxruntime-web use SharedArrayBuffer threads
// in the browser-only mode. Everything the app loads is same-origin anyway.
const isolation = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };

export default defineConfig({
  plugins: [react(), tailwindcss(), contentSecurityPolicy()],
  server: { port: 5173, strictPort: false, proxy, headers: isolation },
  preview: { port: 4173, proxy, headers: isolation },
  worker: { format: "es" },
  // onnxruntime-web loads its own .mjs/.wasm at runtime; letting Vite's dependency
  // optimiser discover it mid-session triggers a full page reload in dev (the
  // first browser-only job would silently restart). Exclude it, as ORT advises.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
