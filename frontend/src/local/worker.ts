/**
 * Browser-only separation worker: Kim Vocal 2 on onnxruntime-web (WASM, SIMD,
 * multi-threaded when the page is cross-origin isolated). Runs off the main
 * thread so the UI stays at 60 fps. The audio never leaves this worker.
 */
import * as ort from "onnxruntime-web/wasm";
import ortMjs from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import ortWasm from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import { buildStemAnalysis, extractFeatures, spectrogramPng, type Features } from "./analysis";
import { blockCount, DIM_F, DIM_T, fromModelOutput, GEN, modelInput, TRIM, windowFor } from "./mdx";

export interface RunRequest {
  type: "run";
  left: Float32Array<ArrayBuffer>;
  right: Float32Array<ArrayBuffer>;
  modelUrl: string;
  colors: Record<string, string>;
}

export type WorkerMessage =
  | { type: "progress"; stage: "model" | "voices" | "analysis"; progress: number; message: string }
  | { type: "done"; stems: Record<string, [Float32Array<ArrayBuffer>, Float32Array<ArrayBuffer>]>; analysis: Record<string, unknown>; spectrograms: Record<string, Blob>; seconds: number; threads: number }
  | { type: "error"; message: string };

const post = (m: WorkerMessage, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

async function loadModel(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(res.status === 404 ? "The model is not on the server yet: run one normal job first (it downloads it)." : `Could not load the model (${res.status}).`);
  const total = Number(res.headers.get("content-length") ?? 0);
  if (!res.body || !total) return res.arrayBuffer();
  const reader = res.body.getReader();
  const buf = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, got);
    got += value.length;
    post({ type: "progress", stage: "model", progress: got / total, message: `Loading the model into the browser… ${Math.round((got / total) * 100)}%` });
  }
  return buf.buffer;
}

self.onmessage = async (e: MessageEvent<RunRequest>) => {
  if (e.data.type !== "run") return;
  const t0 = performance.now();
  try {
    const { left, right, modelUrl, colors } = e.data;
    ort.env.wasm.wasmPaths = { mjs: ortMjs, wasm: ortWasm };
    const isolated = (self as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
    const threads = isolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1)) : 1;
    ort.env.wasm.numThreads = threads;
    ort.env.wasm.simd = true;

    post({ type: "progress", stage: "model", progress: 0, message: "Loading the model into the browser…" });
    const model = await loadModel(modelUrl);
    const session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });

    const n = left.length;
    const voicesL = new Float32Array(n);
    const voicesR = new Float32Array(n);
    const blocks = blockCount(n);
    for (let b = 0; b < blocks; b++) {
      post({ type: "progress", stage: "voices", progress: b / blocks, message: `Separating voices from music · block ${b + 1} of ${blocks}` });
      const input = modelInput(windowFor(left, b), windowFor(right, b));
      const result = await session.run({ input: new ort.Tensor("float32", input, [1, 4, DIM_F, DIM_T]) });
      const [vl, vr] = fromModelOutput(result.output!.data as Float32Array);
      const start = b * GEN;
      const len = Math.min(GEN, n - start);
      voicesL.set(vl.subarray(TRIM, TRIM + len), start);
      voicesR.set(vr.subarray(TRIM, TRIM + len), start);
    }
    await session.release();

    // music is the residual: voices + music == mix, exactly, as on the server.
    const musicL = new Float32Array(n);
    const musicR = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      musicL[i] = left[i]! - voicesL[i]!;
      musicR[i] = right[i]! - voicesR[i]!;
    }

    post({ type: "progress", stage: "analysis", progress: 0, message: "Analysing frequency content" });
    const stems = { voices: [voicesL, voicesR], music: [musicL, musicR] } as const;
    const feats: Record<string, Features> = {};
    for (const [name, [l, r]] of Object.entries(stems)) feats[name] = extractFeatures(l, r);
    let mixEnergy = 0;
    for (let i = 0; i < n; i++) mixEnergy += left[i]! * left[i]! + right[i]! * right[i]!;
    const all = Object.values(feats);
    const bandRef = Math.max(1e-20, ...all.flatMap((f) => f.bandEnergy.map((b) => Math.max(...b))));
    const melRef = Math.max(1e-20, ...all.flatMap((f) => f.mel.map((m) => Math.max(...m))));
    const spectrograms: Record<string, Blob> = {};
    const analysis: Record<string, unknown> = {};
    for (const [name, f] of Object.entries(feats)) {
      spectrograms[name] = await spectrogramPng(f.mel, melRef, colors[name] ?? "#ffffff");
      analysis[name] = buildStemAnalysis(f, bandRef, mixEnergy, ""); // URL is filled in on the main thread
    }
    post(
      { type: "done", stems: { voices: [voicesL, voicesR], music: [musicL, musicR] }, analysis, spectrograms, seconds: (performance.now() - t0) / 1000, threads },
      [voicesL.buffer, voicesR.buffer, musicL.buffer, musicR.buffer],
    );
  } catch (err) {
    post({ type: "error", message: err instanceof Error ? err.message : "Separation in the browser failed." });
  }
};
