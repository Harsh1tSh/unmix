/**
 * Browser-only mode (phase 4): separate voices from music entirely in this
 * browser. The file is decoded here, separated in a Web Worker with
 * onnxruntime-web, and the result is shown on the normal stem board. Only the
 * model file (67 MB, cached by the browser) is fetched from the local server;
 * the audio is never uploaded.
 */
import type { Analysis, Job, JobEvent, Stem, StemAnalysis } from "../api/types";
import { encodeWav } from "../lib/wav";
import { FPS, SR } from "./analysis";
import type { RunRequest, WorkerMessage } from "./worker";

export const LOCAL_MODEL_URL = "/api/models/Kim_Vocal_2.onnx";
const MAX_SECONDS = 6 * 60;
const COLORS = { voices: "#FF6B9A", music: "#A78BFA" };
export const LOCAL_STAGES = [
  { name: "preprocess" as const, label: "Decoding in the browser" },
  { name: "voices" as const, label: "Separating voices from music (in this browser)" },
  { name: "analysis" as const, label: "Analysing frequency content" },
];

export interface LocalResult {
  job: Job;
  stems: Stem[];
  analysis: Analysis;
  /** revoke the blob URLs when the board is left */
  dispose: () => void;
}

async function decode(file: Blob): Promise<[Float32Array<ArrayBuffer>, Float32Array<ArrayBuffer>]> {
  const probe = new OfflineAudioContext(2, 1, SR);
  const decoded = await probe.decodeAudioData(await file.arrayBuffer());
  if (decoded.duration > MAX_SECONDS) throw new Error(`Audio is too long (limit ${MAX_SECONDS / 60} minutes).`);
  if (decoded.duration < 1) throw new Error("Audio is too short (minimum 1 s).");
  // Render through an OfflineAudioContext at 44.1 kHz stereo: resampling + up-mixing in one step.
  const frames = Math.ceil(decoded.duration * SR);
  const ctx = new OfflineAudioContext(2, frames, SR);
  const src = ctx.createBufferSource();
  src.buffer = decoded;
  src.connect(ctx.destination);
  src.start();
  const out = await ctx.startRendering();
  const l = out.getChannelData(0);
  const r = out.getChannelData(1);
  // Peak-normalise to -1 dBFS (the server uses a loudness target; this is the
  // simple equivalent that keeps quiet recordings in the model's range).
  let peak = 0;
  for (let i = 0; i < frames; i++) peak = Math.max(peak, Math.abs(l[i]!), Math.abs(r[i]!));
  if (peak < 1e-5) throw new Error("The recording is silent.");
  const g = 0.891 / peak;
  const left = new Float32Array(frames);
  const right = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    left[i] = l[i]! * g;
    right[i] = r[i]! * g;
  }
  return [left, right];
}

export async function runLocal(file: File, onEvent: (e: JobEvent) => void, signal: { cancelled: boolean }): Promise<LocalResult> {
  const t0 = performance.now();
  onEvent({ status: "running", stage: "preprocess", progress: 0.01, message: "Decoding in the browser" });
  const [left, right] = await decode(file);
  const duration = left.length / SR;

  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
  const done = new Promise<Extract<WorkerMessage, { type: "done" }>>((resolve, reject) => {
    worker.onmessage = (e: MessageEvent<WorkerMessage>) => {
      const m = e.data;
      if (signal.cancelled) {
        worker.terminate();
        reject(new Error("cancelled"));
      } else if (m.type === "progress") {
        // Overall: model load 0-10 %, separation 10-90 %, analysis 90-100 %.
        const base = m.stage === "model" ? 0.02 + 0.08 * m.progress : m.stage === "voices" ? 0.1 + 0.8 * m.progress : 0.9 + 0.1 * m.progress;
        onEvent({ status: "running", stage: m.stage === "model" ? "preprocess" : m.stage, progress: base, message: m.message });
      } else if (m.type === "error") reject(new Error(m.message));
      else resolve(m);
    };
    worker.onerror = (e) => reject(new Error(e.message || "The browser worker crashed."));
  });
  const req: RunRequest = { type: "run", left, right, modelUrl: LOCAL_MODEL_URL, colors: COLORS };
  worker.postMessage(req, [left.buffer, right.buffer]);
  let result: Extract<WorkerMessage, { type: "done" }>;
  try {
    result = await done;
  } finally {
    worker.terminate();
  }

  const urls: string[] = [];
  const url = (blob: Blob) => {
    const u = URL.createObjectURL(blob);
    urls.push(u);
    return u;
  };
  const id = `local-${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
  const stems: Stem[] = (["voices", "music"] as const).map((name) => ({
    name,
    label: name === "voices" ? "Voices" : "Music",
    parent: "root",
    color: COLORS[name],
    url: url(encodeWav(result.stems[name]!, SR)),
  }));
  const analysisStems: Record<string, StemAnalysis> = {};
  for (const name of ["voices", "music"]) {
    analysisStems[name] = { ...(result.analysis[name] as StemAnalysis), spectrogram_url: url(result.spectrograms[name]!) };
  }
  const total = (performance.now() - t0) / 1000;
  const job: Job = {
    id,
    status: "done",
    stage: "done",
    progress: 1,
    message: "Done",
    mode: "fast",
    filename: file.name,
    split_voices: false,
    lyrics: false,
    stages: LOCAL_STAGES,
    duration_s: duration,
    stems,
    timings: { total },
    models: { voices: `Kim_Vocal_2.onnx in the browser (${result.threads} thread${result.threads === 1 ? "" : "s"})` },
  };
  const analysis: Analysis = { duration_s: duration, sample_rate: SR, frame_rate: FPS, stems: analysisStems };
  return { job, stems, analysis, dispose: () => urls.forEach((u) => URL.revokeObjectURL(u)) };
}
