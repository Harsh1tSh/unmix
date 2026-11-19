/**
 * Browser version of backend/app/analysis.py: peaks, RMS, 7-band energy and a
 * tinted mel spectrogram, normalised across the job the same way.
 */
import type { StemAnalysis } from "../api/types";
import { FFT } from "./fft";

export const SR = 44100;
export const FPS = 20;
const N = 4096;
const HOP = SR / FPS; // 2205
const BAND_NAMES = ["sub-bass", "bass", "low-mid", "mid", "high-mid", "presence", "brilliance"];
const BAND_EDGES = [20, 60, 250, 500, 2000, 4000, 6000, 20000];
const MELS = 96;
const PEAKS = 2000;
const MAX_IMG_W = 1200;

export interface Features {
  peaks: number[];
  rms: number[];
  bandEnergy: Float64Array[]; // [frame] -> 7 values
  mel: Float32Array[]; // [frame] -> MELS values
  energy: number;
}

const fft = new FFT(N);
const hann = Float64Array.from({ length: N }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N));
const hzToMel = (hz: number) => 2595 * Math.log10(1 + hz / 700);
const melToHz = (m: number) => 700 * (10 ** (m / 2595) - 1);
const melFilters = (() => {
  const lo = hzToMel(20);
  const hi = hzToMel(16000);
  const pts = Array.from({ length: MELS + 2 }, (_, i) => melToHz(lo + ((hi - lo) * i) / (MELS + 1)));
  const binHz = SR / N;
  return Array.from({ length: MELS }, (_, m) => {
    const [a, b, c] = [pts[m]!, pts[m + 1]!, pts[m + 2]!];
    const from = Math.floor(a / binHz);
    const to = Math.ceil(c / binHz);
    const weights: [number, number][] = [];
    for (let k = from; k <= to; k++) {
      const hz = k * binHz;
      const w = hz < b ? (hz - a) / (b - a) : (c - hz) / (c - b);
      if (w > 0) weights.push([k, (w * 2) / (c - a)]);
    }
    return weights;
  });
})();

export function extractFeatures(left: Float32Array, right: Float32Array): Features {
  const n = left.length;
  const peaks = new Array<number>(PEAKS).fill(0);
  let energy = 0;
  for (let i = 0; i < n; i++) {
    const a = Math.max(Math.abs(left[i]!), Math.abs(right[i]!));
    const p = Math.min(PEAKS - 1, Math.floor((i / n) * PEAKS));
    if (a > peaks[p]!) peaks[p] = a;
    energy += left[i]! * left[i]! + right[i]! * right[i]!;
  }
  const frames = 1 + Math.floor(n / HOP);
  const rms: number[] = [];
  const bandEnergy: Float64Array[] = [];
  const mel: Float32Array[] = [];
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  const oR = new Float64Array(N);
  const oI = new Float64Array(N);
  const power = new Float64Array(N / 2 + 1);
  for (let t = 0; t < frames; t++) {
    const center = t * HOP;
    let sq = 0;
    power.fill(0);
    for (const ch of [left, right]) {
      for (let k = 0; k < N; k++) {
        const j = center - N / 2 + k;
        const v = j >= 0 && j < n ? ch[j]! : 0;
        re[k] = v * hann[k]!;
        im[k] = 0;
        if (ch === left) sq += ((v + (j >= 0 && j < n ? right[j]! : 0)) / 2) ** 2;
      }
      fft.transform(re, im, oR, oI);
      for (let k = 0; k <= N / 2; k++) power[k]! += (oR[k]! * oR[k]! + oI[k]! * oI[k]!) / 2;
    }
    rms.push(Math.sqrt(sq / N));
    const bands = new Float64Array(BAND_NAMES.length);
    for (let k = 0; k <= N / 2; k++) {
      const hz = (k * SR) / N;
      for (let b = 0; b < BAND_NAMES.length; b++) if (hz >= BAND_EDGES[b]! && hz < BAND_EDGES[b + 1]!) bands[b]! += power[k]!;
    }
    bandEnergy.push(bands);
    mel.push(Float32Array.from(melFilters, (ws) => ws.reduce((acc, [k, w]) => acc + w * power[k]!, 0)));
  }
  return { peaks, rms, bandEnergy, mel, energy };
}

function hexRgb(color: string): [number, number, number] {
  const c = color.replace("#", "");
  return [parseInt(c.slice(0, 2), 16), parseInt(c.slice(2, 4), 16), parseInt(c.slice(4, 6), 16)];
}

/** Mel spectrogram as a PNG (near-black -> stem colour -> white), like the server's. */
export async function spectrogramPng(mel: Float32Array[], ref: number, color: string): Promise<Blob> {
  const [r, g, b] = hexRgb(color);
  const ramp = (v: number, c: number) => (v < 0.7 ? 8 + ((c - 8) * v) / 0.7 : c + ((255 - c) * (v - 0.7)) / 0.3);
  const w = Math.min(mel.length, MAX_IMG_W);
  const img = new ImageData(w, MELS);
  for (let x = 0; x < w; x++) {
    const from = Math.floor((x / w) * mel.length);
    const to = Math.max(from + 1, Math.floor(((x + 1) / w) * mel.length));
    for (let m = 0; m < MELS; m++) {
      let p = 0;
      for (let t = from; t < to; t++) p = Math.max(p, mel[t]![m]!);
      const db = 10 * Math.log10(Math.max(p, 1e-20) / Math.max(ref, 1e-20));
      const v = Math.min(1, Math.max(0, (db + 80) / 80));
      const o = ((MELS - 1 - m) * w + x) * 4; // low frequencies at the bottom
      img.data[o] = ramp(v, r);
      img.data[o + 1] = ramp(v, g);
      img.data[o + 2] = ramp(v, b);
      img.data[o + 3] = 255;
    }
  }
  const canvas = new OffscreenCanvas(w, MELS);
  canvas.getContext("2d")!.putImageData(img, 0, 0);
  return canvas.convertToBlob({ type: "image/png" });
}

/** Normalise across the job (like build_analysis on the server). */
export function buildStemAnalysis(f: Features, bandRef: number, mixEnergy: number, spectrogramUrl: string): StemAnalysis {
  const round = (x: number, d = 3) => Math.round(x * 10 ** d) / 10 ** d;
  return {
    peaks: f.peaks.map((p) => round(p)),
    rms: f.rms.map((v) => round(v, 4)),
    bands: {
      names: BAND_NAMES,
      hz_edges: BAND_EDGES,
      frames: f.bandEnergy.map((bands) => Array.from(bands, (e) => round(Math.min(1, Math.max(0, (10 * Math.log10(Math.max(e, 1e-20) / bandRef) + 60) / 60))))),
    },
    energy_share: mixEnergy > 0 ? round(f.energy / mixEnergy, 4) : 0,
    spectrogram_url: spectrogramUrl,
  };
}
