/**
 * Kim Vocal 2 (MDX-Net) pre/post-processing, matching audio-separator exactly:
 * torch.stft / torch.istft with a periodic Hann window, center=True with
 * reflect padding, n_fft 7680, hop 1024; the model sees the lowest 3072 bins of
 * 256 frames as 4 planes [L.re, L.im, R.re, R.im], with bins 0-2 zeroed.
 *
 * Processing uses MDX's own chunking without overlap: each 261 120-sample
 * window keeps only its middle 253 440 samples, so edges always have context.
 */
import { FFT } from "./fft";

export const N_FFT = 7680;
export const HOP = 1024;
export const DIM_F = 3072;
export const DIM_T = 256;
export const CHUNK = HOP * (DIM_T - 1); // 261 120
export const TRIM = N_FFT / 2; // 3 840
export const GEN = CHUNK - 2 * TRIM; // 253 440 new samples per model pass

const fft = new FFT(N_FFT);
const window = (() => {
  const w = new Float64Array(N_FFT);
  for (let n = 0; n < N_FFT; n++) w[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT); // periodic Hann
  return w;
})();
const scratch = { aRe: new Float64Array(N_FFT), aIm: new Float64Array(N_FFT), bRe: new Float64Array(N_FFT), bIm: new Float64Array(N_FFT) };

/** reflect pad like torch (does not repeat the edge sample) */
function reflectPad(x: Float32Array, pad: number): Float32Array {
  const out = new Float32Array(x.length + 2 * pad);
  out.set(x, pad);
  for (let i = 0; i < pad; i++) {
    out[pad - 1 - i] = x[i + 1] ?? 0;
    out[pad + x.length + i] = x[x.length - 2 - i] ?? 0;
  }
  return out;
}

/** Model input for one stereo window (each channel CHUNK long): Float32Array(4·DIM_F·DIM_T). */
export function modelInput(left: Float32Array, right: Float32Array): Float32Array {
  const input = new Float32Array(4 * DIM_F * DIM_T);
  const { aRe, aIm, bRe, bIm } = scratch;
  [left, right].forEach((ch, c) => {
    const padded = reflectPad(ch, N_FFT / 2);
    for (let t = 0; t < DIM_T; t++) {
      const off = t * HOP;
      for (let n = 0; n < N_FFT; n++) {
        aRe[n] = padded[off + n]! * window[n]!;
        aIm[n] = 0;
      }
      fft.transform(aRe, aIm, bRe, bIm);
      const reBase = (2 * c) * DIM_F * DIM_T;
      const imBase = (2 * c + 1) * DIM_F * DIM_T;
      for (let f = 3; f < DIM_F; f++) {
        // bins 0-2 stay zero, as in MDX
        input[reBase + f * DIM_T + t] = bRe[f]!;
        input[imBase + f * DIM_T + t] = bIm[f]!;
      }
    }
  });
  return input;
}

/** Inverse of modelInput for the model output: two channels, each CHUNK long. */
export function fromModelOutput(out: Float32Array): [Float32Array, Float32Array] {
  const { aRe, aIm, bRe, bIm } = scratch;
  const total = N_FFT + HOP * (DIM_T - 1);
  const result: Float32Array[] = [];
  const env = new Float64Array(total);
  for (let t = 0; t < DIM_T; t++) for (let n = 0; n < N_FFT; n++) env[t * HOP + n]! += window[n]! * window[n]!;
  for (let c = 0; c < 2; c++) {
    const acc = new Float64Array(total);
    const reBase = (2 * c) * DIM_F * DIM_T;
    const imBase = (2 * c + 1) * DIM_F * DIM_T;
    for (let t = 0; t < DIM_T; t++) {
      aRe.fill(0);
      aIm.fill(0);
      for (let f = 0; f < DIM_F; f++) {
        const re = out[reBase + f * DIM_T + t]!;
        const im = out[imBase + f * DIM_T + t]!;
        aRe[f] = re;
        aIm[f] = im;
        if (f > 0) {
          aRe[N_FFT - f] = re; // Hermitian: real signal
          aIm[N_FFT - f] = -im;
        }
      }
      fft.transform(aRe, aIm, bRe, bIm, true);
      const off = t * HOP;
      for (let n = 0; n < N_FFT; n++) acc[off + n]! += (bRe[n]! / N_FFT) * window[n]!;
    }
    const y = new Float32Array(CHUNK);
    const start = N_FFT / 2;
    for (let i = 0; i < CHUNK; i++) {
      const e = env[start + i]!;
      y[i] = e > 1e-11 ? acc[start + i]! / e : 0;
    }
    result.push(y);
  }
  return [result[0]!, result[1]!];
}

/** Window starts for a signal of `length` samples (the signal is preceded by TRIM zeros). */
export function blockCount(length: number): number {
  return Math.max(1, Math.ceil(length / GEN));
}

/** The CHUNK-long window for block b of a channel (with TRIM leading zeros, zero padded at the end). */
export function windowFor(channel: Float32Array, b: number): Float32Array {
  const w = new Float32Array(CHUNK);
  const startInSignal = b * GEN - TRIM;
  for (let i = 0; i < CHUNK; i++) {
    const j = startInSignal + i;
    if (j >= 0 && j < channel.length) w[i] = channel[j]!;
  }
  return w;
}
