import { describe, expect, it } from "vitest";
import { FFT } from "./fft";
import { CHUNK, DIM_F, DIM_T, fromModelOutput, GEN, modelInput, TRIM, windowFor } from "./mdx";

function naiveDft(re: Float64Array, im: Float64Array): [Float64Array, Float64Array] {
  const n = re.length;
  const oR = new Float64Array(n);
  const oI = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    for (let t = 0; t < n; t++) {
      const a = (-2 * Math.PI * k * t) / n;
      oR[k]! += re[t]! * Math.cos(a) - im[t]! * Math.sin(a);
      oI[k]! += re[t]! * Math.sin(a) + im[t]! * Math.cos(a);
    }
  }
  return [oR, oI];
}

describe("FFT", () => {
  it.each([8, 12, 15, 30, 60, 120, 240])("matches a direct DFT for n = %i", (n) => {
    const re = Float64Array.from({ length: n }, (_, i) => Math.sin(i * 1.3) + (i % 3));
    const im = Float64Array.from({ length: n }, (_, i) => Math.cos(i * 0.7));
    const outR = new Float64Array(n);
    const outI = new Float64Array(n);
    new FFT(n).transform(re, im, outR, outI);
    const [eR, eI] = naiveDft(re, im);
    for (let k = 0; k < n; k++) {
      expect(outR[k]).toBeCloseTo(eR[k]!, 8);
      expect(outI[k]).toBeCloseTo(eI[k]!, 8);
    }
  });

  it("inverts (7680, the model's FFT size)", () => {
    const n = 7680;
    const f = new FFT(n);
    const re = Float64Array.from({ length: n }, (_, i) => Math.sin(i / 7) * Math.cos(i / 31));
    const im = new Float64Array(n);
    const R = new Float64Array(n);
    const I = new Float64Array(n);
    const back = new Float64Array(n);
    const backI = new Float64Array(n);
    f.transform(re, im, R, I);
    f.transform(R, I, back, backI, true);
    let err = 0;
    for (let i = 0; i < n; i++) err = Math.max(err, Math.abs(back[i]! / n - re[i]!));
    expect(err).toBeLessThan(1e-9);
  });

  it("refuses sizes with other prime factors", () => {
    expect(() => new FFT(7)).toThrow();
  });
});

describe("MDX STFT/ISTFT", () => {
  it("round-trips a window (identity 'model') in the region the model output is used", () => {
    const left = Float32Array.from({ length: CHUNK }, (_, i) => 0.3 * Math.sin((2 * Math.PI * 220 * i) / 44100));
    const right = Float32Array.from({ length: CHUNK }, (_, i) => 0.2 * Math.sin((2 * Math.PI * 1000 * i) / 44100));
    const spec = modelInput(left, right);
    expect(spec.length).toBe(4 * DIM_F * DIM_T);
    const [l, r] = fromModelOutput(spec);
    // Bins 0-2 (< 17 Hz) are zeroed and bins >= 3072 (> 17.6 kHz) dropped by design;
    // these tones are far from both, so the middle (the kept part) must reconstruct.
    let err = 0;
    for (let i = TRIM; i < TRIM + GEN; i += 7) err = Math.max(err, Math.abs(l[i]! - left[i]!), Math.abs(r[i]! - right[i]!));
    expect(err).toBeLessThan(1e-3);
  }, 30_000);

  it("windows cover the signal with TRIM samples of context", () => {
    const sig = Float32Array.from({ length: GEN + 100 }, (_, i) => i + 1);
    const w0 = windowFor(sig, 0);
    expect(w0[TRIM - 1]).toBe(0); // leading context is silence
    expect(w0[TRIM]).toBe(1); // first sample of the signal
    const w1 = windowFor(sig, 1);
    expect(w1[TRIM]).toBe(GEN + 1); // block 1 starts GEN samples later
    expect(w1[TRIM + 100]).toBe(0); // zero padded past the end
  });
});
