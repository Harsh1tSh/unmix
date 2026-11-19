/**
 * Mixed-radix complex FFT for sizes whose prime factors are 2, 3 and 5.
 *
 * The browser-only mode needs N = 7680 = 2^9 · 3 · 5 (the FFT size Kim Vocal 2
 * was trained with), which power-of-two FFT libraries cannot do. Recursive
 * decimation in time; twiddles and scratch space are precomputed per plan.
 */
export class FFT {
  readonly n: number;
  private readonly factors: number[];
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  private readonly yRe: Float64Array[];
  private readonly yIm: Float64Array[];

  constructor(n: number) {
    this.n = n;
    this.factors = [];
    let m = n;
    for (const p of [4, 2, 3, 5]) {
      while (m % p === 0) {
        this.factors.push(p);
        m /= p;
      }
    }
    if (m !== 1) throw new Error(`FFT size ${n} has prime factors other than 2, 3, 5`);
    this.cos = new Float64Array(n);
    this.sin = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      this.cos[k] = Math.cos((2 * Math.PI * k) / n);
      this.sin[k] = -Math.sin((2 * Math.PI * k) / n); // e^{-2πik/n}
    }
    this.yRe = this.factors.map((r) => new Float64Array(r));
    this.yIm = this.factors.map((r) => new Float64Array(r));
  }

  /** out = DFT(in) (or the unscaled inverse when inverse = true). Arrays of length n. */
  transform(inRe: Float64Array, inIm: Float64Array, outRe: Float64Array, outIm: Float64Array, inverse = false): void {
    this.rec(inRe, inIm, 0, 1, outRe, outIm, 0, this.n, 0, inverse ? -1 : 1);
  }

  private rec(inRe: Float64Array, inIm: Float64Array, inOff: number, stride: number, outRe: Float64Array, outIm: Float64Array, outOff: number, n: number, fi: number, sign: number): void {
    if (n === 1) {
      outRe[outOff] = inRe[inOff]!;
      outIm[outOff] = inIm[inOff]!;
      return;
    }
    const r = this.factors[fi]!;
    const m = n / r;
    for (let k = 0; k < r; k++) this.rec(inRe, inIm, inOff + k * stride, stride * r, outRe, outIm, outOff + k * m, m, fi + 1, sign);

    const N = this.n;
    const step = N / n; // twiddle index step at this level
    const rStep = N / r;
    const yr = this.yRe[fi]!;
    const yi = this.yIm[fi]!;
    const c = this.cos;
    const s = this.sin;
    for (let q = 0; q < m; q++) {
      // y_k = out[k*m + q] · W_n^{k q}
      for (let k = 0; k < r; k++) {
        const idx = outOff + k * m + q;
        const t = (k * q * step) % N;
        const wr = c[t]!;
        const wi = sign * s[t]!;
        const xr = outRe[idx]!;
        const xi = outIm[idx]!;
        yr[k] = xr * wr - xi * wi;
        yi[k] = xr * wi + xi * wr;
      }
      // X[q + j m] = Σ_k y_k · W_r^{k j}
      for (let j = 0; j < r; j++) {
        let accR = 0;
        let accI = 0;
        for (let k = 0; k < r; k++) {
          const t = (k * j * rStep) % N;
          const wr = c[t]!;
          const wi = sign * s[t]!;
          accR += yr[k]! * wr - yi[k]! * wi;
          accI += yr[k]! * wi + yi[k]! * wr;
        }
        outRe[outOff + j * m + q] = accR;
        outIm[outOff + j * m + q] = accI;
      }
    }
  }
}
