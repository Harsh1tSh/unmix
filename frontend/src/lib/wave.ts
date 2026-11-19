/** The brand gradient (the stem colours), horizontally across `width`. */
export function auroraGradient(ctx: CanvasRenderingContext2D, width: number): CanvasGradient {
  const g = ctx.createLinearGradient(0, 0, width, 0);
  [["#FF6B9A", 0], ["#E879F9", 0.28], ["#A78BFA", 0.52], ["#38BDF8", 0.78], ["#34D399", 1]].forEach(([c, o]) => g.addColorStop(o as number, c as string));
  return g;
}

/** Mirrored bar waveform. Shared by the card waveforms and the split animation. */
export function drawBars(ctx: CanvasRenderingContext2D, width: number, height: number, peaks: number[], scale: number, color: string | CanvasGradient, barWidth = 2, gap = 1): void {
  const bars = Math.max(1, Math.floor(width / (barWidth + gap)));
  const mid = height / 2;
  ctx.fillStyle = color;
  for (let i = 0; i < bars; i++) {
    const from = Math.floor((i / bars) * peaks.length);
    const to = Math.max(from + 1, Math.floor(((i + 1) / bars) * peaks.length));
    let peak = 0;
    for (let j = from; j < to; j++) peak = Math.max(peak, peaks[j] ?? 0);
    const h = Math.max(1, Math.min(1, peak / scale) * (height - 2));
    const x = i * (barWidth + gap);
    // rounded caps once the bars are wide enough to show them
    if (barWidth >= 2.5 && h > barWidth && ctx.roundRect) {
      ctx.beginPath();
      ctx.roundRect(x, mid - h / 2, barWidth, h, barWidth / 2);
      ctx.fill();
    } else ctx.fillRect(x, mid - h / 2, barWidth, h);
  }
}

/** Decode an audio file in the browser and reduce it to `points` peak values (0..1). */
export async function peaksFromFile(file: Blob, points = 800): Promise<number[]> {
  const buf = await file.arrayBuffer();
  const ctx = new OfflineAudioContext(1, 1, 44100);
  const audio = await ctx.decodeAudioData(buf);
  const channels = Array.from({ length: audio.numberOfChannels }, (_, c) => audio.getChannelData(c));
  const n = audio.length;
  const out = new Array<number>(points).fill(0);
  for (let p = 0; p < points; p++) {
    const from = Math.floor((p / points) * n);
    const to = Math.floor(((p + 1) / points) * n);
    let m = 0;
    // Stride keeps this fast for long files; peaks of a stride-sampled signal are a fine overview.
    const stride = Math.max(1, Math.floor((to - from) / 400));
    for (const ch of channels) for (let i = from; i < to; i += stride) m = Math.max(m, Math.abs(ch[i]!));
    out[p] = m;
  }
  return out;
}

export function maxPeak(peaks: number[]): number {
  let m = 0;
  for (const v of peaks) if (v > m) m = v;
  return m;
}
