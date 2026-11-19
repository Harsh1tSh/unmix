/** 16-bit PCM WAV from planar float channels. */
export function encodeWav(channels: Float32Array[], sampleRate: number): Blob {
  const n = channels[0]?.length ?? 0;
  const c = channels.length;
  const pcm = new Int16Array(n * c);
  for (let i = 0; i < n; i++) for (let ch = 0; ch < c; ch++) pcm[i * c + ch] = Math.max(-1, Math.min(1, channels[ch]![i]!)) * 0x7fff;
  const header = new ArrayBuffer(44);
  const v = new DataView(header);
  const w = (off: number, s: string) => [...s].forEach((ch, i) => v.setUint8(off + i, ch.charCodeAt(0)));
  w(0, "RIFF");
  v.setUint32(4, 36 + pcm.byteLength, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, c, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * c * 2, true);
  v.setUint16(32, c * 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, pcm.byteLength, true);
  return new Blob([header, pcm], { type: "audio/wav" });
}

/** Interleaved stereo chunks -> planar channels. */
export function deinterleave(chunks: Float32Array[], frames: number): [Float32Array, Float32Array] {
  const l = new Float32Array(frames);
  const r = new Float32Array(frames);
  let o = 0;
  for (const chunk of chunks) {
    for (let i = 0; i + 1 < chunk.length && o < frames; i += 2, o++) {
      l[o] = chunk[i]!;
      r[o] = chunk[i + 1]!;
    }
  }
  return [l, r];
}
