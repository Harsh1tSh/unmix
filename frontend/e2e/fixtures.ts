/** Builds a small stereo 16-bit WAV in memory: bass note + chord + "voice" wobble + noise bursts. */
export function makeTestWav(seconds = 4, sampleRate = 44100): Buffer {
  const frames = Math.floor(seconds * sampleRate);
  const dataBytes = frames * 4;
  const buf = Buffer.alloc(44 + dataBytes);
  buf.write("RIFF", 0);
  buf.writeUInt32LE(36 + dataBytes, 4);
  buf.write("WAVE", 8);
  buf.write("fmt ", 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(2, 22); // stereo
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 4, 28);
  buf.writeUInt16LE(4, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36);
  buf.writeUInt32LE(dataBytes, 40);

  let seed = 1;
  const noise = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 1073741823.5 - 1;
  };
  for (let i = 0; i < frames; i++) {
    const t = i / sampleRate;
    const bass = 0.3 * Math.sin(2 * Math.PI * 80 * t);
    const chord = 0.15 * (Math.sin(2 * Math.PI * 440 * t) + Math.sin(2 * Math.PI * 554 * t));
    const voice = 0.25 * Math.sin(2 * Math.PI * (900 + 200 * Math.sin(2 * Math.PI * 3 * t)) * t);
    const hats = Math.sin(2 * Math.PI * 4 * t) > 0.7 ? 0.08 * noise() : 0;
    const sample = Math.max(-1, Math.min(1, bass + chord + voice + hats));
    const v = Math.round(sample * 32767);
    buf.writeInt16LE(v, 44 + i * 4);
    buf.writeInt16LE(v, 46 + i * 4);
  }
  return buf;
}
