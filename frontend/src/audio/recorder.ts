/**
 * Microphone capture for the landing orb.
 *
 * getUserMedia -> MediaStreamSource -> AnalyserNode (drives the orb)
 *                                  \-> MediaRecorder (the upload)
 *
 * The mic is opened only when the user starts a recording and closed again
 * when it stops: no ambient listening.
 */

export const MAX_RECORD_S = 6 * 60;
export const MIN_RECORD_S = 1;

const MIME_CANDIDATES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];

export interface Recording {
  file: File;
  seconds: number;
}

export class RecorderError extends Error {}

export class MicRecorder {
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private startedAt = 0;
  private bins: Float32Array<ArrayBuffer> = new Float32Array(0);
  private wave: Float32Array<ArrayBuffer> = new Float32Array(0);
  analyser: AnalyserNode | null = null;

  static supported(): boolean {
    return typeof window !== "undefined" && !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== "undefined";
  }

  get recording(): boolean {
    return this.recorder?.state === "recording";
  }

  elapsed(): number {
    return this.recording ? (performance.now() - this.startedAt) / 1000 : 0;
  }

  async start(): Promise<void> {
    if (!MicRecorder.supported()) throw new RecorderError("This browser cannot record audio. Drop a file instead.");
    try {
      // Processing that suits calls (noise suppression, AGC) would damage music, so it is off.
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 2 },
      });
    } catch (err) {
      const name = err instanceof DOMException ? err.name : "";
      throw new RecorderError(
        name === "NotAllowedError"
          ? "Microphone access was blocked. Allow it in the browser's site settings, or drop a file instead."
          : name === "NotFoundError"
            ? "No microphone was found."
            : "Could not open the microphone.",
      );
    }
    this.ctx = new AudioContext();
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.6;
    source.connect(this.analyser);
    this.bins = new Float32Array(this.analyser.frequencyBinCount);
    this.wave = new Float32Array(this.analyser.fftSize);

    const mimeType = MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m));
    this.recorder = new MediaRecorder(this.stream, mimeType ? { mimeType, audioBitsPerSecond: 192_000 } : undefined);
    this.chunks = [];
    this.recorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data);
    };
    this.recorder.start(250);
    this.startedAt = performance.now();
  }

  /** Stops and returns the recording (null if it was cancelled). */
  async stop(): Promise<Recording | null> {
    const rec = this.recorder;
    if (!rec || rec.state === "inactive") {
      this.release();
      return null;
    }
    const seconds = this.elapsed();
    const done = new Promise<void>((resolve) => (rec.onstop = () => resolve()));
    rec.stop();
    await done;
    const type = rec.mimeType || "audio/webm";
    const ext = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm";
    const blob = new Blob(this.chunks, { type });
    this.release();
    if (seconds < MIN_RECORD_S) throw new RecorderError("That was too short. Hold for at least a second.");
    const stamp = new Date().toTimeString().slice(0, 8).replace(/:/g, "-");
    return { file: new File([blob], `recording-${stamp}.${ext}`, { type }), seconds };
  }

  cancel(): void {
    if (this.recorder && this.recorder.state !== "inactive") {
      this.recorder.onstop = null;
      this.recorder.stop();
    }
    this.release();
  }

  /** RMS loudness 0..1 and 7 band levels 0..1, for drawing. */
  levels(bandEdges: number[]): { loudness: number; bands: number[] } {
    const a = this.analyser;
    if (!a || !this.ctx) return { loudness: 0, bands: bandEdges.slice(1).map(() => 0) };
    a.getFloatTimeDomainData(this.wave);
    let sum = 0;
    for (const v of this.wave) sum += v * v;
    const rms = Math.sqrt(sum / this.wave.length);
    a.getFloatFrequencyData(this.bins);
    const hzPerBin = this.ctx.sampleRate / a.fftSize;
    const bands = bandEdges.slice(1).map((hi, i) => {
      const lo = bandEdges[i]!;
      const from = Math.max(1, Math.floor(lo / hzPerBin));
      const to = Math.min(this.bins.length - 1, Math.ceil(hi / hzPerBin));
      let peak = -140;
      for (let k = from; k <= to; k++) peak = Math.max(peak, this.bins[k]!);
      return Math.min(1, Math.max(0, (peak + 90) / 70)); // -90..-20 dBFS -> 0..1
    });
    // RMS -> perceptual 0..1 (-50..-6 dBFS)
    const db = 20 * Math.log10(Math.max(rms, 1e-6));
    return { loudness: Math.min(1, Math.max(0, (db + 50) / 44)), bands };
  }

  private release(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    void this.ctx?.close().catch(() => undefined);
    this.ctx = null;
    this.analyser = null;
    this.recorder = null;
  }
}
