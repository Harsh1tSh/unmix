/**
 * Live session: microphone -> server (WebSocket) -> separated voices/music
 * played back on one clock.
 *
 *   mic -> MediaStreamSource -> AudioWorklet (capture) -> WebSocket
 *                            -> AnalyserNode (input meter)
 *   WebSocket blocks -> AudioBufferSourceNode -> voicesGain / musicGain -> monitor -> speakers
 *
 * The server needs 5.75 s of audio before the first block and then about as
 * long again to compute it (on a laptop CPU), so playback runs ~11 s behind.
 * Blocks are scheduled at base + block.start / 44100 on the AudioContext
 * clock; if one arrives too late, the base moves back and the delay grows.
 */

import { deinterleave, encodeWav } from "../lib/wav";
import workletUrl from "./captureWorklet.ts?worker&url";

export const LIVE_SR = 44100;

export interface LiveBlock {
  index: number;
  start: number; // sample position in the input stream
  frames: number;
  voices: [Float32Array<ArrayBuffer>, Float32Array<ArrayBuffer>];
  music: [Float32Array<ArrayBuffer>, Float32Array<ArrayBuffer>];
  computeS: number;
}

export interface LiveStats {
  state: "connecting" | "waiting" | "live" | "stopping" | "ended" | "error";
  error: string | null;
  sentSeconds: number;
  blocks: number;
  /** seconds between speaking and hearing it back */
  delay: number | null;
  /** compute time / audio time of the last block: > 1 means the machine cannot keep up */
  speed: number | null;
  skippedSeconds: number;
}

type Listener = (stats: LiveStats) => void;

export class LiveSession {
  readonly ctx: AudioContext;
  readonly inputAnalyser: AnalyserNode;
  readonly voicesGain: GainNode;
  readonly musicGain: GainNode;
  readonly monitor: GainNode;
  readonly voicesAnalyser: AnalyserNode;
  readonly musicAnalyser: AnalyserNode;
  /** every block received, for the scrolling lanes */
  readonly blocks: LiveBlock[] = [];

  private ws: WebSocket | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private recorded: Float32Array<ArrayBuffer>[] = [];
  private recordedFrames = 0;
  private base: number | null = null; // ctx time at which input sample 0 is heard
  private pendingHeader: { index: number; start: number; frames: number; compute_s: number } | null = null;
  private listeners = new Set<Listener>();
  private sources: AudioBufferSourceNode[] = [];
  private stopResolve: (() => void) | null = null;
  private disposed = false;
  stats: LiveStats = { state: "connecting", error: null, sentSeconds: 0, blocks: 0, delay: null, speed: null, skippedSeconds: 0 };

  constructor() {
    this.ctx = new AudioContext({ sampleRate: LIVE_SR, latencyHint: "playback" });
    this.inputAnalyser = this.ctx.createAnalyser();
    this.inputAnalyser.fftSize = 1024;
    this.monitor = this.ctx.createGain();
    this.monitor.gain.value = 0; // off until the user chooses to listen (feedback risk on speakers)
    this.voicesGain = this.ctx.createGain();
    this.musicGain = this.ctx.createGain();
    this.voicesAnalyser = this.ctx.createAnalyser();
    this.musicAnalyser = this.ctx.createAnalyser();
    this.voicesGain.connect(this.voicesAnalyser).connect(this.monitor);
    this.musicGain.connect(this.musicAnalyser).connect(this.monitor);
    this.monitor.connect(this.ctx.destination);
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    listener(this.stats);
    return () => this.listeners.delete(listener);
  }

  private set(patch: Partial<LiveStats>): void {
    this.stats = { ...this.stats, ...patch };
    this.listeners.forEach((l) => l(this.stats));
  }

  async start(): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 2 },
      });
    } catch (err) {
      const name = err instanceof DOMException ? err.name : "";
      return this.fail(name === "NotAllowedError" ? "Microphone access was blocked. Allow it in the browser's site settings." : "Could not open the microphone.");
    }
    // Disposed while waiting (e.g. the screen was left): release the mic and stop.
    if (this.disposed) return void this.stream.getTracks().forEach((t) => t.stop());
    try {
      await this.ctx.resume();
      await this.ctx.audioWorklet.addModule(workletUrl);
    } catch {
      return this.disposed ? undefined : this.fail("Could not start audio capture in this browser.");
    }
    if (this.disposed) return void this.stream.getTracks().forEach((t) => t.stop());
    const source = this.ctx.createMediaStreamSource(this.stream);
    source.connect(this.inputAnalyser);
    this.node = new AudioWorkletNode(this.ctx, "unmix-capture", { numberOfInputs: 1, numberOfOutputs: 0, channelCount: 2, channelCountMode: "explicit" });
    source.connect(this.node);

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/api/live`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onopen = () => ws.send(JSON.stringify({ type: "start", sample_rate: LIVE_SR, channels: 2 }));
    ws.onerror = () => this.fail("Could not reach the live server. Is the backend running?");
    ws.onclose = () => {
      if (this.stats.state !== "ended" && this.stats.state !== "error") this.set({ state: "ended" });
      this.stopResolve?.();
    };
    ws.onmessage = (msg) => this.onMessage(msg.data as string | ArrayBuffer);

    this.node.port.onmessage = (e: MessageEvent<Float32Array<ArrayBuffer>>) => {
      if (this.stats.state !== "waiting" && this.stats.state !== "live") return;
      const frame = e.data;
      this.recorded.push(frame);
      this.recordedFrames += frame.length / 2;
      if (ws.readyState === WebSocket.OPEN) ws.send(frame.buffer);
      this.set({ sentSeconds: this.recordedFrames / LIVE_SR });
    };
  }

  private onMessage(data: string | ArrayBuffer): void {
    if (typeof data === "string") {
      const msg = JSON.parse(data) as { type: string; message?: string; skipped_frames?: number; index: number; start: number; frames: number; compute_s: number };
      if (msg.type === "ready") this.set({ state: "waiting" });
      else if (msg.type === "block") this.pendingHeader = msg;
      else if (msg.type === "behind") this.set({ skippedSeconds: this.stats.skippedSeconds + (msg.skipped_frames ?? 0) / LIVE_SR });
      else if (msg.type === "end") {
        this.set({ state: "ended" });
        this.stopResolve?.();
      } else if (msg.type === "error") this.fail(msg.message ?? "The live session failed.");
      return;
    }
    const h = this.pendingHeader;
    this.pendingHeader = null;
    if (!h) return;
    const all = new Float32Array(data);
    const n = h.frames;
    const block: LiveBlock = {
      index: h.index,
      start: h.start,
      frames: n,
      voices: [all.subarray(0, n), all.subarray(n, 2 * n)],
      music: [all.subarray(2 * n, 3 * n), all.subarray(3 * n, 4 * n)],
      computeS: h.compute_s,
    };
    this.blocks.push(block);
    this.schedule(block);
    this.set({ state: this.stats.state === "waiting" ? "live" : this.stats.state, blocks: this.blocks.length, speed: block.computeS / (n / LIVE_SR) });
  }

  private schedule(block: LiveBlock): void {
    const now = this.ctx.currentTime;
    const at = (b: number) => b + block.start / LIVE_SR;
    // First block, or one that would start in the past: (re)base with a small safety margin.
    if (this.base === null || at(this.base) < now + 0.05) this.base = now + 0.15 - block.start / LIVE_SR;
    const when = at(this.base);
    for (const [stem, gain] of [
      [block.voices, this.voicesGain],
      [block.music, this.musicGain],
    ] as const) {
      const buf = this.ctx.createBuffer(2, block.frames, LIVE_SR);
      buf.copyToChannel(stem[0], 0);
      buf.copyToChannel(stem[1], 1);
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.connect(gain);
      src.start(when);
      src.onended = () => (this.sources = this.sources.filter((s) => s !== src));
      this.sources.push(src);
    }
    // Delay = when this block plays minus when its audio was captured.
    const capturedAt = this.captureClock(block.start);
    if (capturedAt !== null) this.set({ delay: when - capturedAt });
  }

  private captureStart: number | null = null;
  /** ctx time at which input sample `pos` was captured (approximate). */
  private captureClock(pos: number): number | null {
    if (this.captureStart === null) this.captureStart = this.ctx.currentTime - this.recordedFrames / LIVE_SR;
    return this.captureStart + pos / LIVE_SR;
  }

  /** Playback position in input samples (for the scrolling lanes). */
  playheadSample(): number | null {
    if (this.base === null) return null;
    return (this.ctx.currentTime - this.base) * LIVE_SR;
  }

  setListening(on: boolean): void {
    this.monitor.gain.setTargetAtTime(on ? 1 : 0, this.ctx.currentTime, 0.03);
  }

  setLevels(voices: number, music: number): void {
    this.voicesGain.gain.setTargetAtTime(voices, this.ctx.currentTime, 0.03);
    this.musicGain.gain.setTargetAtTime(music, this.ctx.currentTime, 0.03);
  }

  /** Stop capturing; the server finishes the last block, then the session ends. */
  async stop(): Promise<void> {
    if (this.stats.state === "ended" || this.stats.state === "error") return;
    this.set({ state: "stopping" });
    this.stream?.getTracks().forEach((t) => t.stop());
    this.node?.disconnect();
    const done = new Promise<void>((resolve) => (this.stopResolve = resolve));
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: "stop" }));
    else this.stopResolve?.();
    await Promise.race([done, new Promise((r) => setTimeout(r, 30_000))]);
  }

  /** The whole recording as a 16-bit WAV file, to run through the full pipeline. */
  recordingFile(): File | null {
    if (this.recordedFrames < LIVE_SR) return null;
    const stamp = new Date().toTimeString().slice(0, 8).replace(/:/g, "-");
    const wav = encodeWav(deinterleave(this.recorded, this.recordedFrames), LIVE_SR);
    return new File([wav], `live-${stamp}.wav`, { type: "audio/wav" });
  }

  private fail(message: string): void {
    this.set({ state: "error", error: message });
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ws?.close();
  }

  dispose(): void {
    this.disposed = true;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.sources.forEach((s) => {
      try {
        s.stop();
      } catch {
        // already stopped
      }
    });
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) this.ws.close();
    this.listeners.clear();
    void this.ctx.close().catch(() => undefined);
  }
}
