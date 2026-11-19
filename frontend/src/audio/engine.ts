/**
 * StemEngine: sample-synchronised playback of all stems through Web Audio.
 *
 * How sync is guaranteed
 *   - every leaf stem is fully decoded into an AudioBuffer (same length, same rate),
 *   - on play/seek one AudioBufferSourceNode per leaf is created and all of them
 *     are started with the SAME (when, offset) on the audio clock,
 *   - mixing is done only with GainNodes, which never shift time.
 * The stems therefore stay phase-locked for the whole playback; because they
 * also sum to the mix, the full mix is literally all gains at 1.
 *
 * Graph:  source[leaf] -> gain[leaf] -> master (fades) -> output (user volume) -> analyser -> destination
 *
 * React reads coarse state through subscribe()/getSnapshot() (useSyncExternalStore)
 * and per-frame position through onFrame(), which never causes a re-render.
 */

import type { Stem } from "../api/types";
import { buildTree, computeLeafGains, DEFAULT_CONTROL, type MixState, type StemControl, type StemTree } from "./mixer";

export const CROSSFADE_S = 0.15;
const START_LEAD_S = 0.06; // schedule slightly ahead so every source is ready
const PREVIEW_RELEASE_MS = 220; // hover-preview keeps playing this long after the cursor leaves

export interface EngineSnapshot {
  status: "loading" | "ready" | "error";
  /** 0..1 while stems are fetched and decoded */
  loadProgress: number;
  error: string | null;
  playing: boolean;
  /** playing only because a card is hovered */
  preview: boolean;
  /** the browser refused to start audio without a click */
  audioBlocked: boolean;
  duration: number;
  /** restart from the beginning at the end */
  loop: boolean;
  /** 0..1, the transport's volume */
  masterVolume: number;
  hover: string | null;
  controls: Record<string, StemControl>;
  /** current target gain per leaf */
  gains: Record<string, number>;
}

type FrameListener = (positionS: number, playing: boolean) => void;

export class StemEngine {
  readonly tree: StemTree;
  private readonly ctx: AudioContext;
  private readonly master: GainNode;
  private readonly output: GainNode;
  private readonly analyser: AnalyserNode;
  private readonly gainNodes = new Map<string, GainNode>();
  private readonly buffers = new Map<string, AudioBuffer>();
  private sources: AudioBufferSourceNode[] = [];

  private startedAt = 0; // ctx time at which `startOffset` is heard
  private startOffset = 0;
  private pausedAt = 0;
  private generation = 0;
  private raf = 0;
  private previewTimer: ReturnType<typeof setTimeout> | null = null;
  private disposed = false;
  private readonly abort = new AbortController();

  private mix: MixState;
  private snapshot: EngineSnapshot;
  private readonly listeners = new Set<() => void>();
  private readonly frameListeners = new Set<FrameListener>();

  constructor(private readonly stems: Stem[]) {
    this.tree = buildTree(stems);
    // Match the stems' rate so the browser does not resample on decode.
    this.ctx = new AudioContext({ sampleRate: 44100, latencyHint: "interactive" });
    this.master = this.ctx.createGain();
    this.output = this.ctx.createGain();
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.master.connect(this.output).connect(this.analyser).connect(this.ctx.destination);
    for (const leaf of this.tree.leaves) {
      const g = this.ctx.createGain();
      g.connect(this.master);
      this.gainNodes.set(leaf, g);
    }
    this.mix = { hover: null, controls: Object.fromEntries(stems.map((s) => [s.name, { ...DEFAULT_CONTROL }])) };
    this.snapshot = {
      status: "loading",
      loadProgress: 0,
      error: null,
      playing: false,
      preview: false,
      audioBlocked: false,
      duration: 0,
      hover: null,
      controls: this.mix.controls,
      gains: computeLeafGains(this.tree, this.mix),
      loop: false,
      masterVolume: 1,
    };
  }

  // ------------------------------------------------------------ store API

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): EngineSnapshot => this.snapshot;

  /** Called every animation frame while playing, and once after seek/pause. */
  onFrame(listener: FrameListener): () => void {
    this.frameListeners.add(listener);
    listener(this.position(), this.snapshot.playing);
    return () => this.frameListeners.delete(listener);
  }

  private update(patch: Partial<EngineSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    this.listeners.forEach((l) => l());
  }

  private emitFrame = (): void => {
    const pos = this.position();
    this.frameListeners.forEach((l) => l(pos, this.snapshot.playing));
  };

  private loop = (): void => {
    this.emitFrame();
    if (this.snapshot.playing && !this.disposed) this.raf = requestAnimationFrame(this.loop);
  };

  // -------------------------------------------------------------- loading

  /** Fetch + decode every leaf stem. Two at a time keeps peak memory in check. */
  async load(): Promise<void> {
    const leaves = this.tree.leaves;
    const urls = new Map(this.stems.map((s) => [s.name, s.url]));
    let done = 0;
    const queue = [...leaves];
    const worker = async (): Promise<void> => {
      for (let name = queue.shift(); name !== undefined; name = queue.shift()) {
        const res = await fetch(urls.get(name)!, { signal: this.abort.signal });
        if (!res.ok) throw new Error(`Could not load the "${name}" stem (${res.status}).`);
        const encoded = await res.arrayBuffer();
        if (this.disposed) return;
        this.buffers.set(name, await this.ctx.decodeAudioData(encoded));
        done += 1;
        this.update({ loadProgress: done / leaves.length });
      }
    };
    try {
      await Promise.all([worker(), worker()]);
      if (this.disposed) return;
      const lengths = leaves.map((n) => this.buffers.get(n)!.length);
      if (new Set(lengths).size > 1) console.warn("[unmix] stems differ in length", lengths);
      const duration = Math.min(...leaves.map((n) => this.buffers.get(n)!.duration));
      this.update({ status: "ready", duration, loadProgress: 1 });
      this.emitFrame();
    } catch (err) {
      if (!this.disposed) this.update({ status: "error", error: err instanceof Error ? err.message : String(err) });
    }
  }

  // ------------------------------------------------------------ transport

  position(): number {
    if (!this.snapshot.playing) return this.pausedAt;
    const t = this.startOffset + Math.max(0, this.ctx.currentTime - this.startedAt);
    return Math.min(t, this.snapshot.duration);
  }

  async play(opts: { preview?: boolean } = {}): Promise<boolean> {
    if (this.snapshot.status !== "ready" || this.disposed) return false;
    if (this.snapshot.playing) {
      // Pressing play during a hover-preview turns it into normal playback.
      if (!opts.preview && this.snapshot.preview) this.update({ preview: false });
      return true;
    }
    if (this.ctx.state !== "running") {
      // resume() only succeeds after a user gesture (autoplay policy); a hover is not one.
      await Promise.race([this.ctx.resume().catch(() => undefined), new Promise((r) => setTimeout(r, 300))]);
      // Re-read: the state changes while awaiting (TypeScript still assumes the narrowed value).
      if ((this.ctx.state as AudioContextState) !== "running") {
        this.update({ audioBlocked: true });
        return false;
      }
    }
    if (this.snapshot.playing || this.disposed) return true;
    if (opts.preview && this.mix.hover === null) return false; // cursor already left
    if (this.pausedAt >= this.snapshot.duration - 0.05) this.pausedAt = 0;
    this.startSources(this.pausedAt);
    this.update({ playing: true, preview: opts.preview ?? false, audioBlocked: false });
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(this.loop);
    return true;
  }

  pause(): void {
    if (!this.snapshot.playing) return;
    this.pausedAt = this.position();
    this.stopSources(true);
    this.update({ playing: false, preview: false });
    this.emitFrame();
  }

  toggle(): void {
    if (this.snapshot.playing && !this.snapshot.preview) this.pause();
    else void this.play();
  }

  seek(seconds: number): void {
    const t = Math.max(0, Math.min(seconds, this.snapshot.duration));
    if (this.snapshot.playing) {
      this.stopSources(false);
      this.startSources(t);
    } else {
      this.pausedAt = t;
    }
    this.emitFrame();
  }

  private startSources(offset: number): void {
    const when = this.ctx.currentTime + START_LEAD_S;
    const gen = ++this.generation;
    // Short fade-in on the master avoids a click when starting mid-waveform.
    this.master.gain.cancelScheduledValues(0);
    this.master.gain.setValueAtTime(0, this.ctx.currentTime);
    this.master.gain.setValueAtTime(0, when);
    this.master.gain.linearRampToValueAtTime(1, when + 0.03);

    this.sources = this.tree.leaves.map((leaf) => {
      const src = this.ctx.createBufferSource();
      src.buffer = this.buffers.get(leaf)!;
      src.connect(this.gainNodes.get(leaf)!);
      src.start(when, offset); // identical (when, offset) for every stem => sample-locked
      return src;
    });
    const first = this.sources[0];
    if (first) {
      first.onended = () => {
        if (gen !== this.generation || !this.snapshot.playing) return;
        this.stopSources(false);
        if (this.snapshot.loop && !this.snapshot.preview) {
          this.startSources(0); // seamless restart, same clock for every stem
          this.emitFrame();
          return;
        }
        this.pausedAt = 0;
        this.update({ playing: false, preview: false });
        this.emitFrame();
      };
    }
    this.startedAt = when;
    this.startOffset = offset;
  }

  private stopSources(fade: boolean): void {
    this.generation += 1; // invalidates pending onended handlers
    const now = this.ctx.currentTime;
    const stopAt = fade ? now + 0.04 : now;
    if (fade) {
      this.master.gain.cancelScheduledValues(0);
      this.master.gain.setValueAtTime(this.master.gain.value, now);
      this.master.gain.linearRampToValueAtTime(0, stopAt);
    }
    for (const src of this.sources) {
      src.onended = null;
      try {
        src.stop(stopAt);
      } catch {
        // already stopped
      }
    }
    this.sources = [];
  }

  /** Jump relative to the playhead (keyboard and skip buttons). */
  skip(seconds: number): void {
    this.seek(this.position() + seconds);
  }

  setLoop(loop: boolean): void {
    this.update({ loop });
  }

  setMasterVolume(volume: number): void {
    const v = Math.max(0, Math.min(1, volume));
    const now = this.ctx.currentTime;
    this.output.gain.cancelScheduledValues(now);
    this.output.gain.setValueAtTime(this.output.gain.value, now);
    this.output.gain.linearRampToValueAtTime(v * v, now + 0.05); // squared: closer to perceived loudness
    this.update({ masterVolume: v });
  }

  // --------------------------------------------------------------- mixing

  private applyMix(): void {
    const gains = computeLeafGains(this.tree, this.mix);
    const now = this.ctx.currentTime;
    for (const [leaf, node] of this.gainNodes) {
      const target = gains[leaf] ?? 0;
      node.gain.cancelScheduledValues(now);
      node.gain.setValueAtTime(node.gain.value, now);
      node.gain.linearRampToValueAtTime(target, now + CROSSFADE_S);
    }
    this.update({ gains, hover: this.mix.hover, controls: this.mix.controls });
  }

  private setControl(name: string, patch: Partial<StemControl>): void {
    const current = this.mix.controls[name] ?? DEFAULT_CONTROL;
    this.mix = { ...this.mix, controls: { ...this.mix.controls, [name]: { ...current, ...patch } } };
    this.applyMix();
  }

  setVolume(name: string, volume: number): void {
    this.setControl(name, { volume: Math.max(0, Math.min(1, volume)) });
  }

  toggleMute(name: string): void {
    this.setControl(name, { muted: !(this.mix.controls[name] ?? DEFAULT_CONTROL).muted });
  }

  toggleSolo(name: string): void {
    this.setControl(name, { solo: !(this.mix.controls[name] ?? DEFAULT_CONTROL).solo });
  }

  /**
   * HOVER = LISTEN. Solos `name` with a crossfade; if nothing is playing, starts
   * a preview that stops again shortly after the cursor leaves every card.
   */
  hoverStart(name: string): void {
    if (this.previewTimer) clearTimeout(this.previewTimer);
    this.previewTimer = null;
    if (this.mix.hover !== name) {
      this.mix = { ...this.mix, hover: name };
      this.applyMix();
    }
    if (!this.snapshot.playing) void this.play({ preview: true });
  }

  hoverEnd(name: string): void {
    if (this.mix.hover !== name) return; // the cursor already entered another card
    this.mix = { ...this.mix, hover: null };
    this.applyMix();
    if (this.snapshot.preview) {
      this.previewTimer = setTimeout(() => {
        this.previewTimer = null;
        if (this.snapshot.preview && this.mix.hover === null) this.pause();
      }, PREVIEW_RELEASE_MS);
    }
  }

  // ---------------------------------------------------------- diagnostics

  /** RMS of what is actually reaching the speakers right now (used by e2e tests). */
  outputLevel(): number {
    const data = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (const v of data) sum += v * v;
    return Math.sqrt(sum / data.length);
  }

  /** Instantaneous (ramping) gain values, as opposed to snapshot.gains targets. */
  liveGains(): Record<string, number> {
    return Object.fromEntries([...this.gainNodes].map(([k, n]) => [k, n.gain.value]));
  }

  contextState(): AudioContextState {
    return this.ctx.state;
  }

  dispose(): void {
    this.disposed = true;
    this.abort.abort(); // cancel stem downloads still in flight
    if (this.previewTimer) clearTimeout(this.previewTimer);
    cancelAnimationFrame(this.raf);
    this.stopSources(false);
    this.listeners.clear();
    this.frameListeners.clear();
    void this.ctx.close().catch(() => undefined);
  }
}
