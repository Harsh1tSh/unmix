import { useEffect, useRef, useState } from "react";
import type { Analysis, Health, JobEvent, StageInfo, Stem } from "../api/types";
import { useCanvas } from "../lib/canvas";
import { formatTime } from "../lib/format";
import { auroraGradient, drawBars, maxPeak } from "../lib/wave";
import { SoundSphere } from "../visual/SoundSphere";
import { SplitStrands, type StrandRects } from "./Split";
import { useHealth } from "./UploadScreen";

/** Used until the job's own stage list has loaded. */
const DEFAULT_STAGES: StageInfo[] = [
  { name: "preprocess", label: "Normalising loudness" },
  { name: "voices", label: "Separating voices from music" },
  { name: "music", label: "Splitting music into instruments" },
  { name: "analysis", label: "Analysing frequency content" },
];

interface Props {
  filename: string;
  /** 0..1 while the file is uploading, null once the job exists */
  uploadProgress: number | null;
  event: JobEvent | null;
  stages: StageInfo[] | null;
  /** peaks of the local file, decoded in the browser (null for restored jobs) */
  mixPeaks: number[] | null;
  /** set when the job is done: plays the split, then calls onSplitDone */
  split: { stems: Stem[]; analysis: Analysis } | null;
  onSplitDone: (rects: StrandRects) => void;
  onCancel: () => void;
  /** separating inside this browser rather than on the server */
  inBrowser?: boolean;
}

function MixWave({ peaks, progress, fading }: { peaks: number[]; progress: number; fading: boolean }) {
  const scale = Math.max(0.05, maxPeak(peaks));
  const dim = useCanvas((ctx, w, h) => drawBars(ctx, w, h, peaks, scale, "rgba(255,255,255,0.11)", 3, 2), [peaks]);
  const lit = useCanvas((ctx, w, h) => drawBars(ctx, w, h, peaks, scale, auroraGradient(ctx, w), 3, 2), [peaks]);
  return (
    <div className="absolute inset-0 transition-opacity duration-500" style={{ opacity: fading ? 0 : 1 }}>
      <canvas ref={dim} className="absolute inset-0 h-full w-full" />
      <div className="absolute inset-0 transition-[clip-path] duration-700 ease-out" style={{ clipPath: `inset(0 ${(1 - progress) * 100}% 0 0)` }}>
        <canvas ref={lit} className={`absolute inset-0 h-full w-full ${fading ? "" : "wave-sweep"}`} />
      </div>
    </div>
  );
}

/** What the speed note under the stages says, for the machine actually doing the work. */
function speedNote(health: Health | null, inBrowser: boolean): string {
  if (inBrowser) return "in this browser this takes several times the clip length";
  if (!health) return "";
  const gpu = health.device !== "cpu" || health.accelerator?.onnx !== "cpu";
  const summary = health.accelerator?.summary ?? health.device.toUpperCase();
  if (gpu) return `running on ${summary}${health.remote ? " (remote)" : ""}`;
  return "on a CPU this takes several times the clip length";
}

export function ProgressScreen({ filename, uploadProgress, event, stages, mixPeaks, split, onSplitDone, onCancel, inBrowser = false }: Props) {
  const [startedAt] = useState(() => Date.now());
  const [elapsed, setElapsed] = useState(0);
  const maxProgress = useRef(0);
  const { health } = useHealth();
  useEffect(() => {
    const timer = setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 1000);
    return () => clearInterval(timer);
  }, [startedAt]);

  const list = stages ?? DEFAULT_STAGES;
  const uploading = uploadProgress !== null;
  const done = split !== null;
  const raw = done ? 1 : uploading ? 0 : (event?.progress ?? 0);
  maxProgress.current = Math.max(maxProgress.current, raw);
  const progress = maxProgress.current;
  const currentIndex = done ? list.length : list.findIndex((s) => s.name === event?.stage);
  const message = done ? "Separated" : uploading ? `Uploading… ${Math.round((uploadProgress ?? 0) * 100)}%` : (event?.message ?? "Starting…");
  const waiting = !done && (uploading || !event || event.status === "queued");
  const note = speedNote(health, inBrowser);

  return (
    <main className="relative isolate mx-auto grid min-h-dvh max-w-6xl content-center items-center gap-4 overflow-x-clip px-4 py-10 sm:px-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,min(25rem,60vh))] lg:gap-14 2xl:max-w-[1440px] 2xl:grid-cols-[minmax(0,1fr)_minmax(0,min(32rem,60vh))]" data-testid="progress-screen">
      <div className="order-2 min-w-0 lg:order-1">
        <p className="eyebrow flex items-center gap-2.5">
          <span className={done ? "text-text" : ""}>{done ? "Unmixed" : "Unmixing"}</span>
          <span className="text-faint" aria-hidden>
            ·
          </span>
          <span className="num text-faint" title="Time since you started">
            {formatTime(elapsed)}
          </span>
        </p>
        <h1 className="display mt-4 truncate pb-1 text-[clamp(2.25rem,5.2vw,4.5rem)]" title={filename}>
          {filename}
        </h1>

        <div className="relative mt-8 h-24 sm:mt-10 sm:h-32" data-testid="mix-wave">
          {mixPeaks ? <MixWave peaks={mixPeaks} progress={progress} fading={done} /> : <div className="absolute inset-x-0 top-1/2 h-px bg-line-strong" />}
          {split && <SplitStrands stems={split.stems} analysis={split.analysis} onDone={onSplitDone} />}
        </div>

        <div className="mt-8 flex items-end justify-between gap-4">
          <p className="min-w-0 truncate pb-1 text-base text-text" aria-live="polite" data-testid="progress-message">
            {message}
          </p>
          <p className="num shrink-0 font-display text-5xl leading-none font-light tracking-[-0.04em] sm:text-6xl" data-testid="progress-percent">
            {Math.round(progress * 100)}
            <span className="text-muted">%</span>
          </p>
        </div>
        <div
          className="relative mt-4 h-[3px] overflow-hidden rounded-full bg-white/[0.08]"
          role="progressbar"
          aria-label="Separation progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(progress * 100)}
        >
          {waiting ? (
            <div className="h-full w-1/4 rounded-full bg-white/60" style={{ animation: "indeterminate 1.4s ease-in-out infinite" }} />
          ) : (
            <div className="h-full rounded-full transition-[width] duration-700 ease-out" style={{ width: `${progress * 100}%`, background: "var(--aurora)", boxShadow: "0 0 16px rgb(167 139 250 / 0.7)" }} />
          )}
        </div>

        <ol className="mt-9" data-testid="stage-list">
          {list.map((s, i) => {
            const state = uploading || currentIndex < 0 ? "todo" : i < currentIndex ? "done" : i === currentIndex ? "active" : "todo";
            const last = i === list.length - 1;
            return (
              <li key={s.name} className="relative flex items-center gap-4 py-[7px] text-sm" data-stage={s.name} data-state={state} aria-current={state === "active" ? "step" : undefined}>
                {!last && <span className={`absolute top-[calc(50%+11px)] left-[9px] h-[calc(100%-22px)] w-px ${state === "done" ? "bg-white/40" : "bg-line-strong"}`} aria-hidden />}
                <span
                  className={`grid h-[19px] w-[19px] shrink-0 place-items-center rounded-full border text-[10px] ${
                    state === "done" ? "border-transparent bg-text text-ink" : state === "active" ? "stage-node-active" : "border-line-strong text-faint"
                  }`}
                  aria-hidden
                >
                  {state === "done" ? (
                    <svg width="9" height="9" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <path d="m2 5.2 2 2L8 3" />
                    </svg>
                  ) : state === "active" ? (
                    <span className="h-1.5 w-1.5 rounded-full bg-white" />
                  ) : (
                    <span className="num font-mono">{i + 1}</span>
                  )}
                </span>
                <span className={state === "active" ? "font-medium text-text" : state === "done" ? "text-muted" : "text-faint"}>{s.label}</span>
              </li>
            );
          })}
        </ol>

        <div className="mt-9 flex items-center justify-between gap-4 border-t border-line pt-5 text-xs text-faint">
          <span className="block min-w-0 first-letter:uppercase">{note}</span>
          {!done && (
            <button type="button" onClick={onCancel} className="btn btn-ghost h-8 shrink-0 px-4 text-xs" data-testid="cancel">
              Cancel
            </button>
          )}
        </div>
      </div>

      <div className="order-1 flex justify-center lg:order-2">
        <div className="relative aspect-square w-[min(52vw,14rem)] lg:w-full">
          <SoundSphere mode={done ? "done" : "separating"} progress={progress} opacity={done ? 0.35 : 1} radius={0.41} className="absolute inset-0" />
        </div>
      </div>
    </main>
  );
}
