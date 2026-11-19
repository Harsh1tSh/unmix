import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api/client";
import type { Analysis, Job, JobEvent, Lyrics, StageInfo, Stem } from "./api/types";
import { ProgressScreen } from "./components/ProgressScreen";
import { prefersReducedMotion, type StrandRects } from "./components/Split";
import { StemBoard } from "./components/StemBoard";
import { LiveScreen } from "./components/LiveScreen";
import { UploadScreen, type JobOptions } from "./components/UploadScreen";
import { peaksFromFile } from "./lib/wave";
import { LOCAL_STAGES, runLocal } from "./local/runLocal";
import { SpaceBackdrop } from "./visual/SpaceBackdrop";

/**
 * Screen state machine:
 *   idle -> uploading -> processing -> splitting -> ready
 *              \            \-> idle (with error)
 * "splitting" is the progress screen playing the split animation with the
 * finished job's data; the board then continues it (FlyIn) from `intro`.
 * The job id lives in the URL hash (#job=<id>) so a refresh restores the board.
 */
type Results = { job: Job; stems: Stem[]; analysis: Analysis; lyrics: Lyrics | null };
type Screen =
  | { name: "idle"; error: string | null }
  | { name: "live" }
  | { name: "uploading"; filename: string; progress: number }
  | { name: "processing"; jobId: string; filename: string; event: JobEvent | null; stages: StageInfo[] | null }
  | ({ name: "splitting"; filename: string; stages: StageInfo[] | null } & Results)
  | ({ name: "ready"; intro: StrandRects | null } & Results);

const HASH_RE = /^#job=([0-9a-f]{32})$/;

function jobIdFromHash(): string | null {
  return HASH_RE.exec(window.location.hash)?.[1] ?? null;
}

function setHash(jobId: string | null): void {
  const url = jobId ? `#job=${jobId}` : window.location.pathname + window.location.search;
  window.history.replaceState(null, "", url);
}

const message = (err: unknown): string => (err instanceof ApiError ? err.message : err instanceof Error ? err.message : "Something went wrong.");

async function loadResults(jobId: string): Promise<Results> {
  const [job, analysis] = await Promise.all([api.getJob(jobId), api.getAnalysis(jobId)]);
  if (job.status !== "done" || !job.stems?.length) throw new Error(job.error ?? "The job finished without any stems.");
  // Lyrics are optional: a failure here must not hide the stems.
  const lyrics = job.lyrics_url ? await api.getLyrics(job.lyrics_url).catch(() => null) : null;
  return { job, stems: job.stems, analysis, lyrics };
}

export function App() {
  const [screen, setScreen] = useState<Screen>({ name: "idle", error: null });
  const [mixPeaks, setMixPeaks] = useState<number[] | null>(null);
  const unwatch = useRef<(() => void) | null>(null);
  // Browser-only jobs: cancel flag for the running one, cleanup for the shown one.
  const localRun = useRef<{ cancelled: boolean } | null>(null);
  const localDispose = useRef<(() => void) | null>(null);
  // Bumped whenever the user starts over, so late async results are ignored.
  const session = useRef(0);

  const fail = useCallback((error: string) => {
    unwatch.current?.();
    unwatch.current = null;
    setHash(null);
    setScreen({ name: "idle", error });
  }, []);

  const showResults = useCallback(
    async (jobId: string, token: number, animate: boolean) => {
      try {
        const results = await loadResults(jobId);
        if (token !== session.current) return;
        setScreen((s) =>
          animate && s.name === "processing" && !prefersReducedMotion()
            ? { name: "splitting", filename: s.filename, stages: s.stages, ...results }
            : { name: "ready", intro: null, ...results },
        );
      } catch (err) {
        if (token === session.current) fail(message(err));
      }
    },
    [fail],
  );

  const watch = useCallback(
    (jobId: string, filename: string, token: number) => {
      setHash(jobId);
      setScreen({ name: "processing", jobId, filename, event: null, stages: null });
      api.getJob(jobId).then(
        (job) => token === session.current && setScreen((s) => (s.name === "processing" && s.jobId === jobId ? { ...s, stages: job.stages } : s)),
        () => undefined,
      );
      unwatch.current?.();
      unwatch.current = api.watchJob(
        jobId,
        (event) => {
          if (token !== session.current) return;
          if (event.status === "done") void showResults(jobId, token, true);
          else if (event.status === "error") fail(event.error ?? event.message);
          else setScreen((s) => (s.name === "processing" && s.jobId === jobId ? { ...s, event } : s));
        },
        (err) => {
          if (token === session.current) fail(err.message);
        },
      );
    },
    [fail, showResults],
  );

  const startLocal = useCallback(
    async (file: File, token: number) => {
      const signal = { cancelled: false };
      localRun.current = signal;
      setScreen({ name: "processing", jobId: "local", filename: file.name, event: null, stages: LOCAL_STAGES });
      try {
        const result = await runLocal(
          file,
          (event) => token === session.current && setScreen((s) => (s.name === "processing" && s.jobId === "local" ? { ...s, event } : s)),
          signal,
        );
        if (token !== session.current) return result.dispose();
        localDispose.current = result.dispose;
        const results = { job: result.job, stems: result.stems, analysis: result.analysis, lyrics: null };
        setScreen(prefersReducedMotion() ? { name: "ready", intro: null, ...results } : { name: "splitting", filename: file.name, stages: LOCAL_STAGES, ...results });
      } catch (err) {
        if (token === session.current && !signal.cancelled) fail(message(err));
      }
    },
    [fail],
  );

  const start = useCallback(
    async (file: File, options: JobOptions) => {
      const token = ++session.current;
      setMixPeaks(null);
      // Decode locally for the progress waveform; independent of the upload.
      peaksFromFile(file).then(
        (p) => token === session.current && setMixPeaks(p),
        () => undefined,
      );
      if (options.local) return startLocal(file, token);
      setScreen({ name: "uploading", filename: file.name, progress: 0 });
      try {
        const { job_id } = await api.createJob(file, file.name, {
          mode: options.mode,
          splitVoices: options.splitVoices,
          lyrics: options.lyrics,
          onUploadProgress: (progress) => {
            if (token === session.current) setScreen((s) => (s.name === "uploading" ? { ...s, progress } : s));
          },
        });
        if (token !== session.current) {
          void api.deleteJob(job_id).catch(() => undefined); // cancelled while uploading
          return;
        }
        watch(job_id, file.name, token);
      } catch (err) {
        if (token === session.current) fail(message(err));
      }
    },
    [fail, watch, startLocal],
  );

  const reset = useCallback((deleteJobId?: string) => {
    session.current += 1;
    if (localRun.current) localRun.current.cancelled = true;
    localRun.current = null;
    localDispose.current?.();
    localDispose.current = null;
    unwatch.current?.();
    unwatch.current = null;
    if (deleteJobId && !deleteJobId.startsWith("local")) void api.deleteJob(deleteJobId).catch(() => undefined);
    setHash(null);
    setMixPeaks(null);
    setScreen({ name: "idle", error: null });
  }, []);

  // Restore a job from the URL on first load.
  useEffect(() => {
    const jobId = jobIdFromHash();
    if (!jobId) return;
    const token = ++session.current;
    api.getJob(jobId).then(
      (job) => {
        if (token !== session.current) return;
        if (job.status === "done") void showResults(jobId, token, false);
        else if (job.status === "error") fail(job.error ?? "That job failed.");
        else watch(jobId, job.filename || "your recording", token);
      },
      () => {
        if (token === session.current) fail("That job no longer exists (results are kept for 24 hours).");
      },
    );
    return () => {
      unwatch.current?.();
      unwatch.current = null;
    };
    // Run once on mount; the callbacks are stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  let content;
  switch (screen.name) {
    case "idle":
      content = <UploadScreen onSubmit={(file, options) => void start(file, options)} onLive={() => setScreen({ name: "live" })} error={screen.error} />;
      break;
    case "live":
      content = (
        <LiveScreen
          onExit={() => setScreen({ name: "idle", error: null })}
          onUnmixFully={(file) => void start(file, { mode: "fast", splitVoices: false, lyrics: true, local: false })}
        />
      );
      break;
    case "uploading":
      content = (
        <ProgressScreen filename={screen.filename} uploadProgress={screen.progress} event={null} stages={null} mixPeaks={mixPeaks} split={null} onSplitDone={() => undefined} onCancel={() => reset()} />
      );
      break;
    case "processing":
      content = (
        <ProgressScreen
          filename={screen.filename}
          uploadProgress={null}
          event={screen.event}
          stages={screen.stages}
          mixPeaks={mixPeaks}
          split={null}
          onSplitDone={() => undefined}
          onCancel={() => reset(screen.jobId)}
          inBrowser={screen.jobId === "local"}
        />
      );
      break;
    case "splitting": {
      const { job, stems, analysis, lyrics } = screen;
      content = (
        <ProgressScreen
          filename={screen.filename}
          uploadProgress={null}
          event={null}
          stages={screen.stages}
          mixPeaks={mixPeaks}
          split={{ stems, analysis }}
          onSplitDone={(rects) => setScreen({ name: "ready", intro: rects, job, stems, analysis, lyrics })}
          onCancel={() => reset()}
          inBrowser={job.id.startsWith("local")}
        />
      );
      break;
    }
    case "ready":
      content = (
        <StemBoard
          job={screen.job}
          stems={screen.stems}
          analysis={screen.analysis}
          lyrics={screen.lyrics}
          intro={screen.intro}
          onReset={() => reset()}
          onDelete={() => reset(screen.job.id)}
        />
      );
      break;
  }

  return (
    <>
      <div className="atmosphere" aria-hidden />
      {/* the sky (and the sphere) behind every screen; still on the board, where you listen and read meters */}
      <SpaceBackdrop mode={screen.name === "ready" ? "still" : "animated"} />
      {content}
      <div className="grain" aria-hidden />
    </>
  );
}
