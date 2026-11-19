import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { CleanState } from "../api/types";
import type { StemEngine } from "../audio/engine";
import { DownloadIcon } from "./icons";

interface Props {
  jobId: string;
  stem: string;
  label: string;
  engine: StemEngine;
}

/**
 * "Clean up": removes background noise and room sound from a voice
 * (MossFormer2 SE 48K on the server). The result is a separate download; the
 * stem itself is unchanged so the mix still adds up.
 */
export function CleanVoice({ jobId, stem, label, engine }: Props) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // Never two players at once: the main transport (or a hover preview) pauses this one.
  useEffect(
    () =>
      engine.subscribe(() => {
        if (engine.getSnapshot().playing) audioRef.current?.pause();
      }),
    [engine],
  );
  const [state, setState] = useState<CleanState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const poll = (delay = 1500) => {
    clearTimeout(timer.current);
    timer.current = setTimeout(async () => {
      try {
        const s = await api.getClean(jobId, stem);
        setState(s);
        if (s.status === "queued" || s.status === "running") poll();
      } catch (err) {
        setError(err instanceof ApiError ? err.message : "Lost track of the clean-up.");
      }
    }, delay);
  };

  // A clean-up requested earlier (or still running) shows up again after a reload.
  useEffect(() => {
    let alive = true;
    api.getClean(jobId, stem).then(
      (s) => {
        if (!alive) return;
        setState(s);
        if (s.status === "queued" || s.status === "running") poll();
      },
      () => undefined, // 404 = never requested
    );
    return () => {
      alive = false;
      clearTimeout(timer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId, stem]);

  const start = async () => {
    setError(null);
    try {
      const s = await api.requestClean(jobId, stem);
      setState(s);
      if (s.status !== "done") poll(800);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not start the clean-up.");
    }
  };

  const busy = state?.status === "queued" || state?.status === "running";
  return (
    <div className="mt-4 rounded-2xl border border-line bg-white/[0.015] p-2 text-xs" data-testid={`clean-${stem}`} data-status={state?.status ?? "idle"}>
      {state?.status === "done" && state.url ? (
        <div className="flex flex-wrap items-center gap-2 pl-2">
          <span className="eyebrow text-[10px]">Cleaned</span>
          {/* Plays on its own, outside the synced mix: it is a different signal from the stem. */}
          <audio ref={audioRef} onPlay={() => engine.pause()} controls preload="none" src={state.url} className="h-8 min-w-0 flex-1" aria-label={`${label}, cleaned up`} />
          <a href={state.url} download className="btn btn-ghost h-8 px-3 text-xs" title="Download the cleaned voice (48 kHz WAV)">
            <DownloadIcon size={12} />
            WAV
          </a>
        </div>
      ) : (
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={start}
            disabled={busy}
            className="btn btn-ghost h-8 shrink-0 px-3.5 text-xs disabled:cursor-wait"
            title="Remove background noise and room sound from this voice (about 2× its length on a CPU)"
          >
            {busy ? (state?.status === "queued" ? "Waiting…" : <span className="num">Cleaning… {Math.round((state?.progress ?? 0) * 100)}%</span>) : state?.status === "error" ? "Try clean-up again" : "Clean up this voice"}
          </button>
          <span className={`min-w-0 truncate text-faint ${error || state?.status === "error" ? "" : "hidden sm:inline"}`}>{error ?? (state?.status === "error" ? state.error : "noise and room sound removed, as a separate file")}</span>
        </div>
      )}
    </div>
  );
}
