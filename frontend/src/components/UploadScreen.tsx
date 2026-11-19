import { useEffect, useRef, useState, type CSSProperties } from "react";
import { api } from "../api/client";
import type { Health, Mode } from "../api/types";
import { readSetting, writeSetting } from "../lib/storage";
import { UploadIcon } from "./icons";
import { Orb } from "./Orb";

const ACCEPT = ".wav,.mp3,.m4a,.webm,.ogg,.opus,.flac,.aac,.mp4,.aiff,audio/*";
const MAX_MB = 200;

export interface JobOptions {
  mode: Mode;
  splitVoices: boolean;
  lyrics: boolean;
  /** separate in this browser instead of on the server (voices vs music only) */
  local: boolean;
}

interface Props {
  onSubmit: (file: File, options: JobOptions) => void;
  /** open live mode (near-real-time voices/music over a WebSocket) */
  onLive: () => void;
  error: string | null;
}

/** The backend's health, polled while it is loading or unreachable. Shared with the progress screen. */
export function useHealth(): { health: Health | null; unreachable: boolean } {
  const [health, setHealth] = useState<Health | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const h = await api.health();
        if (cancelled) return;
        setHealth(h);
        setUnreachable(false);
        if (h.status === "loading") timer = setTimeout(poll, 2000);
      } catch {
        if (cancelled) return;
        setUnreachable(true);
        timer = setTimeout(poll, 3000);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);
  return { health, unreachable };
}

const delay = (i: number) => ({ "--i": i }) as CSSProperties;

/** Landing: headline and actions on the left, the sphere (record) on the right; a file can be dropped anywhere. */
export function UploadScreen({ onSubmit, onLive, error }: Props) {
  const [mode, setMode] = useState<Mode>(() => (readSetting("mode") === "quality" ? "quality" : "fast"));
  const [splitVoices, setSplitVoices] = useState(() => readSetting("splitVoices") === "1");
  const [lyrics, setLyrics] = useState(() => readSetting("lyrics") !== "0");
  const [local, setLocal] = useState(() => readSetting("local") === "1");
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const dragDepth = useRef(0);
  const { health, unreachable } = useHealth();
  const canSplit = health?.features?.speaker_split ?? false;

  useEffect(() => writeSetting("mode", mode), [mode]);
  useEffect(() => writeSetting("splitVoices", splitVoices ? "1" : "0"), [splitVoices]);
  useEffect(() => writeSetting("lyrics", lyrics ? "1" : "0"), [lyrics]);
  useEffect(() => writeSetting("local", local ? "1" : "0"), [local]);

  const options = (): JobOptions => ({ mode, splitVoices: splitVoices && canSplit && !local, lyrics: lyrics && !local, local });

  const submit = (file: File | undefined) => {
    if (!file) return;
    if (file.size === 0) return setLocalError("That file is empty.");
    if (file.size > MAX_MB * 1024 * 1024) return setLocalError(`That file is larger than ${MAX_MB} MB.`);
    setLocalError(null);
    onSubmit(file, options());
  };
  // The page-wide drop listeners are added once and always call the latest submit.
  const submitRef = useRef(submit);
  submitRef.current = submit;

  useEffect(() => {
    const isFiles = (e: DragEvent) => !!e.dataTransfer && [...e.dataTransfer.types].includes("Files");
    const enter = (e: DragEvent) => {
      if (!isFiles(e)) return;
      e.preventDefault();
      dragDepth.current += 1;
      setDragging(true);
    };
    const over = (e: DragEvent) => {
      if (isFiles(e)) e.preventDefault();
    };
    const leave = () => {
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (dragDepth.current === 0) setDragging(false);
    };
    const drop = (e: DragEvent) => {
      if (!isFiles(e)) return;
      e.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      submitRef.current(e.dataTransfer?.files[0]);
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragover", over);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragover", over);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", drop);
    };
  }, []);

  const shownError = localError ?? error;

  return (
    <main className="relative isolate mx-auto flex min-h-dvh w-full max-w-[1760px] flex-col overflow-x-clip px-4 pt-4 pb-5 sm:px-8 sm:pt-6 lg:px-12" data-testid="landing">
      <div className="horizon" aria-hidden />

      <header className="relative z-10 flex items-center justify-between gap-3">
        <Wordmark />
        <div className="flex min-w-0 items-center gap-2">
          <HealthPill health={health} unreachable={unreachable} />
          <button
            type="button"
            onClick={onLive}
            disabled={unreachable}
            className="btn btn-ghost h-9 shrink-0 px-4 text-[13px]"
            title="Hear your voice and the music separated while you talk (about 11 s behind on a laptop CPU)"
            data-testid="go-live"
          >
            <span className="relative flex h-2 w-2" aria-hidden>
              <span className="absolute inset-0 animate-ping rounded-full bg-danger/60" />
              <span className="relative h-2 w-2 rounded-full bg-danger" />
            </span>
            Go live
          </button>
        </div>
      </header>

      <section className="grid flex-1 items-center gap-2 py-4 lg:grid-cols-[1.08fr_1fr] lg:gap-6 lg:py-8">
        <div className="order-2 flex flex-col items-center pt-8 text-center lg:order-1 lg:items-start lg:pt-0 lg:text-left">
          <p className="eyebrow rise" style={delay(0)}>
            Source separation · runs on your machine
          </p>
          <h1 className="display mt-5 text-[clamp(2.75rem,min(6.6vw,12.5vh),7rem)] text-text">
            <span className="reveal-line">
              <span style={delay(0)}>Pull a recording</span>
            </span>
            <span className="reveal-line">
              <span style={delay(1)} className="aurora-text">
                apart by source.
              </span>
            </span>
          </h1>
          <p className="rise mt-6 max-w-[34rem] text-base leading-relaxed text-muted sm:text-lg 2xl:max-w-[40rem] 2xl:text-xl" style={delay(3)}>
            Talk over a song, or drop any clip. Neural models separate the voices from the music, the music into instruments, and write out the lyrics.
          </p>
          <div className="rise mt-8 flex flex-wrap items-center justify-center gap-x-4 gap-y-3 lg:justify-start" style={delay(4)}>
            <button type="button" onClick={() => inputRef.current?.click()} className="btn btn-primary h-12 px-6 text-[15px]" data-testid="dropzone">
              <UploadIcon size={15} />
              Choose a file
            </button>
            <span className="text-sm text-muted">or drop it anywhere</span>
          </div>
          <input ref={inputRef} type="file" accept={ACCEPT} className="hidden" data-testid="file-input" onChange={(e) => submit(e.target.files?.[0])} />

          {shownError && (
            <p className="mt-6 max-w-lg rounded-2xl border border-danger/35 bg-danger/10 px-4 py-3 text-left text-sm text-danger" role="alert" data-testid="upload-error">
              {shownError}
            </p>
          )}
        </div>

        <div className="order-1 flex justify-center pt-6 lg:order-2 lg:pt-0">
          <Orb onRecorded={(file) => submit(file)} onError={setLocalError} disabled={unreachable} className="w-[min(60vw,18rem,38vh)] sm:w-[min(46vw,24rem,40vh)] lg:w-[min(34vw,30rem,54vh)] 2xl:w-[min(30vw,40rem,56vh)]" />
        </div>
      </section>

      <footer className="rise relative z-10 flex flex-col items-center gap-3 lg:flex-row lg:justify-between" style={delay(6)}>
        <p className="eyebrow hidden text-faint lg:block">wav · mp3 · m4a · flac · ogg · webm</p>
        <div className="glass flex flex-wrap items-center justify-center gap-1 rounded-[22px] p-1.5 sm:rounded-full">
          <fieldset className="flex items-center gap-0.5 rounded-full bg-black/30 p-0.5" data-testid="mode-toggle">
            <legend className="sr-only">Separation mode</legend>
            {(
              [
                ["fast", "Fast", "Voices + drums, bass, guitar, piano, other"],
                ["quality", "Quality", "Best voice model and cleaner drums/bass (no guitar/piano). Several times slower, very slow without a GPU."],
              ] as const
            ).map(([value, label, hint]) => (
              <label
                key={value}
                title={hint}
                className={`cursor-pointer rounded-full px-4 py-1.5 text-[13px] font-medium transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-white/80 ${mode === value ? "bg-text text-ink shadow-[0_0_18px_rgb(255_255_255/0.25)]" : "text-muted hover:text-text"}`}
              >
                <input type="radio" name="mode" value={value} checked={mode === value} onChange={() => setMode(value)} className="sr-only" />
                {label}
              </label>
            ))}
          </fieldset>
          <Toggle
            checked={local}
            onChange={setLocal}
            label="In this browser"
            hint="Separate voices from music inside this browser: the audio is never uploaded. Slower, and voices vs. music only (no instruments, lyrics or voice splitting)."
            testId="opt-local"
          />
          <Toggle
            checked={splitVoices && canSplit && !local}
            onChange={setSplitVoices}
            disabled={!canSplit || local}
            label="Split voices"
            hint={
              !canSplit
                ? "Not installed on the server: run `uv sync` in backend/speakers"
                : health?.device === "cuda"
                  ? "Separate your voice from the singer (runs on the GPU)"
                  : "Separate your voice from the singer (slow on CPU: roughly 17× the clip length)"
            }
            testId="opt-split-voices"
          />
          <Toggle checked={lyrics && !local} onChange={setLyrics} disabled={local} label="Lyrics" hint="Transcribe the voices with word timing, shown in sync with playback" testId="opt-lyrics" />
        </div>
        <p className="eyebrow hidden text-faint lg:block">up to 6 min · 200 MB</p>
      </footer>

      {dragging && (
        <div className="pointer-events-none fixed inset-3 z-40 grid place-items-center rounded-[28px] border border-white/30 bg-ink/70 backdrop-blur-md" data-testid="drop-overlay">
          <div className="text-center">
            <p className="display text-[clamp(2.5rem,6vw,5rem)]">
              Drop to <span className="aurora-text">unmix</span>
            </p>
            <p className="eyebrow mt-4">Any song, voice memo or clip up to 6 minutes</p>
          </div>
        </div>
      )}
    </main>
  );
}

/** "unmix" with a dot of every source colour. */
export function Wordmark({ onClick }: { onClick?: () => void }) {
  const content = (
    <>
      <span className="font-display text-[19px] font-semibold tracking-[-0.04em] text-text">unmix</span>
      <span className="ml-1.5 h-1.5 w-6 rounded-full" style={{ background: "var(--aurora)" }} aria-hidden />
    </>
  );
  return onClick ? (
    <button type="button" onClick={onClick} className="flex shrink-0 items-center rounded-md" title="Back to the start">
      {content}
    </button>
  ) : (
    <p className="flex shrink-0 items-center">{content}</p>
  );
}

function Toggle({ checked, onChange, label, hint, disabled = false, testId }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint: string; disabled?: boolean; testId: string }) {
  return (
    <label
      title={hint}
      className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-[13px] transition-colors ${disabled ? "cursor-not-allowed text-faint" : checked ? "cursor-pointer text-text" : "cursor-pointer text-muted hover:text-text"}`}
    >
      <input type="checkbox" className="peer sr-only" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} data-testid={testId} />
      <span className="switch-track shrink-0" aria-hidden />
      {label}
    </label>
  );
}

/** " · GPU (Tesla T4) · remote": what the server runs on. */
function where(health: Health): string {
  const summary = health.accelerator?.summary ?? health.device.toUpperCase();
  return ` · ${summary}${health.remote ? " · remote" : ""}`;
}

function HealthPill({ health, unreachable }: { health: Health | null; unreachable: boolean }) {
  let dot = "bg-faint";
  let text = "Connecting to backend…";
  if (unreachable) {
    dot = "bg-danger";
    text = "Backend not reachable";
  } else if (health?.status === "error") {
    dot = "bg-danger";
    text = `Models failed to load: ${health.models_error ?? "unknown error"}`;
  } else if (health?.status === "loading") {
    dot = "bg-warn animate-pulse";
    text = `Loading models${where(health)}…`;
  } else if (health) {
    dot = "bg-ok";
    text = `Ready${where(health)}`;
  }
  const notes = health?.accelerator?.notes ?? [];
  return (
    <p
      className="glass flex h-9 min-w-0 items-center gap-2 rounded-full px-3.5 text-xs text-muted"
      data-testid="health"
      data-status={unreachable ? "unreachable" : (health?.status ?? "unknown")}
      title={[text, ...notes].join("\n")}
    >
      <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${dot}`} aria-hidden />
      <span className="truncate">{text}</span>
    </p>
  );
}
