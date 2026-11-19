import { useCallback, useEffect, useRef, useState } from "react";
import { MAX_RECORD_S, MicRecorder, RecorderError } from "../audio/recorder";
import { formatTime } from "../lib/format";
import { SoundSphere, type LiveLevels } from "../visual/SoundSphere";

const BAND_EDGES = [20, 60, 250, 500, 2000, 4000, 6000, 20000];
const HOLD_MS = 350; // pressed longer than this = hold-to-record; shorter = tap toggles

interface Props {
  onRecorded: (file: File) => void;
  onError: (message: string) => void;
  disabled?: boolean;
  className?: string;
}

type Mode = "idle" | "starting" | "recording" | "finishing";

/**
 * The landing sphere is the record button.
 *   - hold it to record, release to send
 *   - or tap once to start and tap again to stop (easier for long takes)
 * While recording, the sphere swells with the mic's loudness and each of its
 * seven latitude bands lights up with that frequency band's level.
 */
export function Orb({ onRecorded, onError, disabled = false, className = "" }: Props) {
  const recorder = useRef<MicRecorder | null>(null);
  const pressStart = useRef(0);
  const holdMode = useRef(false);
  const modeRef = useRef<Mode>("idle");
  const [mode, setModeState] = useState<Mode>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [hover, setHover] = useState(false);
  const [pressed, setPressed] = useState(false);
  const supported = MicRecorder.supported();

  const setMode = (m: Mode) => {
    modeRef.current = m;
    setModeState(m);
  };

  const finish = useCallback(async () => {
    const rec = recorder.current;
    if (!rec || modeRef.current !== "recording") return;
    setMode("finishing");
    try {
      const result = await rec.stop();
      if (result) onRecorded(result.file);
    } catch (err) {
      onError(err instanceof Error ? err.message : "Recording failed.");
    } finally {
      recorder.current = null;
      setMode("idle");
      setElapsed(0);
    }
  }, [onError, onRecorded]);

  const begin = useCallback(async () => {
    if (modeRef.current !== "idle" || disabled) return;
    setMode("starting");
    const rec = new MicRecorder();
    try {
      await rec.start();
      recorder.current = rec;
      setMode("recording");
      // Released before the mic finished opening: that was a quick hold, keep recording in tap mode.
    } catch (err) {
      rec.cancel();
      setMode("idle");
      onError(err instanceof RecorderError ? err.message : "Could not start recording.");
    }
  }, [disabled, onError]);

  // Elapsed time and the length limit, while recording.
  useEffect(() => {
    if (mode !== "recording") return;
    const timer = setInterval(() => {
      const t = recorder.current?.elapsed() ?? 0;
      setElapsed(t);
      if (t >= MAX_RECORD_S) void finish();
    }, 250);
    return () => clearInterval(timer);
  }, [mode, finish]);

  useEffect(() => () => recorder.current?.cancel(), []);

  const live = useCallback((): LiveLevels | null => {
    const rec = recorder.current;
    return rec && modeRef.current === "recording" ? rec.levels(BAND_EDGES) : null;
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0 || disabled) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setPressed(true);
    pressStart.current = performance.now();
    if (modeRef.current === "recording") {
      holdMode.current = false;
      void finish(); // second tap in tap mode
      return;
    }
    holdMode.current = true;
    void begin();
  };

  const onPointerUp = () => {
    setPressed(false);
    if (!holdMode.current) return;
    holdMode.current = false;
    const held = performance.now() - pressStart.current;
    if (held >= HOLD_MS) void finish(); // hold-to-record: release sends
    // shorter press: tap mode, keeps recording until the next tap
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    e.preventDefault();
    if (modeRef.current === "recording") void finish();
    else void begin();
  };

  const recording = mode === "recording";
  const label = recording
    ? `Recording ${formatTime(elapsed)} · release or tap to unmix`
    : mode === "starting"
      ? "Opening the microphone…"
      : mode === "finishing"
        ? "Preparing the recording…"
        : supported
          ? "Hold to record · tap to start and stop"
          : "Recording is not supported in this browser";

  return (
    <div className={`flex flex-col items-center ${className}`}>
      <button
        type="button"
        className="orb-button group relative aspect-square w-full touch-none rounded-full select-none disabled:cursor-not-allowed"
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerEnter={(e) => e.pointerType === "mouse" && setHover(true)}
        onPointerLeave={() => {
          setHover(false);
          setPressed(false);
        }}
        onKeyDown={onKeyDown}
        disabled={!supported || disabled}
        aria-label={recording ? "Stop recording and unmix" : "Record from the microphone"}
        aria-pressed={recording}
        data-testid="orb"
        data-mode={mode}
      >
        <SoundSphere mode={recording ? "recording" : "idle"} live={live} hover={hover && !disabled} pressed={pressed} opacity={!supported || disabled ? 0.4 : 1} className="absolute inset-0" />
      </button>
      <p className="mt-9 flex min-h-5 max-w-[20rem] items-center justify-center gap-2 text-center font-mono text-[11px] leading-relaxed tracking-[0.18em] text-muted uppercase sm:mt-12 sm:max-w-none" aria-live="polite" data-testid="orb-status">
        {recording && <span className="h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-danger" aria-hidden />}
        <span>{label}</span>
      </p>
    </div>
  );
}
