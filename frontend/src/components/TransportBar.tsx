import { useEffect, useRef, type CSSProperties, type ReactNode } from "react";
import type { EngineSnapshot, StemEngine } from "../audio/engine";
import { formatTime } from "../lib/format";
import { BackIcon, ForwardIcon, LoopIcon, PauseIcon, PlayIcon, VolumeIcon } from "./icons";

interface Props {
  engine: StemEngine;
  snapshot: EngineSnapshot;
  /** stem name -> what the board calls it (e.g. "You" after "That's me") */
  labels?: Record<string, string>;
}

const SKIP_S = 5;

/**
 * Shared transport: one play head for every stem, in a floating dock.
 * Keys: Space play/pause · ←/→ skip 5 s · L loop · Home back to start.
 */
export function TransportBar({ engine, snapshot, labels = {} }: Props) {
  const seekRef = useRef<HTMLInputElement | null>(null);
  const timeRef = useRef<HTMLSpanElement | null>(null);
  const dragging = useRef(false);

  // Position updates bypass React: write straight to the DOM each frame.
  useEffect(
    () =>
      engine.onFrame((pos) => {
        const duration = engine.getSnapshot().duration;
        if (timeRef.current) timeRef.current.textContent = formatTime(pos);
        const input = seekRef.current;
        if (input && !dragging.current) {
          input.value = String(pos);
          input.style.setProperty("--fill", `${duration > 0 ? (pos / duration) * 100 : 0}%`);
        }
      }),
    [engine],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (e.metaKey || e.ctrlKey || e.altKey || tag === "INPUT" || tag === "TEXTAREA" || target?.isContentEditable) return;
      if (e.code === "Space" && tag !== "BUTTON" && tag !== "A") {
        if (e.repeat) return;
        e.preventDefault();
        engine.toggle();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        engine.skip(-SKIP_S);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        engine.skip(SKIP_S);
      } else if (e.code === "KeyL") {
        engine.setLoop(!engine.getSnapshot().loop);
      } else if (e.code === "Home") {
        engine.seek(0);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [engine]);

  const reallyPlaying = snapshot.playing && !snapshot.preview;
  const hoverLabel = snapshot.hover ? (labels[snapshot.hover] ?? snapshot.hover.replace("_", " ")) : null;
  const status = snapshot.audioBlocked ? "Click play once to enable audio" : hoverLabel ? `Listening to ${hoverLabel} only` : "Hover a card to hear it alone";

  return (
    <div className="fixed inset-x-3 bottom-3 z-20 sm:bottom-5" data-testid="transport">
      <div className="glass-solid mx-auto flex max-w-4xl 2xl:max-w-5xl items-center gap-2 rounded-full py-2 pr-3 pl-2 sm:gap-3 sm:pr-5">
        <button
          type="button"
          onClick={() => engine.toggle()}
          className="btn btn-primary h-11 w-11 shrink-0 rounded-full p-0"
          aria-label={reallyPlaying ? "Pause (Space)" : "Play (Space)"}
          title={reallyPlaying ? "Pause (Space)" : "Play (Space)"}
          data-testid="play-toggle"
          data-playing={snapshot.playing}
        >
          {reallyPlaying ? <PauseIcon /> : <PlayIcon className="translate-x-px" />}
        </button>
        <IconButton label={`Back ${SKIP_S} seconds (←)`} onClick={() => engine.skip(-SKIP_S)} className="hidden sm:grid">
          <BackIcon />
        </IconButton>
        <IconButton label={`Forward ${SKIP_S} seconds (→)`} onClick={() => engine.skip(SKIP_S)} className="hidden sm:grid">
          <ForwardIcon />
        </IconButton>

        <span ref={timeRef} className="num w-9 shrink-0 text-right font-mono text-[11px] text-text sm:ml-1" data-testid="time-current">
          0:00
        </span>
        <input
          ref={seekRef}
          type="range"
          min={0}
          max={snapshot.duration || 1}
          step={0.01}
          defaultValue={0}
          aria-label="Seek"
          className="min-w-0 flex-1"
          style={{ "--fill": "0%" } as CSSProperties}
          data-testid="seek"
          onPointerDown={() => (dragging.current = true)}
          onPointerUp={() => (dragging.current = false)}
          onPointerCancel={() => (dragging.current = false)}
          onBlur={() => (dragging.current = false)}
          onChange={(e) => {
            const value = Number(e.target.value);
            e.target.style.setProperty("--fill", `${snapshot.duration > 0 ? (value / snapshot.duration) * 100 : 0}%`);
            engine.seek(value);
          }}
        />
        <span className="num w-9 shrink-0 font-mono text-[11px] text-muted" data-testid="time-total">
          {formatTime(snapshot.duration)}
        </span>

        <IconButton label="Loop (L)" onClick={() => engine.setLoop(!snapshot.loop)} active={snapshot.loop} testId="loop">
          <LoopIcon />
        </IconButton>
        <div className="hidden items-center gap-2 md:flex" title="Volume">
          <button
            type="button"
            className="text-muted transition-colors hover:text-text"
            aria-label={snapshot.masterVolume === 0 ? "Unmute" : "Mute all"}
            onClick={() => engine.setMasterVolume(snapshot.masterVolume === 0 ? 1 : 0)}
          >
            <VolumeIcon muted={snapshot.masterVolume === 0} />
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={snapshot.masterVolume}
            onChange={(e) => engine.setMasterVolume(Number(e.target.value))}
            aria-label="Volume"
            className="w-20"
            style={{ "--fill": `${snapshot.masterVolume * 100}%` } as CSSProperties}
            data-testid="master-volume"
          />
        </div>

        <span className="hidden w-44 shrink-0 truncate text-right text-xs text-muted lg:block" aria-live="polite">
          {status}
        </span>
      </div>
    </div>
  );
}

function IconButton({ label, onClick, children, active = false, className = "grid", testId }: { label: string; onClick: () => void; children: ReactNode; active?: boolean; className?: string; testId?: string }) {
  return (
    <button type="button" onClick={onClick} aria-label={label} aria-pressed={active} title={label} data-testid={testId} className={`btn-icon h-9 w-9 ${className}`}>
      {children}
    </button>
  );
}
