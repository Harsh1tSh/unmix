import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import type { LyricSegment, Lyrics } from "../api/types";
import type { StemEngine } from "../audio/engine";
import { locate, wordProgress } from "../lib/lyrics";
import { downloadText, toLRC, toSRT } from "../lib/lyricsExport";

interface Props {
  lyrics: Lyrics;
  engine: StemEngine;
  /** stem name -> colour, to tint words by who sings/speaks them */
  colors: Record<string, string>;
  /** stem name -> label, for the lane headings */
  labels: Record<string, string>;
  fallbackColor: string;
  /** used to name exported files */
  filename: string;
}

/**
 * Karaoke-style lyrics synced to the shared transport.
 *
 * When the voices were split, each speaker gets their own lane: you and the
 * singer can be heard at the same moment, and their lines overlap in time.
 * Without a split there is one lane with the previous and upcoming lines.
 */
export function LyricsPanel({ lyrics, engine, colors, labels, fallbackColor, filename }: Props) {
  const lanes = useMemo(() => {
    const bySpeaker = new Map<string, LyricSegment[]>();
    for (const seg of lyrics.segments) {
      const key = seg.speaker ?? "";
      if (!bySpeaker.has(key)) bySpeaker.set(key, []);
      bySpeaker.get(key)!.push(seg);
    }
    return [...bySpeaker.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [lyrics]);

  if (lyrics.segments.length === 0) {
    return (
      <p className="mt-4 rounded-2xl border border-line px-4 py-3 text-sm text-muted" data-testid="lyrics-empty">
        {lyrics.note ?? "No lyrics were found."}
      </p>
    );
  }

  const multi = lanes.length > 1;
  const base = filename.replace(/\.[^.]+$/, "") || "lyrics";
  return (
    <section className="mt-4 rounded-2xl border border-line bg-black/30 px-4 py-4 sm:px-5" aria-label="Lyrics" data-testid="lyrics">
      <header className="flex items-center justify-between gap-3 text-[11px] text-faint">
        <span className="eyebrow text-[10px]">Lyrics</span>
        <span className="flex items-center gap-2">
          <span className="hidden sm:inline" title={`Transcribed locally with ${lyrics.model}`}>
            auto-transcribed
          </span>
          {(
            [
              ["LRC", "Synced lyrics for karaoke apps and music players", () => downloadText(toLRC(lyrics, labels, base), `${base}.lrc`)],
              ["SRT", "Subtitles with start and end times", () => downloadText(toSRT(lyrics, labels), `${base}.srt`, "application/x-subrip")],
            ] as const
          ).map(([fmt, hint, run]) => (
            <button key={fmt} type="button" onClick={run} title={`Download .${fmt.toLowerCase()}: ${hint}`} className="rounded-full border border-line px-2 py-0.5 font-mono text-[10px] text-muted transition-colors hover:border-line-strong hover:text-text" data-testid={`export-${fmt.toLowerCase()}`}>
              .{fmt.toLowerCase()}
            </button>
          ))}
        </span>
      </header>
      <div className={multi ? "mt-2 space-y-3" : "mt-2"}>
        {lanes.map(([speaker, segments]) => (
          <Lane
            key={speaker || "all"}
            segments={segments}
            engine={engine}
            color={(speaker && colors[speaker]) || fallbackColor}
            label={speaker ? (labels[speaker] ?? speaker) : null}
            compact={multi}
            speaker={speaker || null}
          />
        ))}
      </div>
    </section>
  );
}

/**
 * One speaker's lines. React re-renders only when the current line changes;
 * the per-word fill runs every animation frame by writing a CSS variable on
 * the word spans directly. Clicking a word seeks there.
 */
function Lane({ segments, engine, color, label, compact, speaker }: { segments: LyricSegment[]; engine: StemEngine; color: string; label: string | null; compact: boolean; speaker: string | null }) {
  const [current, setCurrent] = useState(0);
  const [active, setActive] = useState(false);
  const lineRef = useRef<HTMLParagraphElement | null>(null);
  const state = useRef({ current: 0, active: false });

  // Paint the karaoke fill of the rendered line for playhead time t.
  const paint = useCallback(
    (t: number) => {
      const line = lineRef.current;
      const seg = segments[state.current.current];
      if (!line || !seg) return;
      line.querySelectorAll<HTMLSpanElement>("[data-word]").forEach((span, i) => {
        const word = seg.words[i];
        const fill = word ? wordProgress(word.start, word.end, t) : 0;
        span.style.setProperty("--fill", `${Math.round(fill * 100)}%`);
        span.dataset.state = fill >= 1 ? "sung" : fill > 0 ? "singing" : "upcoming";
      });
    },
    [segments],
  );

  useEffect(
    () =>
      engine.onFrame((t) => {
        const pos = locate(segments, t);
        if (pos.segment >= 0 && (pos.segment !== state.current.current || pos.inside !== state.current.active)) {
          state.current = { current: pos.segment, active: pos.inside };
          setCurrent(pos.segment);
          setActive(pos.inside);
        }
        paint(t);
      }),
    [engine, segments, paint],
  );

  // A new line has just rendered: fill it right away. Needed when paused,
  // since no further frames arrive after a seek to a different line.
  useLayoutEffect(() => paint(engine.position()), [current, engine, paint]);

  const seg = segments[current]!;
  const previous = !compact && current > 0 ? segments[current - 1] : undefined;
  const upcoming = segments.slice(current + 1, current + 1 + (compact ? 1 : 2));

  return (
    <div data-testid="lyrics-lane" data-speaker={speaker ?? ""} className={compact ? "transition-opacity duration-300" : ""} style={compact ? { opacity: active ? 1 : 0.55 } : undefined}>
      {label && (
        <p className="flex items-center gap-1.5 text-[11px] text-muted">
          <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} aria-hidden />
          {label}
        </p>
      )}
      {previous && <p className="truncate text-sm text-faint">{previous.text}</p>}
      <p
        ref={lineRef}
        key={current}
        className={`mt-1 font-display leading-snug font-medium tracking-[-0.02em] ${compact ? "text-lg sm:text-xl" : "text-xl sm:text-[1.6rem]"}`}
        data-testid="lyrics-current"
        data-segment={current}
      >
        {seg.words.map((w, i) => (
          <span
            key={i}
            data-word={i}
            data-state="upcoming"
            onClick={() => engine.seek(w.start)}
            className="lyric-word cursor-pointer"
            style={{ "--word": color, "--fill": "0%" } as CSSProperties}
            title={`${w.start.toFixed(1)} s${w.probability < 0.5 ? " · low confidence" : ""}`}
          >
            {w.text}
          </span>
        ))}
      </p>
      {upcoming.map((u) => (
        <p key={u.start} className="mt-1 truncate text-sm text-muted">
          <button type="button" className="text-left hover:text-text" onClick={() => engine.seek(u.start)}>
            {u.text}
          </button>
        </p>
      ))}
    </div>
  );
}
