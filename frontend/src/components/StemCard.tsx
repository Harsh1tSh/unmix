import { useEffect, useMemo, useRef, type CSSProperties, type ReactNode } from "react";
import { api } from "../api/client";
import type { Stem, StemAnalysis } from "../api/types";
import type { EngineSnapshot, StemEngine } from "../audio/engine";
import { DEFAULT_CONTROL, isAudible } from "../audio/mixer";
import { formatHz, formatPercent } from "../lib/format";
import { BandMeter } from "./BandMeter";
import { DownloadIcon, HeadphonesIcon, VolumeIcon } from "./icons";
import { Waveform } from "./Waveform";

export interface StemCardProps {
  stem: Stem;
  analysis: StemAnalysis;
  frameRate: number;
  engine: StemEngine;
  snapshot: EngineSnapshot;
  /** shared waveform scale so loud and quiet stems are visually comparable */
  peakScale: number;
  /** "lg" for top-level cards, "sm" for instrument cards */
  size?: "lg" | "sm";
  /** hide the waveform (while the split animation is still landing it) */
  waveHidden?: boolean;
  /** overrides stem.label (e.g. "You" after "That's me") */
  displayLabel?: string;
  /** small control in the header, e.g. the "That's me" toggle */
  headerAction?: ReactNode;
  /** e.g. the expand/collapse control of a parent card */
  footer?: ReactNode;
  /**
   * Extra visual layers, rendered inside the card's stacking context:
   * `backdrop` behind the content, `overlay` above it. The card exposes --stem
   * (accent colour), --mx/--my (cursor position, 0..100%) and the
   * data-hovered / data-audible attributes for styling.
   */
  backdrop?: ReactNode;
  overlay?: ReactNode;
  className?: string;
  style?: CSSProperties;
}

/** Stems below this share of the mix are marked as barely present. */
const QUIET_SHARE = 0.01;

/** "strongest 250–500 Hz": the band with the most energy on average. */
function describe(analysis: StemAnalysis): string {
  const { frames, names, hz_edges } = analysis.bands;
  if (!frames.length) return "";
  const sums = names.map((_, b) => frames.reduce((acc, f) => acc + (f[b] ?? 0), 0));
  const best = sums.indexOf(Math.max(...sums));
  return `strongest ${formatHz(hz_edges[best] ?? 0)}–${formatHz(hz_edges[best + 1] ?? 0)} Hz`;
}

export function StemCard({ stem, analysis, frameRate, engine, snapshot, peakScale, size = "lg", waveHidden = false, displayLabel, headerAction, footer, backdrop, overlay, className = "", style }: StemCardProps) {
  const control = snapshot.controls[stem.name] ?? DEFAULT_CONTROL;
  const audible = isAudible(engine.tree, snapshot.gains, stem.name);
  const hovered = snapshot.hover === stem.name;
  const specHead = useRef<HTMLDivElement | null>(null);
  const large = size === "lg";
  const label = displayLabel ?? stem.label;
  const quiet = analysis.energy_share < QUIET_SHARE;
  // Sums every analysis frame: once per stem, not on every hover/play re-render.
  const description = useMemo(() => (quiet ? "Barely present in this recording" : describe(analysis)), [analysis, quiet]);

  useEffect(
    () =>
      engine.onFrame((pos) => {
        const d = engine.getSnapshot().duration;
        if (specHead.current) specHead.current.style.left = `${d > 0 ? Math.min(1, pos / d) * 100 : 0}%`;
      }),
    [engine],
  );

  // Hover = listen applies to real pointers only; touch users get the Solo button.
  const onEnter = (e: React.PointerEvent) => {
    if (e.pointerType === "mouse") engine.hoverStart(stem.name);
  };
  const onLeave = (e: React.PointerEvent<HTMLElement>) => {
    if (e.pointerType === "mouse") engine.hoverEnd(stem.name);
    e.currentTarget.style.setProperty("--rx", "0deg");
    e.currentTarget.style.setProperty("--ry", "0deg");
  };
  // Tilt toward the cursor and move the glow with it. Written straight to CSS
  // variables (no React state), so it costs nothing per frame. The tilt is
  // disabled for prefers-reduced-motion in index.css.
  const onMove = (e: React.PointerEvent<HTMLElement>) => {
    if (e.pointerType !== "mouse") return;
    const el = e.currentTarget;
    const r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width;
    const py = (e.clientY - r.top) / r.height;
    const max = large ? 2.5 : 4; // degrees; big cards tilt less
    el.style.setProperty("--rx", `${((0.5 - py) * max).toFixed(2)}deg`);
    el.style.setProperty("--ry", `${((px - 0.5) * max).toFixed(2)}deg`);
    el.style.setProperty("--mx", `${(px * 100).toFixed(1)}%`);
    el.style.setProperty("--my", `${(py * 100).toFixed(1)}%`);
  };

  return (
    <article
      className={`stem-card relative isolate overflow-hidden rounded-3xl ${large ? "p-5 sm:p-7" : "p-4 sm:p-5"} ${className}`}
      style={{ "--stem": stem.color, ...style } as CSSProperties}
      data-testid={`stem-card-${stem.name}`}
      data-stem={stem.name}
      data-audible={audible}
      data-hovered={hovered}
      data-quiet={quiet}
      onPointerEnter={onEnter}
      onPointerLeave={onLeave}
      onPointerMove={onMove}
      aria-label={`${label} stem`}
    >
      <div className="stem-glow pointer-events-none absolute inset-0 -z-10" aria-hidden />
      {backdrop && <div className="pointer-events-none absolute inset-0 -z-10">{backdrop}</div>}

      <header className="flex items-start gap-3">
        <span className={`stem-dot shrink-0 rounded-full ${large ? "mt-[13px] h-2.5 w-2.5" : "mt-[9px] h-2 w-2"} ${hovered ? "is-live" : ""}`} style={{ background: stem.color }} aria-hidden />
        <div className="min-w-0 flex-1">
          <h3 className={`truncate font-display font-normal tracking-[-0.035em] ${large ? "text-[1.75rem] leading-9 sm:text-[2.1rem] sm:leading-10" : "text-xl leading-7"}`} data-testid="stem-label">
            {label}
          </h3>
          <p className="mt-0.5 truncate text-xs text-muted">{description}</p>
        </div>
        {headerAction}
        <div className="shrink-0 text-right" title="Share of the whole mix's energy that belongs to this stem">
          <p className={`num font-mono leading-none text-text ${large ? "mt-2 text-lg" : "mt-1.5 text-sm"}`} data-testid="energy-share">
            {formatPercent(analysis.energy_share)}
          </p>
          {large && <p className="mt-1.5 font-mono text-[10px] tracking-[0.14em] text-faint uppercase">of mix</p>}
        </div>
      </header>

      <Waveform peaks={analysis.peaks} color={stem.color} engine={engine} scale={peakScale} stem={stem.name} hidden={waveHidden} className={`mt-5 ${large ? "h-20" : "h-12"}`} />

      <div className="relative mt-2.5 h-8 overflow-hidden rounded-lg bg-black/50 ring-1 ring-white/[0.04]">
        <img src={analysis.spectrogram_url} alt={`Mel spectrogram of ${label}`} className="h-full w-full opacity-90" style={{ imageRendering: "auto" }} loading="lazy" draggable={false} />
        <div ref={specHead} className="pointer-events-none absolute inset-y-0 w-px bg-white/80 shadow-[0_0_8px_white]" style={{ left: 0 }} />
      </div>

      <BandMeter bands={analysis.bands} frameRate={frameRate} color={stem.color} engine={engine} className="mt-4" />

      <div className="mt-4 flex items-center gap-2">
        <ToggleButton active={control.solo} onClick={() => engine.toggleSolo(stem.name)} label={`Solo ${label}`} text={large ? "Solo" : undefined} testId="solo">
          <HeadphonesIcon />
        </ToggleButton>
        <ToggleButton active={control.muted} onClick={() => engine.toggleMute(stem.name)} label={`Mute ${label}`} text={large ? "Mute" : undefined} testId="mute" danger>
          <VolumeIcon muted />
        </ToggleButton>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={control.volume}
          onChange={(e) => engine.setVolume(stem.name, Number(e.target.value))}
          aria-label={`${label} volume`}
          className="mx-1 min-w-0 flex-1"
          style={{ "--fill": `${control.volume * 100}%` } as CSSProperties}
          data-testid="volume"
        />
        <a
          href={api.downloadUrl(stem.url)}
          download={stem.url.startsWith("blob:") ? `${label}.wav` : true}
          className="btn-icon h-9 w-9"
          aria-label={`Download ${label} as WAV`}
          title="Download WAV"
          data-testid="download"
        >
          <DownloadIcon />
        </a>
      </div>

      {footer}
      {overlay && <div className="pointer-events-none absolute inset-0 z-10">{overlay}</div>}
    </article>
  );
}

function ToggleButton({ active, onClick, label, text, children, testId, danger = false }: { active: boolean; onClick: () => void; label: string; text?: string; children: ReactNode; testId: string; danger?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      aria-label={label}
      title={label}
      data-testid={testId}
      data-danger={danger || undefined}
      className={`stem-toggle btn-icon h-9 gap-1.5 text-xs font-medium ${text ? "flex items-center px-3.5" : "w-9"}`}
    >
      {children}
      {text && <span>{text}</span>}
    </button>
  );
}
