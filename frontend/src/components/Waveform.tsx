import { useEffect, useRef } from "react";
import type { StemEngine } from "../audio/engine";
import { useCanvas } from "../lib/canvas";
import { drawBars } from "../lib/wave";

interface Props {
  peaks: number[];
  color: string;
  engine: StemEngine;
  /** peak value that maps to full height (shared across cards so quiet stems look quiet) */
  scale?: number;
  className?: string;
  /** stem name, so the split animation can find where this waveform sits */
  stem?: string;
  /** hidden while the split animation's strand is still flying in */
  hidden?: boolean;
}

/**
 * Static waveform drawn twice (dim + stem colour); playback progress only moves
 * a CSS clip on the coloured layer, so nothing is redrawn per frame.
 * Click or drag to seek.
 */
export function Waveform({ peaks, color, engine, scale = 1, className = "", stem, hidden = false }: Props) {
  const dimRef = useCanvas((ctx, w, h) => drawBars(ctx, w, h, peaks, scale, "rgba(255,255,255,0.13)"), [peaks, scale]);
  const litRef = useCanvas((ctx, w, h) => drawBars(ctx, w, h, peaks, scale, color), [peaks, scale, color]);
  const litWrap = useRef<HTMLDivElement | null>(null);
  const playhead = useRef<HTMLDivElement | null>(null);

  useEffect(
    () =>
      engine.onFrame((pos) => {
        const duration = engine.getSnapshot().duration;
        const frac = duration > 0 ? Math.min(1, pos / duration) : 0;
        if (litWrap.current) litWrap.current.style.clipPath = `inset(0 ${(1 - frac) * 100}% 0 0)`;
        if (playhead.current) playhead.current.style.left = `${frac * 100}%`;
      }),
    [engine],
  );

  const seekFromEvent = (e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    engine.seek(frac * engine.getSnapshot().duration);
  };

  return (
    <div
      className={`relative cursor-pointer touch-none select-none ${className}`}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        seekFromEvent(e);
      }}
      onPointerMove={(e) => {
        if (e.buttons & 1) seekFromEvent(e);
      }}
      data-testid="waveform"
      data-wave-for={stem}
      style={{ opacity: hidden ? 0 : 1, transition: "opacity 250ms ease-out" }}
    >
      <canvas ref={dimRef} className="absolute inset-0 h-full w-full" />
      <div ref={litWrap} className="absolute inset-0" style={{ clipPath: "inset(0 100% 0 0)" }}>
        <canvas ref={litRef} className="absolute inset-0 h-full w-full" />
      </div>
      <div ref={playhead} className="pointer-events-none absolute inset-y-0 w-px bg-white/80 shadow-[0_0_10px_rgb(255_255_255/0.8)]" style={{ left: 0 }} />
    </div>
  );
}
