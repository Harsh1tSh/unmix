import { animate } from "motion";
import { useEffect, useLayoutEffect, useRef } from "react";
import type { Analysis, Stem } from "../api/types";
import { useCanvas } from "../lib/canvas";
import { drawBars } from "../lib/wave";

/** Viewport rectangles of each stem's strand at the end of the split. */
export type StrandRects = Record<string, { x: number; y: number; width: number; height: number }>;

export const prefersReducedMotion = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function Strand({ peaks, color, scale }: { peaks: number[]; color: string; scale: number }) {
  const ref = useCanvas((ctx, w, h) => drawBars(ctx, w, h, peaks, scale, color), [peaks, color, scale]);
  return <canvas ref={ref} className="absolute inset-0 h-full w-full" />;
}

/**
 * Part 1, on the progress screen: the mix waveform separates into one strand
 * per leaf stem (drawn from that stem's real peaks), fanning out vertically.
 * Calls onDone with where each strand ended up.
 */
export function SplitStrands({ stems, analysis, onDone }: { stems: Stem[]; analysis: Analysis; onDone: (rects: StrandRects) => void }) {
  const box = useRef<HTMLDivElement | null>(null);
  const els = useRef<Record<string, HTMLDivElement | null>>({});
  const scale = Math.max(0.05, ...stems.map((s) => Math.max(...(analysis.stems[s.name]?.peaks ?? [0]))));
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  useLayoutEffect(() => {
    const container = box.current;
    if (!container) return;
    const finish = () => {
      const rects: StrandRects = {};
      for (const s of stems) {
        const r = els.current[s.name]?.getBoundingClientRect();
        if (r) rects[s.name] = { x: r.x, y: r.y, width: r.width, height: r.height };
      }
      doneRef.current(rects);
    };
    if (prefersReducedMotion()) {
      finish();
      return;
    }
    const n = stems.length;
    const gap = Math.min(28, container.clientHeight / Math.max(n, 1));
    const controls = stems.map((s, i) => {
      const el = els.current[s.name]!;
      const offset = (i - (n - 1) / 2) * gap;
      return animate(el, { opacity: [0, 1], y: [0, offset], scaleY: [1, 0.55] }, { duration: 0.75, delay: 0.04 * i, ease: [0.22, 1, 0.36, 1] });
    });
    let cancelled = false;
    void Promise.all(controls.map((c) => c.finished)).then(() => !cancelled && setTimeout(() => !cancelled && finish(), 150));
    return () => {
      cancelled = true;
      controls.forEach((c) => c.stop());
    };
  }, [stems]);

  return (
    <div ref={box} className="pointer-events-none absolute inset-0" data-testid="split-strands">
      {stems.map((s) => (
        <div key={s.name} ref={(el) => void (els.current[s.name] = el)} className="absolute inset-0 opacity-0" style={{ mixBlendMode: "screen" }}>
          <Strand peaks={analysis.stems[s.name]?.peaks ?? []} color={s.color} scale={scale} />
        </div>
      ))}
    </div>
  );
}

/**
 * Part 2, on the stem board: each strand flies from where part 1 left it into
 * its card's waveform ([data-wave-for]). Cards keep their waveform hidden until
 * onLanded(name) fires for them.
 */
export function FlyIn({ from, stems, analysis, onLanded, onDone }: { from: StrandRects; stems: Stem[]; analysis: Analysis; onLanded: (name: string) => void; onDone: () => void }) {
  const els = useRef<Record<string, HTMLDivElement | null>>({});
  const cb = useRef({ onLanded, onDone });
  cb.current = { onLanded, onDone };
  const scale = Math.max(0.05, ...Object.values(analysis.stems).map((s) => Math.max(...s.peaks)));

  useEffect(() => {
    let cancelled = false;
    const controls: { stop: () => void }[] = [];
    // Two frames: let the board lay out before measuring the targets.
    const raf = requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        if (cancelled) return;
        const flights = stems.map((s, i) => {
          const el = els.current[s.name];
          const target = document.querySelector<HTMLElement>(`[data-wave-for="${s.name}"]`)?.getBoundingClientRect();
          const start = from[s.name];
          if (!el || !target || !start) {
            cb.current.onLanded(s.name);
            return Promise.resolve();
          }
          const dx = target.x - start.x;
          const dy = target.y - start.y;
          const c = animate(
            el,
            { x: [0, dx], y: [0, dy], scaleX: [1, target.width / start.width], scaleY: [1, target.height / start.height] },
            { duration: 0.85, delay: 0.06 * i, ease: [0.65, 0, 0.35, 1] },
          );
          controls.push(c);
          return c.finished.then(() => {
            if (cancelled) return;
            cb.current.onLanded(s.name);
            controls.push(animate(el, { opacity: 0 }, { duration: 0.25 }));
          });
        });
        void Promise.all(flights).then(() => !cancelled && setTimeout(() => !cancelled && cb.current.onDone(), 300));
      }),
    );
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      controls.forEach((c) => c.stop());
    };
  }, [from, stems]);

  return (
    <div className="pointer-events-none fixed inset-0 z-40" aria-hidden data-testid="fly-in">
      {stems.map((s) => {
        const r = from[s.name];
        if (!r) return null;
        return (
          <div
            key={s.name}
            ref={(el) => void (els.current[s.name] = el)}
            className="absolute"
            style={{ left: r.x, top: r.y, width: r.width, height: r.height, transformOrigin: "0 0", mixBlendMode: "screen" }}
          >
            <Strand peaks={analysis.stems[s.name]?.peaks ?? []} color={s.color} scale={scale} />
          </div>
        );
      })}
    </div>
  );
}
