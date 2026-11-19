import { useEffect, useRef } from "react";
import type { Bands } from "../api/types";
import type { StemEngine } from "../audio/engine";
import { prepareCanvas } from "../lib/canvas";
import { formatHz } from "../lib/format";

interface Props {
  bands: Bands;
  frameRate: number;
  color: string;
  engine: StemEngine;
  className?: string;
}

const FALL_PER_S = 1.8; // meters drop at most this much (of full scale) per second
const PEAK_HOLD_S = 0.6; // a peak cap stays this long before it falls
const PEAK_FALL_PER_S = 0.5;

/**
 * 7-band energy meter driven by the precomputed analysis, sampled at the
 * engine's playback position (interpolated between 20 fps frames) and drawn on
 * a canvas every animation frame. Fast attack, slower release and a peak cap,
 * like a hardware meter.
 */
export function BandMeter({ bands, frameRate, color, engine, className = "" }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const n = bands.names.length;
    const shown = new Float32Array(n);
    const peaks = new Float32Array(n);
    const peakAge = new Float32Array(n);
    let lastTs = performance.now();
    let gradient: CanvasGradient | null = null;
    let gradientFor = 0;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    return engine.onFrame((pos, playing) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const prepared = prepareCanvas(canvas);
      if (!prepared) return;
      const { ctx, width, height } = prepared;

      const now = performance.now();
      const dt = Math.min(0.1, (now - lastTs) / 1000);
      lastTs = now;

      if (!gradient || gradientFor !== height) {
        gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, color);
        gradient.addColorStop(1, `${color}33`);
        gradientFor = height;
      }

      const f = pos * frameRate;
      const i0 = Math.min(bands.frames.length - 1, Math.max(0, Math.floor(f)));
      const i1 = Math.min(bands.frames.length - 1, i0 + 1);
      const t = f - Math.floor(f);
      const a = bands.frames[i0];
      const b = bands.frames[i1];

      const gap = Math.max(4, width * 0.025);
      const barW = (width - gap * (n - 1)) / n;
      const radius = Math.min(4, barW / 2);
      for (let k = 0; k < n; k++) {
        const target = a && b ? (a[k] ?? 0) * (1 - t) + (b[k] ?? 0) * t : 0;
        const prev = shown[k] ?? 0;
        // While paused (or with reduced motion) show the exact value at the playhead.
        const live = playing && !reduceMotion;
        const value = live ? Math.max(target, prev - FALL_PER_S * dt) : target;
        shown[k] = value;
        if (!live || value >= (peaks[k] ?? 0)) {
          peaks[k] = value;
          peakAge[k] = 0;
        } else {
          peakAge[k]! += dt;
          if (peakAge[k]! > PEAK_HOLD_S) peaks[k] = Math.max(value, peaks[k]! - PEAK_FALL_PER_S * dt);
        }

        const x = k * (barW + gap);
        ctx.fillStyle = "rgba(255,255,255,0.045)";
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x, 0, barW, height, radius);
        else ctx.rect(x, 0, barW, height);
        ctx.fill();

        const h = value * height;
        if (h > 0.5) {
          ctx.fillStyle = gradient;
          ctx.globalAlpha = 0.45 + 0.55 * value;
          ctx.beginPath();
          if (ctx.roundRect) ctx.roundRect(x, height - h, barW, h, Math.min(radius, h / 2));
          else ctx.rect(x, height - h, barW, h);
          ctx.fill();
          ctx.globalAlpha = 1;
        }
        const peakY = height - (peaks[k] ?? 0) * height;
        if ((peaks[k] ?? 0) > 0.02) {
          ctx.fillStyle = "rgba(255,255,255,0.85)";
          ctx.fillRect(x + 1, Math.max(0, peakY - 1.5), barW - 2, 1.5);
        }
      }
    });
  }, [bands, frameRate, color, engine]);

  const label = (i: number) => `${bands.names[i]}: ${formatHz(bands.hz_edges[i] ?? 0)}–${formatHz(bands.hz_edges[i + 1] ?? 0)} Hz`;

  return (
    <div className={className}>
      <canvas ref={canvasRef} className="block h-12 w-full" role="img" aria-label="Energy in 7 frequency bands" data-testid="band-meter" />
      <div className="mt-1.5 grid grid-cols-7 gap-[2.5%] text-center font-mono text-[9px] leading-none tracking-wide text-faint">
        {bands.names.map((name, i) => (
          <span key={name} title={label(i)} className="truncate">
            {formatHz(bands.hz_edges[i] ?? 0)}
          </span>
        ))}
      </div>
    </div>
  );
}
