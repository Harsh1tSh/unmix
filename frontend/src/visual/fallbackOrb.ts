import { BAND_COLORS } from "./sphere";

/**
 * 2D-canvas stand-in for the WebGL sphere (no WebGL2, or the context was
 * lost): a glowing core with one ring per band, pushed out by that band's level.
 */
export function drawFallbackOrb(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  state: { time: number; levels: ArrayLike<number>; loudness: number; color: number; split: number; scale: number },
): void {
  const cx = width / 2;
  const cy = height / 2;
  const base = (Math.min(width, height) * 0.3 * state.scale) / (1 + state.split * 0.45);
  const radius = base * (0.8 + state.loudness * 0.35);

  BAND_COLORS.forEach((color, i) => {
    const lvl = state.levels[i] ?? 0;
    const r = radius * (1.08 + i * 0.07) + lvl * base * 0.3 + state.split * (i - 3) * base * 0.12;
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(1, r), 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.1 + lvl * 0.55 + state.color * 0.12;
    ctx.lineWidth = 1.25;
    ctx.stroke();
  });
  ctx.globalAlpha = 1;

  const glow = ctx.createRadialGradient(cx, cy, radius * 0.2, cx, cy, radius * 1.7);
  glow.addColorStop(0, `rgba(196,181,253,${0.18 + state.color * 0.2})`);
  glow.addColorStop(1, "rgba(5,6,10,0)");
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(cx, cy, radius * 1.7, 0, Math.PI * 2);
  ctx.fill();

  const body = ctx.createRadialGradient(cx - radius * 0.35, cy - radius * 0.4, radius * 0.1, cx, cy, radius);
  body.addColorStop(0, "#dfe3ee");
  body.addColorStop(0.35, "#6f7790");
  body.addColorStop(1, "#11141c");
  ctx.fillStyle = body;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fill();
}
