import { useEffect, useRef, useState } from "react";
import { drawFallbackOrb } from "./fallbackOrb";
import { scene } from "./sceneStore";
import { paintNebula, StarRenderer } from "./space";
import { SphereRenderer } from "./sphere";

interface Props {
  /** "animated": twinkle, drift, parallax, shooting stars. "still": one dimmed sky (the stem board). */
  mode: "animated" | "still";
}

const BANDS = 7;
const REDUCED_TIME = 14; // a fixed moment of the noise fields when motion is reduced
/** Pixels the scene canvas may have: full-screen buffers above this cost a weak GPU too much to composite. */
const PIXEL_BUDGET = 5e6;

const ease = (value: number, target: number, per60: number, dt: number) => value + (target - value) * (1 - Math.pow(1 - per60, dt * 60));

function starCount(): number {
  const area = (window.screen?.width || window.innerWidth) * (window.screen?.height || window.innerHeight);
  return Math.round(Math.min(1700, Math.max(450, (area / 1e6) * 520)));
}

function sceneDpr(): number {
  const area = Math.max(1, window.innerWidth * window.innerHeight);
  return Math.max(0.75, Math.min(window.devicePixelRatio || 1, 1.5, Math.sqrt(PIXEL_BUDGET / area)));
}

/**
 * Everything behind the interface, mounted once in App so it never restarts:
 *   - a nebula, painted once at low resolution (no per-frame cost),
 *   - ONE WebGL canvas that draws the star field and the sound sphere. Screens
 *     only say where the sphere goes (<SoundSphere>); it glides there, and fades
 *     out where no screen wants it (the board). One context for the whole app,
 *     so nothing new is created when the screen changes.
 *   - the odd shooting star.
 * The loop sleeps when nothing moves (board, reduced motion) and in hidden tabs.
 */
export function SpaceBackdrop({ mode }: Props) {
  const nebulaRef = useRef<HTMLCanvasElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const [nebulaReady, setNebulaReady] = useState(false);
  const [reduce] = useState(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  // Nebula: painted once, after the page's first paint, then faded in.
  useEffect(() => {
    const canvas = nebulaRef.current;
    if (!canvas) return;
    const idle = window.requestIdleCallback ?? ((cb: () => void) => window.setTimeout(cb, 60));
    const cancel = window.cancelIdleCallback ?? window.clearTimeout;
    const handle = idle(() => {
      paintNebula(canvas, window.innerWidth / Math.max(1, window.innerHeight));
      setNebulaReady(true);
    });
    return () => cancel(handle as number);
  }, []);

  useEffect(() => scene.kick?.(), [mode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl2", { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false, powerPreference: "low-power" });
    let stars: StarRenderer | null = null;
    let sphere: SphereRenderer | null = null;
    const build = () => {
      if (!gl) return;
      stars = StarRenderer.create(gl, starCount());
      sphere = SphereRenderer.create(gl);
    };
    build();
    const ctx2d = gl ? null : canvas.getContext("2d");
    scene.renderer = gl && stars && sphere ? "webgl" : "2d";
    canvas.dataset.renderer = scene.renderer;
    let fallbackSky: HTMLCanvasElement | null = null;

    // sphere state, eased every frame
    const levels = new Float32Array(BANDS);
    const phases = new Float32Array(BANDS);
    let loud = 0;
    let color = 0.22;
    let split = 0;
    let scale = 1;
    let yaw = 0.6;
    let orbit = 0;
    let x = 0;
    let y = 0;
    let radius = 0;
    let alpha = 0;
    let placed = false;
    // sky state
    let dim = modeRef.current === "still" ? 0.55 : 1;
    let time = 0;
    let pointerX = 0;
    let pointerY = 0;
    let tiltX = 0;
    let tiltY = 0;
    // loop
    let raf = 0;
    let running = false;
    let last = performance.now();
    let width = 0;
    let height = 0;
    let dpr = 1;

    const resize = () => {
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      dpr = sceneDpr();
      const w = Math.round(width * dpr);
      const h = Math.round(height * dpr);
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        fallbackSky = null;
      }
    };

    const frame = (now: number) => {
      raf = 0;
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const t = scene.target;
      const animated = modeRef.current === "animated" && !reduce;
      const recording = t?.mode === "recording";
      let moving = false;
      const settle = (value: number, target: number, per60: number) => {
        const next = reduce && per60 < 0.2 ? target : ease(value, target, per60, dt);
        if (Math.abs(next - target) > 0.002) moving = true;
        return next;
      };

      if (!reduce) time += dt;
      tiltX = reduce ? 0 : ease(tiltX, pointerX, 0.05, dt);
      tiltY = reduce ? 0 : ease(tiltY, pointerY, 0.05, dt);
      dim = settle(dim, modeRef.current === "still" ? 0.55 : 1, 0.05);

      // where the sphere should be
      const rect = t?.el?.isConnected ? t.el.getBoundingClientRect() : null;
      if (t && rect && rect.width > 0) {
        const gx = rect.left + rect.width / 2;
        const gy = rect.top + rect.height / 2;
        const gr = rect.width * t.radius;
        if (!placed) {
          x = gx;
          y = gy;
          radius = gr * 0.85;
          placed = true;
        }
        x = settle(x, gx, 0.075);
        y = settle(y, gy, 0.075);
        radius = settle(radius, gr, 0.075);
        alpha = settle(alpha, t.opacity, 0.06);
      } else {
        radius = settle(radius, radius * 0.92, 0.05);
        alpha = settle(alpha, 0, 0.08);
        if (alpha < 0.003) {
          alpha = 0;
          placed = false;
        }
      }

      // what the sphere is doing
      const mic = recording ? (t?.live?.() ?? null) : null;
      for (let b = 0; b < BANDS; b++) {
        const target = mic ? (mic.bands[b] ?? 0) : reduce ? 0.07 : 0.05 + 0.045 * Math.sin(time * 0.8 + b * 0.9);
        levels[b] = ease(levels[b]!, target, target > levels[b]! ? 0.4 : 0.12, dt);
      }
      const m = t?.mode ?? "idle";
      loud = ease(loud, mic ? mic.loudness : 0, 0.3, dt);
      color = settle(color, recording ? 1 : m === "separating" ? 0.6 : m === "done" ? 0.8 : t?.hover ? 0.5 : 0.22, 0.06);
      split = settle(split, m === "separating" ? 0.12 + 0.88 * Math.min(1, Math.max(0, t?.progress ?? 0)) : m === "done" ? 1 : 0, 0.035);
      scale = settle(scale, t?.pressed ? 0.965 : t?.hover || recording ? 1.04 : 1, 0.12);
      if (!reduce) {
        yaw += dt * 0.042;
        for (let b = 0; b < BANDS; b++) phases[b]! += dt * split * (0.22 + 0.11 * b) * (b % 2 ? -1 : 1);
        orbit += dt * split * 0.07;
      }

      const sphereState = { time: reduce ? REDUCED_TIME : time, levels, loudness: loud, color, split, phases, orbit, yaw: yaw + tiltX * 0.35, pitch: 0.32 + tiltY * 0.2, scale };
      if (gl && stars && sphere) {
        if (gl.isContextLost()) {
          running = false;
          return;
        }
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE);
        (stars as StarRenderer).render(reduce ? REDUCED_TIME : time, -tiltX, -tiltY, dim, dpr);
        if (alpha > 0.003) (sphere as SphereRenderer).render(sphereState, { x, y, radius, alpha, dpr });
      } else if (ctx2d) {
        ctx2d.setTransform(1, 0, 0, 1, 0, 0);
        ctx2d.clearRect(0, 0, canvas.width, canvas.height);
        fallbackSky ??= paintStaticSky(canvas.width, canvas.height, dpr);
        ctx2d.globalAlpha = dim;
        ctx2d.drawImage(fallbackSky, 0, 0);
        if (alpha > 0.003) {
          const size = radius / 0.3; // drawFallbackOrb sizes the orb to 0.3 of its box
          ctx2d.setTransform(dpr, 0, 0, dpr, (x - size / 2) * dpr, (y - size / 2) * dpr);
          ctx2d.globalAlpha = alpha;
          drawFallbackOrb(ctx2d, size, size, sphereState);
        }
        ctx2d.globalAlpha = 1;
      }

      const sphereAlive = alpha > 0;
      const keepGoing = !document.hidden && (animated || recording || moving || (sphereAlive && !reduce));
      if (keepGoing) raf = requestAnimationFrame(frame);
      else running = false;
    };

    const start = () => {
      if (running || document.hidden) return;
      running = true;
      last = performance.now();
      raf = requestAnimationFrame(frame);
    };
    scene.kick = start;

    const onPointer = (e: PointerEvent) => {
      if (e.pointerType !== "mouse") return;
      pointerX = (e.clientX / window.innerWidth) * 2 - 1;
      pointerY = (e.clientY / window.innerHeight) * 2 - 1;
    };
    const onVisibility = () => !document.hidden && start();
    const onLost = (e: Event) => e.preventDefault(); // allows the context to be restored
    const onRestored = () => {
      build();
      start();
    };
    const ro = new ResizeObserver(() => {
      resize();
      start();
    });
    resize();
    ro.observe(canvas);
    window.addEventListener("pointermove", onPointer, { passive: true });
    // the sphere follows its anchor when the page scrolls (phones)
    window.addEventListener("scroll", start, { passive: true });
    document.addEventListener("visibilitychange", onVisibility);
    canvas.addEventListener("webglcontextlost", onLost);
    canvas.addEventListener("webglcontextrestored", onRestored);
    start();
    return () => {
      cancelAnimationFrame(raf);
      scene.kick = null;
      ro.disconnect();
      window.removeEventListener("pointermove", onPointer);
      window.removeEventListener("scroll", start);
      document.removeEventListener("visibilitychange", onVisibility);
      canvas.removeEventListener("webglcontextlost", onLost);
      canvas.removeEventListener("webglcontextrestored", onRestored);
      (stars as StarRenderer | null)?.dispose();
      (sphere as SphereRenderer | null)?.dispose();
    };
  }, [reduce]);

  return (
    <>
      <div className="pointer-events-none fixed inset-0 -z-[9] overflow-hidden" aria-hidden>
        <canvas
          ref={nebulaRef}
          className="absolute inset-0 h-full w-full object-cover blur-[6px] transition-opacity duration-[2000ms]"
          style={{ opacity: nebulaReady ? (mode === "still" ? 0.32 : 0.55) : 0 }}
        />
        {mode === "animated" && !reduce && <ShootingStars />}
      </div>
      {/* above the planet's horizon (-z-[8]) so the sphere is never hidden behind it */}
      <canvas ref={canvasRef} className="pointer-events-none fixed inset-0 -z-[7] h-full w-full" aria-hidden data-testid="scene" />
    </>
  );
}

/** Without WebGL2: the sky, drawn once into an offscreen canvas. */
function paintStaticSky(w: number, h: number, dpr: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  if (!ctx) return c;
  let seed = 3;
  const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
  for (let i = 0; i < 700; i++) {
    const d = Math.pow(rand(), 2.4);
    ctx.globalAlpha = 0.2 + d * 0.6;
    ctx.fillStyle = "#dfe6ff";
    ctx.beginPath();
    ctx.arc(rand() * w, rand() * h * 0.9, (0.4 + d * 1.2) * dpr, 0, Math.PI * 2);
    ctx.fill();
  }
  return c;
}

/** Now and then a streak of light crosses the upper part of the sky. */
function ShootingStars() {
  const [star, setStar] = useState<{ id: number; x: number; y: number; angle: number } | null>(null);
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let id = 0;
    const next = (delay: number) => {
      timer = setTimeout(() => {
        if (!document.hidden) setStar({ id: ++id, x: 25 + Math.random() * 70, y: Math.random() * 35, angle: 145 + Math.random() * 25 });
        next(9000 + Math.random() * 12000);
      }, delay);
    };
    next(3500 + Math.random() * 4000);
    return () => clearTimeout(timer);
  }, []);
  if (!star) return null;
  return (
    <span key={star.id} className="absolute block" style={{ left: `${star.x}%`, top: `${star.y}%`, transform: `rotate(${star.angle}deg)` }}>
      <span className="shooting-star block" />
    </span>
  );
}
