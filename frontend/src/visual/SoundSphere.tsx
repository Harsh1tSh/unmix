import { useLayoutEffect, useRef } from "react";
import { scene, type LiveLevels, type SphereMode } from "./sceneStore";

export type { LiveLevels, SphereMode } from "./sceneStore";

interface Props {
  mode: SphereMode;
  /** polled every frame while recording (the microphone's levels) */
  live?: () => LiveLevels | null;
  /** 0..1, how far the separation has got (mode "separating") */
  progress?: number;
  /** the pointer is over the sphere: it swells and warms up */
  hover?: boolean;
  /** pressed down: it gives a little */
  pressed?: boolean;
  /** 0..1, fades the sphere (e.g. behind the split animation) */
  opacity?: number;
  /** sphere radius as a fraction of this element's width */
  radius?: number;
  className?: string;
}

/**
 * Marks where the sound sphere should be. The sphere itself is drawn by the one
 * app-wide scene (SpaceBackdrop), so moving from one screen to the next never
 * creates a new WebGL canvas (a new canvas flashed as a white box on some GPUs)
 * and the sphere glides from its old place to the new one.
 */
export function SoundSphere({ mode, live, progress = 0, hover = false, pressed = false, opacity = 1, radius = 0.445, className = "" }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const target = useRef({ el: null as HTMLElement | null, mode, live, progress, hover, pressed, opacity, radius });
  Object.assign(target.current, { mode, live, progress, hover, pressed, opacity, radius });

  // Register on mount. Layout effects: the old screen's anchor is released first,
  // in the same commit, so the sphere flies straight from one place to the other.
  useLayoutEffect(() => {
    const t = target.current;
    t.el = ref.current;
    scene.target = t;
    scene.kick?.();
    return () => {
      if (scene.target === t) scene.target = null;
      scene.kick?.();
    };
  }, []);

  // Any prop change may need frames (the loop idles under reduced motion).
  useLayoutEffect(() => scene.kick?.());

  return <div ref={ref} className={`pointer-events-none ${className}`} aria-hidden data-testid="sound-sphere" />;
}
