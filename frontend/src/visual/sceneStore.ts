/** Shared between the screens (which say where the sphere is) and the one scene that draws it. */

export type SphereMode = "idle" | "recording" | "separating" | "done";

export interface LiveLevels {
  /** seven band levels, 0..1 */
  bands: ArrayLike<number>;
  /** 0..1 */
  loudness: number;
}

export interface SphereTarget {
  /** the element the sphere is centred on */
  el: HTMLElement | null;
  mode: SphereMode;
  live?: () => LiveLevels | null;
  progress: number;
  hover: boolean;
  pressed: boolean;
  opacity: number;
  /** sphere radius as a fraction of el's width */
  radius: number;
}

export const scene: {
  target: SphereTarget | null;
  /** wakes the render loop (set by the scene while it is mounted) */
  kick: (() => void) | null;
  /** which renderer the scene ended up with, for tests and debugging */
  renderer: "webgl" | "2d" | null;
} = { target: null, kick: null, renderer: null };
