/**
 * Pure mixing logic: given the stem tree and the user's solo/mute/volume/hover
 * state, what gain should each *leaf* stem play at?
 *
 * Only leaves are ever played. A parent (e.g. "music") is the sum of its
 * children (the API guarantees this), so playing both would double it. A
 * parent's controls therefore act on all of its descendants.
 */

import type { Stem } from "../api/types";

export interface StemControl {
  volume: number; // 0..1
  muted: boolean;
  solo: boolean;
}

export interface MixState {
  controls: Record<string, StemControl>;
  /** Stem under the cursor: transient solo that wins over everything else. */
  hover: string | null;
}

export const DEFAULT_CONTROL: StemControl = { volume: 1, muted: false, solo: false };

export interface StemTree {
  /** stems that carry audio (no children) */
  leaves: string[];
  /** name -> chain from the stem itself up to its top-level ancestor */
  lineage: Record<string, string[]>;
  children: Record<string, string[]>;
}

export function buildTree(stems: Pick<Stem, "name" | "parent">[]): StemTree {
  const names = new Set(stems.map((s) => s.name));
  const parentOf: Record<string, string | null> = {};
  const children: Record<string, string[]> = {};
  for (const s of stems) {
    children[s.name] ??= [];
    const parent = s.parent !== "root" && names.has(s.parent) ? s.parent : null;
    parentOf[s.name] = parent;
    if (parent) (children[parent] ??= []).push(s.name);
  }
  const lineage: Record<string, string[]> = {};
  for (const s of stems) {
    const chain: string[] = [];
    for (let cur: string | null = s.name; cur; cur = parentOf[cur] ?? null) chain.push(cur);
    lineage[s.name] = chain;
  }
  const leaves = stems.map((s) => s.name).filter((n) => (children[n] ?? []).length === 0);
  return { leaves, lineage, children };
}

/** Gain (0..1) for every leaf. */
export function computeLeafGains(tree: StemTree, state: MixState): Record<string, number> {
  const control = (name: string) => state.controls[name] ?? DEFAULT_CONTROL;
  const anySolo = Object.keys(tree.lineage).some((n) => control(n).solo);
  const gains: Record<string, number> = {};

  for (const leaf of tree.leaves) {
    const chain = tree.lineage[leaf] ?? [leaf];
    const volume = chain.reduce((v, n) => v * control(n).volume, 1);

    let audible: boolean;
    if (state.hover !== null) {
      // Hover = listen: exactly the hovered stem (or its children), even if muted.
      audible = chain.includes(state.hover);
    } else if (anySolo) {
      audible = chain.some((n) => control(n).solo) && !control(leaf).muted;
    } else {
      audible = !chain.some((n) => control(n).muted);
    }
    gains[leaf] = audible ? volume : 0;
  }
  return gains;
}

/** Is this stem (leaf or parent) currently contributing sound? Used to dim cards. */
export function isAudible(tree: StemTree, gains: Record<string, number>, name: string): boolean {
  const kids = tree.children[name] ?? [];
  if (kids.length === 0) return (gains[name] ?? 0) > 0;
  return kids.some((k) => isAudible(tree, gains, k));
}
