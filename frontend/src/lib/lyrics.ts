import type { LyricSegment } from "../api/types";

export interface LyricPosition {
  /** index of the line to show as current (the active one, or the next upcoming one) */
  segment: number;
  /** index of the word being sung inside that line; -1 when between words */
  word: number;
  /** true while the playhead is inside the line's [start, end] */
  inside: boolean;
}

/**
 * Where is the playhead in the lyrics? Binary search, since this runs every
 * animation frame. A line stays current until the next one starts, so it does
 * not flicker away during short pauses.
 */
export function locate(segments: LyricSegment[], t: number): LyricPosition {
  if (segments.length === 0) return { segment: -1, word: -1, inside: false };
  // last segment whose start <= t
  let lo = 0;
  let hi = segments.length - 1;
  let idx = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (segments[mid]!.start <= t) {
      idx = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  if (idx === -1) return { segment: 0, word: -1, inside: false }; // before the first line
  const seg = segments[idx]!;
  const inside = t <= seg.end;
  let word = -1;
  for (let i = seg.words.length - 1; i >= 0; i--) {
    if (seg.words[i]!.start <= t) {
      word = i;
      break;
    }
  }
  return { segment: idx, word, inside };
}

/** 0..1 progress through a word, for the karaoke fill. */
export function wordProgress(start: number, end: number, t: number): number {
  if (end <= start) return t >= start ? 1 : 0; // zero-length word: on once reached
  if (t <= start) return 0;
  if (t >= end) return 1;
  return (t - start) / (end - start);
}
