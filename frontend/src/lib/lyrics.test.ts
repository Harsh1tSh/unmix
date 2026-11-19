import { describe, expect, it } from "vitest";
import type { LyricSegment } from "../api/types";
import { locate, wordProgress } from "./lyrics";

const w = (start: number, end: number, text: string) => ({ start, end, text, probability: 1 });
const segments: LyricSegment[] = [
  { start: 1, end: 2.5, text: "hello there", words: [w(1, 1.5, " hello"), w(1.8, 2.5, " there")] },
  { start: 4, end: 5, text: "again", words: [w(4, 5, " again")] },
];

describe("locate", () => {
  it("points at the first line before anything is sung", () => {
    expect(locate(segments, 0)).toEqual({ segment: 0, word: -1, inside: false });
  });
  it("tracks the word being sung", () => {
    expect(locate(segments, 1.2)).toEqual({ segment: 0, word: 0, inside: true });
    expect(locate(segments, 1.6)).toEqual({ segment: 0, word: 0, inside: true });
    expect(locate(segments, 2.0)).toEqual({ segment: 0, word: 1, inside: true });
  });
  it("keeps the line through the pause before the next one", () => {
    expect(locate(segments, 3)).toEqual({ segment: 0, word: 1, inside: false });
    expect(locate(segments, 4.1)).toEqual({ segment: 1, word: 0, inside: true });
  });
  it("handles the end and empty lyrics", () => {
    expect(locate(segments, 99).segment).toBe(1);
    expect(locate([], 1)).toEqual({ segment: -1, word: -1, inside: false });
  });
});

describe("wordProgress", () => {
  it("fills linearly across the word", () => {
    expect(wordProgress(1, 2, 0.5)).toBe(0);
    expect(wordProgress(1, 2, 1.5)).toBe(0.5);
    expect(wordProgress(1, 2, 3)).toBe(1);
    expect(wordProgress(1, 1, 1)).toBe(1);
  });
});
