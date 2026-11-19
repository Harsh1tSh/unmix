import { describe, expect, it } from "vitest";
import type { Lyrics } from "../api/types";
import { toLRC, toSRT } from "./lyricsExport";

const w = (start: number, end: number, text: string) => ({ start, end, text, probability: 1 });
const single: Lyrics = {
  language: "en",
  language_probability: 1,
  model: "m",
  duration_s: 75.5,
  segments: [
    { start: 1.234, end: 2.5, text: "hello there", words: [w(1.234, 1.6, " hello"), w(1.8, 2.5, " there")] },
    { start: 61.05, end: 62, text: "a minute in", words: [w(61.05, 62, " a minute in")] },
  ],
};

describe("toLRC", () => {
  it("writes timed lines with a header", () => {
    expect(toLRC(single, {}, "My [song]")).toBe("[ti:My  song ]\n[length:01:15]\n[re:Unmix]\n[00:01.23]hello there\n[01:01.05]a minute in\n");
  });
  it("names speakers only when there are several", () => {
    const two: Lyrics = { ...single, segments: [{ ...single.segments[0]!, speaker: "speaker_1" }, { ...single.segments[1]!, speaker: "speaker_2" }] };
    const out = toLRC(two, { speaker_1: "You", speaker_2: "Other voice" });
    expect(out).toContain("[00:01.23]You: hello there");
    expect(out).toContain("[01:01.05]Other voice: a minute in");
  });
});

describe("toSRT", () => {
  it("numbers cues with start and end times", () => {
    expect(toSRT(single)).toBe("1\n00:00:01,234 --> 00:00:02,500\nhello there\n\n2\n00:01:01,050 --> 00:01:02,000\na minute in\n");
  });
});
