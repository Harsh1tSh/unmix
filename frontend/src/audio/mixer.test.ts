import { describe, expect, it } from "vitest";
import { buildTree, computeLeafGains, isAudible, type MixState, type StemControl } from "./mixer";

const stems = [
  { name: "voices", parent: "root" as const },
  { name: "music", parent: "root" as const },
  { name: "drums", parent: "music" as const },
  { name: "bass", parent: "music" as const },
  { name: "other", parent: "music" as const },
];
const tree = buildTree(stems);

const state = (controls: Record<string, Partial<StemControl>> = {}, hover: string | null = null): MixState => ({
  hover,
  controls: Object.fromEntries(Object.entries(controls).map(([k, v]) => [k, { volume: 1, muted: false, solo: false, ...v }])),
});

describe("buildTree", () => {
  it("plays only leaves, so a parent is never doubled with its children", () => {
    expect(tree.leaves).toEqual(["voices", "drums", "bass", "other"]);
    expect(tree.children.music).toEqual(["drums", "bass", "other"]);
    expect(tree.lineage.drums).toEqual(["drums", "music"]);
  });

  it("treats a stem whose parent is absent from the job as top-level", () => {
    const t = buildTree([{ name: "speaker_1", parent: "voices" }]);
    expect(t.leaves).toEqual(["speaker_1"]);
    expect(t.lineage.speaker_1).toEqual(["speaker_1"]);
  });

  it("makes voices a bus once phase 3 adds speaker children", () => {
    const t = buildTree([...stems, { name: "speaker_1", parent: "voices" }, { name: "speaker_2", parent: "voices" }]);
    expect(t.leaves).toEqual(["drums", "bass", "other", "speaker_1", "speaker_2"]);
  });
});

describe("computeLeafGains", () => {
  it("plays everything at unity by default", () => {
    expect(computeLeafGains(tree, state())).toEqual({ voices: 1, drums: 1, bass: 1, other: 1 });
  });

  it("mute on a parent silences its children", () => {
    expect(computeLeafGains(tree, state({ music: { muted: true } }))).toEqual({ voices: 1, drums: 0, bass: 0, other: 0 });
  });

  it("parent volume scales children", () => {
    const g = computeLeafGains(tree, state({ music: { volume: 0.5 }, drums: { volume: 0.5 } }));
    expect(g).toEqual({ voices: 1, drums: 0.25, bass: 0.5, other: 0.5 });
  });

  it("solo isolates a leaf; several solos add up", () => {
    expect(computeLeafGains(tree, state({ bass: { solo: true } }))).toEqual({ voices: 0, drums: 0, bass: 1, other: 0 });
    expect(computeLeafGains(tree, state({ bass: { solo: true }, voices: { solo: true } }))).toEqual({ voices: 1, drums: 0, bass: 1, other: 0 });
  });

  it("solo on a parent plays all of its children, minus muted ones", () => {
    const g = computeLeafGains(tree, state({ music: { solo: true }, drums: { muted: true } }));
    expect(g).toEqual({ voices: 0, drums: 0, bass: 1, other: 1 });
  });

  it("hover overrides solo and mute, and keeps the volume", () => {
    const s = state({ voices: { solo: true }, drums: { muted: true, volume: 0.8 } }, "drums");
    expect(computeLeafGains(tree, s)).toEqual({ voices: 0, drums: 0.8, bass: 0, other: 0 });
  });

  it("hovering a parent plays its whole subtree", () => {
    expect(computeLeafGains(tree, state({}, "music"))).toEqual({ voices: 0, drums: 1, bass: 1, other: 1 });
  });
});

describe("isAudible", () => {
  it("a parent is audible while any child is", () => {
    const g = computeLeafGains(tree, state({ bass: { solo: true } }));
    expect(isAudible(tree, g, "music")).toBe(true);
    expect(isAudible(tree, g, "bass")).toBe(true);
    expect(isAudible(tree, g, "drums")).toBe(false);
    expect(isAudible(tree, g, "voices")).toBe(false);
  });
});
