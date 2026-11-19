import { useCallback, useEffect, useMemo, useState, useSyncExternalStore, type CSSProperties } from "react";
import type { Analysis, Job, Lyrics, Stem } from "../api/types";
import { StemEngine } from "../audio/engine";
import { formatDuration, formatPercent } from "../lib/format";
import { readSetting, writeSetting } from "../lib/storage";
import { ChevronIcon, PlusIcon, TrashIcon, UserIcon } from "./icons";
import { CleanVoice } from "./CleanVoice";
import { LyricsPanel } from "./LyricsPanel";
import { FlyIn, type StrandRects } from "./Split";
import { StemCard } from "./StemCard";
import { TransportBar } from "./TransportBar";
import { Wordmark } from "./UploadScreen";

interface Props {
  job: Job;
  stems: Stem[];
  analysis: Analysis;
  lyrics: Lyrics | null;
  /** where the split animation's strands are; null = no intro animation */
  intro: StrandRects | null;
  onReset: () => void;
  /** delete the job's files from this computer, then go back to the start */
  onDelete: () => void;
}

/** Creates the engine for a job and tears it down when the job changes. */
function useStemEngine(stems: Stem[]): StemEngine | null {
  const [engine, setEngine] = useState<StemEngine | null>(null);
  useEffect(() => {
    const instance = new StemEngine(stems);
    setEngine(instance);
    void instance.load();
    // Handle for e2e tests and console debugging.
    (window as unknown as { __unmix?: StemEngine }).__unmix = instance;
    return () => {
      instance.dispose();
      setEngine(null);
    };
  }, [stems]);
  return engine;
}

export function StemBoard(props: Props) {
  const engine = useStemEngine(props.stems);
  // Waveforms stay hidden until their strand from the split animation lands.
  const [landed, setLanded] = useState<Set<string> | "all">(() => (props.intro ? new Set() : "all"));
  const [flying, setFlying] = useState(props.intro !== null);
  const onLanded = useCallback((name: string) => setLanded((l) => (l === "all" ? l : new Set(l).add(name))), []);
  const onDone = useCallback(() => {
    setLanded("all");
    setFlying(false);
  }, []);
  const animated = useMemo(() => props.stems.filter((s) => props.analysis.stems[s.name]), [props.stems, props.analysis]);

  return (
    <>
      {engine && <Board {...props} engine={engine} landed={landed} />}
      {flying && props.intro && <FlyIn from={props.intro} stems={animated} analysis={props.analysis} onLanded={onLanded} onDone={onDone} />}
    </>
  );
}

const SPEAKERS = ["speaker_1", "speaker_2"] as const;

/** Which separated voice is the user's (remembered per job in this browser). */
function useMe(jobId: string): [string | null, (name: string | null) => void] {
  const key = `me:${jobId}`;
  const [me, setMeState] = useState<string | null>(() => readSetting(key) || null);
  const setMe = useCallback(
    (name: string | null) => {
      setMeState(name);
      writeSetting(key, name ?? "");
    },
    [key],
  );
  return [me, setMe];
}

const order = (i: number) => ({ "--i": i }) as CSSProperties;

function Board({ engine, job, stems, analysis, lyrics, intro, onReset, onDelete, landed }: Props & { engine: StemEngine; landed: Set<string> | "all" }) {
  const snapshot = useSyncExternalStore(engine.subscribe, engine.getSnapshot);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const roots = useMemo(() => stems.filter((s) => (engine.tree.lineage[s.name] ?? []).length === 1), [stems, engine]);
  const byName = useMemo(() => new Map(stems.map((s) => [s.name, s])), [stems]);
  // One vertical scale for every waveform: the loudest peak in the job.
  const peakScale = useMemo(() => Math.max(0.05, ...Object.values(analysis.stems).map((s) => s.peaks.reduce((m, v) => (v > m ? v : m), 0))), [analysis]);

  const isHidden = (name: string) => landed !== "all" && !landed.has(name);
  const colors = useMemo(() => Object.fromEntries(stems.map((st) => [st.name, st.color])), [stems]);
  const [me, setMe] = useMe(job.id);
  const local = job.id.startsWith("local-"); // separated in this browser: nothing on the server
  const [confirmDelete, setConfirmDelete] = useState(false);
  // "That's me": the chosen voice becomes "You", the other "Other voice" (cards and lyrics).
  const labels = useMemo(() => {
    const out: Record<string, string> = Object.fromEntries(stems.map((st) => [st.name, st.label]));
    if (me && SPEAKERS.includes(me as (typeof SPEAKERS)[number])) {
      for (const sp of SPEAKERS) if (out[sp]) out[sp] = sp === me ? "You" : "Other voice";
    }
    return out;
  }, [stems, me]);
  const meToggle = (name: string) =>
    SPEAKERS.includes(name as (typeof SPEAKERS)[number]) ? (
      <button
        type="button"
        onClick={() => setMe(me === name ? null : name)}
        aria-pressed={me === name}
        title={me === name ? "Undo: this is not me" : "Mark this voice as yours"}
        className={`btn mt-1 h-7 shrink-0 gap-1 px-2.5 text-[11px] ${me === name ? "btn-primary" : "btn-ghost text-muted"}`}
        data-testid={`me-${name}`}
      >
        <UserIcon size={11} />
        {me === name ? "You" : "That's me"}
      </button>
    ) : undefined;

  const childrenOf = (name: string): Stem[] => (engine.tree.children[name] ?? []).map((n) => byName.get(n)).filter((s): s is Stem => !!s && !!analysis.stems[s.name]);
  // The played sources (leaves), for the composition bar.
  const leaves = useMemo(() => stems.filter((s) => !(engine.tree.children[s.name] ?? []).length && analysis.stems[s.name]), [stems, engine, analysis]);

  // Cards fade up into place, unless the split animation's strands are their entrance
  // (the strands aim at where the cards are, so the cards must not move meanwhile).
  const [entrance] = useState(() => (intro ? "" : "rise"));
  let cardIndex = 0;

  return (
    <div className="pb-36" data-testid="stem-board">
      <nav className="sticky top-0 z-30 px-3 pt-3 sm:pt-4">
        <div className="glass mx-auto flex h-14 max-w-6xl items-center 2xl:max-w-[1440px] gap-3 rounded-full pr-2 pl-5">
          <Wordmark onClick={onReset} />
          <span className="h-5 w-px bg-line-strong" aria-hidden />
          <p className="min-w-0 flex-1 truncate text-sm text-muted" title={job.filename} data-testid="board-filename">
            {job.filename}
          </p>
          {confirmDelete ? (
            <span className="flex items-center gap-2 text-xs">
              <span className="hidden text-muted md:inline">{local ? "Discard these stems?" : "Delete the recording, stems and lyrics?"}</span>
              <button type="button" onClick={onDelete} className="btn btn-danger h-9 px-4 text-xs" data-testid="confirm-delete">
                Delete
              </button>
              <button type="button" onClick={() => setConfirmDelete(false)} className="btn btn-ghost h-9 px-4 text-xs">
                Keep
              </button>
            </span>
          ) : (
            <button
              type="button"
              onClick={() => setConfirmDelete(true)}
              className="btn btn-ghost h-10 px-3 text-xs text-muted hover:border-danger/60 hover:text-danger sm:px-4"
              title="Delete this job's files from this computer now (they are deleted automatically after 24 hours)"
              aria-label="Delete"
              data-testid="delete-job"
            >
              <TrashIcon size={13} />
              <span className="hidden sm:inline">Delete</span>
            </button>
          )}
          <button type="button" onClick={onReset} className="btn btn-primary h-10 px-4 text-xs" data-testid="new-file">
            <PlusIcon size={13} />
            New
          </button>
        </div>
      </nav>

      <div className="mx-auto max-w-6xl px-4 pt-10 sm:px-6 sm:pt-16 2xl:max-w-[1440px]">
        <header className="flex flex-wrap items-end justify-between gap-x-10 gap-y-5">
          <div>
            <p className="eyebrow rise" style={order(0)}>
              Unmixed
            </p>
            <h1 className="display mt-3 text-[clamp(3.25rem,8vw,6.5rem)]">
              <span className="reveal-line">
                <span style={order(0)}>
                  <span className="num">{stems.length}</span> <span className="text-muted">sources</span>
                </span>
              </span>
            </h1>
          </div>
          <dl className="rise flex flex-wrap gap-2 pb-2 text-xs" style={order(2)}>
            {[
              ["Length", formatDuration(analysis.duration_s)],
              ["Mode", local ? "In this browser" : job.mode === "quality" ? "Quality" : "Fast"],
              ...(job.timings?.total ? [["Separated in", formatDuration(job.timings.total)]] : []),
              ...(lyrics?.language ? [["Language", lyrics.language.toUpperCase()]] : []),
            ].map(([k, v]) => (
              <div key={k} className="panel rounded-full px-3.5 py-2">
                <dt className="inline font-mono text-[10px] tracking-[0.14em] text-faint uppercase">{k} </dt>
                <dd className="num ml-1 inline text-text">{v}</dd>
              </div>
            ))}
          </dl>
        </header>

        <Composition leaves={leaves} analysis={analysis} labels={labels} engine={engine} hover={snapshot.hover} />

        {snapshot.status === "loading" && (
          <div className="mt-8" role="status" data-testid="stems-loading">
            <p className="text-sm text-muted">
              Loading stems into the player… <span className="num">{Math.round(snapshot.loadProgress * 100)}%</span>
            </p>
            <div className="mt-2 h-[3px] overflow-hidden rounded-full bg-white/[0.08]">
              <div className="h-full rounded-full bg-white transition-[width] duration-300" style={{ width: `${snapshot.loadProgress * 100}%` }} />
            </div>
          </div>
        )}
        {snapshot.status === "error" && (
          <p className="mt-8 rounded-2xl border border-danger/35 bg-danger/10 px-4 py-3 text-sm text-danger" role="alert">
            {snapshot.error}
          </p>
        )}

        {/* Tree layout: top-level stems side by side, each expanded parent's children as a branch below. */}
        <div className="mt-8 grid items-start gap-4 sm:gap-5 lg:grid-cols-2" data-ready={snapshot.status === "ready"} data-testid="stem-grid">
          {roots.map((root) => {
            const rootAnalysis = analysis.stems[root.name];
            if (!rootAnalysis) return null;
            const kids = childrenOf(root.name);
            const open = expanded[root.name] ?? true;
            return (
              <StemCard
                key={root.name}
                className={entrance}
                style={order(3 + cardIndex++)}
                stem={root}
                analysis={rootAnalysis}
                frameRate={analysis.frame_rate}
                engine={engine}
                snapshot={snapshot}
                peakScale={peakScale}
                waveHidden={isHidden(root.name)}
                footer={
                  <>
                    {root.name === "voices" && !local && !childrenOf("voices").length && <CleanVoice jobId={job.id} stem="voices" label={labels.voices ?? "Voices"} engine={engine} />}
                    {root.name === "voices" && lyrics && <LyricsPanel lyrics={lyrics} engine={engine} colors={colors} labels={labels} fallbackColor={root.color} filename={job.filename} />}
                    {kids.length > 0 && (
                      <button
                        type="button"
                        onClick={() => setExpanded((e) => ({ ...e, [root.name]: !open }))}
                        aria-expanded={open}
                        className="mt-5 flex w-full items-center justify-between gap-3 rounded-2xl border border-line bg-white/[0.02] px-4 py-3 text-left text-sm text-muted transition-colors hover:border-line-strong hover:bg-white/[0.04] hover:text-text"
                        data-testid={`expand-${root.name}`}
                      >
                        <span className="flex min-w-0 items-center gap-3">
                          <span className="flex shrink-0 -space-x-1" aria-hidden>
                            {kids.map((k) => (
                              <span key={k.name} className="h-2.5 w-2.5 rounded-full ring-2 ring-surface" style={{ background: k.color }} />
                            ))}
                          </span>
                          <span className="truncate">
                            {open ? "Hide" : "Show"} {kids.length} parts: {kids.map((k) => (labels[k.name] ?? k.label).toLowerCase()).join(", ")}
                          </span>
                        </span>
                        <ChevronIcon open={open} className="shrink-0" />
                      </button>
                    )}
                  </>
                }
              />
            );
          })}
        </div>

        {roots.map((root) => {
          const kids = childrenOf(root.name);
          if (kids.length === 0 || !(expanded[root.name] ?? true)) return null;
          return (
            <section key={root.name} className="relative mt-8 pl-5 sm:pl-8" aria-label={`Parts of ${root.label}`} data-testid={`children-${root.name}`}>
              {/* the branch: a line in the parent's colour that fades out */}
              <span className="absolute top-1 bottom-4 left-0 w-px" style={{ background: `linear-gradient(${root.color}, ${root.color}00)` }} aria-hidden />
              <p className="eyebrow mb-4 flex items-center gap-2" style={{ color: root.color }}>
                <span className="h-px w-4" style={{ background: root.color }} aria-hidden />
                Inside {labels[root.name] ?? root.label}
              </p>
              <div className="grid gap-4 sm:grid-cols-2 sm:gap-5 xl:grid-cols-3">
                {kids.map((kid) => (
                  <StemCard
                    key={kid.name}
                    className={entrance}
                    style={order(3 + cardIndex++)}
                    stem={kid}
                    analysis={analysis.stems[kid.name]!}
                    frameRate={analysis.frame_rate}
                    engine={engine}
                    snapshot={snapshot}
                    peakScale={peakScale}
                    size="sm"
                    waveHidden={isHidden(kid.name)}
                    displayLabel={labels[kid.name]}
                    headerAction={meToggle(kid.name)}
                    footer={kid.parent === "voices" ? <CleanVoice jobId={job.id} stem={kid.name} label={labels[kid.name] ?? kid.label} engine={engine} /> : undefined}
                  />
                ))}
              </div>
            </section>
          );
        })}
      </div>
      {snapshot.status === "ready" && <TransportBar engine={engine} snapshot={snapshot} labels={labels} />}
    </div>
  );
}

/**
 * What the mix is made of: one segment per played source, sized by its share
 * of the mix's energy. Hovering a segment listens to that source, like its card.
 */
function Composition({ leaves, analysis, labels, engine, hover }: { leaves: Stem[]; analysis: Analysis; labels: Record<string, string>; engine: StemEngine; hover: string | null }) {
  const shares = leaves.map((s) => analysis.stems[s.name]?.energy_share ?? 0);
  const total = shares.reduce((a, b) => a + b, 0) || 1;
  if (leaves.length < 2) return null;
  return (
    <div className="rise mt-8" style={order(2)} data-testid="composition">
      <div className="flex h-2.5 gap-[3px] overflow-hidden rounded-full" role="img" aria-label={`Mix composition: ${leaves.map((s, i) => `${labels[s.name] ?? s.label} ${formatPercent(shares[i]!)}`).join(", ")}`}>
        {leaves.map((s, i) => (
          <span
            key={s.name}
            className="h-full min-w-[3px] rounded-full transition-[opacity,filter] duration-300"
            style={{ flexGrow: Math.max(shares[i]! / total, 0.004), background: s.color, opacity: hover && hover !== s.name ? 0.25 : 1, boxShadow: hover === s.name ? `0 0 14px ${s.color}` : undefined }}
            title={`${labels[s.name] ?? s.label}: ${formatPercent(shares[i]!)} of the mix`}
            onPointerEnter={(e) => e.pointerType === "mouse" && engine.hoverStart(s.name)}
            onPointerLeave={(e) => e.pointerType === "mouse" && engine.hoverEnd(s.name)}
          />
        ))}
      </div>
      <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 text-xs text-muted">
        {leaves.map((s, i) => (
          <li key={s.name} className="flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: s.color }} aria-hidden />
            {labels[s.name] ?? s.label}
            <span className="num font-mono text-[11px] text-faint">{formatPercent(shares[i]!)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
