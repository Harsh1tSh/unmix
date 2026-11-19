import { useEffect, useRef, useState } from "react";
import { LIVE_SR, LiveSession, type LiveStats } from "../live/session";
import { prepareCanvas } from "../lib/canvas";
import { formatTime } from "../lib/format";
import { BackIcon, HeadphonesIcon, VolumeIcon } from "./icons";
import { Wordmark } from "./UploadScreen";

interface Props {
  onExit: () => void;
  /** run the live recording through the full pipeline */
  onUnmixFully: (file: File) => void;
}

const LANES = [
  { key: "voices", label: "Voices", color: "#FF6B9A" },
  { key: "music", label: "Music", color: "#A78BFA" },
] as const;
const WINDOW_S = 12; // seconds of history shown in each lane

/**
 * Live mode: talk over music and hear it come back separated, ~11 s later on a
 * laptop CPU (5.75 s blocks plus compute). Voices vs. music only: instruments
 * cannot keep up live on a CPU.
 */
export function LiveScreen(props: Props) {
  // Created inside an effect so React's dev-mode double mount gets two
  // independent sessions instead of one that was already disposed.
  const [session, setSession] = useState<LiveSession | null>(null);
  useEffect(() => {
    const s = new LiveSession();
    setSession(s);
    void s.start();
    (window as unknown as { __live?: LiveSession }).__live = s; // for e2e tests
    return () => {
      s.dispose();
      setSession(null);
    };
  }, []);
  return session ? <Live session={session} {...props} /> : null;
}

function Live({ session, onExit, onUnmixFully }: Props & { session: LiveSession }) {
  const [stats, setStats] = useState<LiveStats>(session.stats);
  const [listening, setListening] = useState(false);
  const [solo, setSolo] = useState<"voices" | "music" | null>(null);
  const meterRef = useRef<HTMLDivElement | null>(null);
  const laneRefs = useRef<Record<string, HTMLCanvasElement | null>>({});

  useEffect(() => session.subscribe(setStats), [session]);

  // Esc goes back to the start, like the Back button.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onExit();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onExit]);

  useEffect(() => session.setListening(listening), [session, listening]);
  useEffect(() => session.setLevels(solo === "music" ? 0 : 1, solo === "voices" ? 0 : 1), [session, solo]);

  // Input meter + scrolling lanes, drawn every frame straight from the session's data.
  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const td = new Float32Array(session.inputAnalyser.fftSize);
    let raf = 0;
    const draw = () => {
      session.inputAnalyser.getFloatTimeDomainData(td);
      let sum = 0;
      for (const v of td) sum += v * v;
      const db = 20 * Math.log10(Math.max(Math.sqrt(sum / td.length), 1e-6));
      if (meterRef.current) meterRef.current.style.transform = `scaleX(${Math.min(1, Math.max(0, (db + 60) / 54))})`;

      const head = session.playheadSample();
      for (const lane of LANES) {
        const canvas = laneRefs.current[lane.key];
        const prepared = canvas && prepareCanvas(canvas);
        if (!prepared) continue;
        const { ctx, width, height } = prepared;
        const mid = height / 2;
        ctx.fillStyle = "rgba(255,255,255,0.06)";
        ctx.fillRect(0, mid, width, 1);
        if (head === null) continue;
        // x = width at the playhead; the past scrolls off to the left.
        const from = head - WINDOW_S * LIVE_SR;
        const perPx = (WINDOW_S * LIVE_SR) / width;
        ctx.fillStyle = lane.color;
        for (const b of session.blocks) {
          if (b.start + b.frames < from || b.start > head) continue;
          const data = lane.key === "voices" ? b.voices[0] : b.music[0];
          const x0 = Math.max(0, Math.floor((b.start - from) / perPx));
          const x1 = Math.min(width, Math.ceil((Math.min(b.start + b.frames, head) - from) / perPx));
          for (let x = x0; x < x1; x += 2) {
            const i0 = Math.max(0, Math.floor(from + x * perPx - b.start));
            const i1 = Math.min(b.frames, Math.floor(from + (x + 2) * perPx - b.start));
            let peak = 0;
            for (let i = i0; i < i1; i += 8) peak = Math.max(peak, Math.abs(data[i]!));
            const h = Math.max(1, Math.min(1, peak * 1.6) * (height - 4));
            ctx.fillRect(x, mid - h / 2, 1.5, h);
          }
        }
        ctx.fillStyle = "rgba(255,255,255,0.7)";
        ctx.fillRect(width - 1, 0, 1, height);
      }
      if (!reduce || session.stats.state === "live") raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [session]);

  const ended = stats.state === "ended";
  const recording = ended ? session.recordingFile() : null;
  const behind = (stats.speed ?? 0) > 1 || stats.skippedSeconds > 0;

  return (
    <main className="relative isolate mx-auto flex min-h-dvh max-w-4xl flex-col 2xl:max-w-5xl px-4 py-5 sm:px-6 sm:py-6" data-testid="live-screen" data-state={stats.state}>
      <div className="horizon" aria-hidden />
      <header className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <button type="button" onClick={onExit} className="btn btn-ghost h-9 px-3.5 text-[13px]" title="Stop and go back to the start (Esc)" data-testid="live-back">
            <BackIcon size={13} />
            Back
          </button>
          <span className="hidden sm:block">
            <Wordmark onClick={onExit} />
          </span>
        </div>
        <span className="glass flex h-9 items-center gap-2 rounded-full px-3.5 text-xs text-muted">
          <span className={`h-2 w-2 rounded-full ${stats.state === "live" || stats.state === "waiting" ? "animate-pulse bg-danger" : "bg-faint"}`} aria-hidden />
          {stats.state === "connecting" && "Connecting…"}
          {stats.state === "waiting" && "Listening · first separated audio after ~6 s of input"}
          {stats.state === "live" && "Live"}
          {stats.state === "stopping" && "Finishing the last block…"}
          {stats.state === "ended" && "Ended"}
          {stats.state === "error" && "Stopped"}
        </span>
      </header>

      <section className="mt-12 sm:mt-16">
        <p className="eyebrow rise flex items-center gap-2">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-danger" aria-hidden />
          Live
        </p>
        <h1 className="display mt-4 text-[clamp(2.5rem,6.5vw,5rem)]">
          <span className="reveal-line">
            <span style={{ "--i": 0 } as React.CSSProperties}>Talk over the music.</span>
          </span>
          <span className="reveal-line">
            <span style={{ "--i": 1 } as React.CSSProperties} className="aurora-text">
              Hear it come back apart.
            </span>
          </span>
        </h1>
        <p className="mt-5 max-w-2xl text-sm leading-relaxed text-muted sm:text-base">
          Voices and music are separated in blocks of about 6 seconds, so you hear the result {stats.delay ? `${stats.delay.toFixed(0)} s` : "around 11 s"} later. Nothing is stored on the server; when you stop, you can unmix the recording fully.
        </p>
      </section>

      {stats.error && (
        <p className="mt-6 rounded-2xl border border-danger/35 bg-danger/10 px-4 py-3 text-sm text-danger" role="alert" data-testid="live-error">
          {stats.error}
        </p>
      )}

      <div className="panel mt-8 rounded-3xl p-5 sm:p-7">
        <div className="flex items-center gap-3 text-xs text-muted">
          <span className="eyebrow w-16 shrink-0 text-[10px]">Mic</span>
          <div className="h-[3px] flex-1 overflow-hidden rounded-full bg-white/[0.08]">
            <div ref={meterRef} className="h-full origin-left rounded-full bg-white shadow-[0_0_10px_white]" style={{ transform: "scaleX(0)" }} />
          </div>
          <span className="num w-12 shrink-0 text-right font-mono" data-testid="live-sent">
            {formatTime(stats.sentSeconds)}
          </span>
        </div>
        {LANES.map((lane) => (
          <div key={lane.key} className="mt-5" data-testid={`live-lane-${lane.key}`}>
            <div className="flex items-center gap-2 text-sm">
              <span className="h-2 w-2 rounded-full" style={{ background: lane.color, boxShadow: `0 0 10px ${lane.color}` }} aria-hidden />
              <span className="flex-1 font-display text-lg tracking-[-0.02em]">{lane.label}</span>
              <button
                type="button"
                onClick={() => setSolo(solo === lane.key ? null : lane.key)}
                aria-pressed={solo === lane.key}
                className={`btn h-8 gap-1.5 px-3 text-xs ${solo === lane.key ? "text-ink" : "btn-ghost text-muted"}`}
                style={solo === lane.key ? { background: lane.color } : undefined}
              >
                <HeadphonesIcon size={12} /> Solo
              </button>
            </div>
            <canvas ref={(el) => void (laneRefs.current[lane.key] = el)} className="mt-2.5 block h-16 w-full rounded-xl bg-black/40 ring-1 ring-white/[0.04]" />
          </div>
        ))}
        <dl className="mt-5 flex flex-wrap gap-2 text-xs">
          <Stat label="Blocks" value={String(stats.blocks)} testId="live-blocks" />
          <Stat label="Delay" value={stats.delay ? `${stats.delay.toFixed(1)} s` : "–"} />
          <Stat label="Speed" value={stats.speed ? `${stats.speed.toFixed(2)}× real time` : "–"} warn={(stats.speed ?? 0) > 1} />
          {stats.skippedSeconds > 0 && <Stat label="Skipped" value={`${stats.skippedSeconds.toFixed(0)} s`} warn />}
        </dl>
        {behind && (
          <p className="mt-3 text-xs text-warn" data-testid="live-behind">
            This computer is separating slower than real time, so the delay grows (and audio is skipped if it falls more than ~17 s behind). Close other apps, or use a GPU.
          </p>
        )}
      </div>

      <footer className="mt-8 flex flex-wrap items-center gap-3">
        {!ended && stats.state !== "error" ? (
          <>
            <button
              type="button"
              onClick={() => setListening(!listening)}
              aria-pressed={listening}
              className={`btn h-11 px-5 ${listening ? "btn-primary" : "btn-ghost"}`}
              title="Play the separated audio. Use headphones: on speakers the microphone hears it again."
              data-testid="live-listen"
            >
              <VolumeIcon size={14} muted={!listening} />
              {listening ? "Listening" : "Listen (headphones)"}
            </button>
            <button
              type="button"
              onClick={() => void session.stop()}
              disabled={stats.state === "stopping" || stats.state === "connecting"}
              className="btn btn-danger h-11 px-6"
              data-testid="live-stop"
            >
              Stop
            </button>
          </>
        ) : (
          <>
            {recording && (
              <button type="button" onClick={() => onUnmixFully(recording)} className="btn btn-primary h-11 px-6" data-testid="live-unmix-fully">
                Unmix the recording fully ({formatTime(stats.sentSeconds)})
              </button>
            )}
            <button type="button" onClick={onExit} className="btn btn-ghost h-11 px-5" data-testid="live-exit">
              {recording ? "Discard" : "Back"}
            </button>
          </>
        )}
      </footer>
    </main>
  );
}

function Stat({ label, value, warn = false, testId }: { label: string; value: string; warn?: boolean; testId?: string }) {
  return (
    <div className={`rounded-full border px-3.5 py-2 ${warn ? "border-warn/50" : "border-line"}`}>
      <dt className="inline font-mono text-[10px] tracking-[0.14em] text-faint uppercase">{label} </dt>
      <dd className={`num ml-1 inline ${warn ? "text-warn" : "text-text"}`} data-testid={testId}>
        {value}
      </dd>
    </div>
  );
}
