import type { Lyrics } from "../api/types";

/** mm:ss.xx for LRC */
function lrcTime(t: number): string {
  const cs = Math.max(0, Math.round(t * 100));
  const m = Math.floor(cs / 6000);
  const s = Math.floor((cs % 6000) / 100);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(cs % 100).padStart(2, "0")}`;
}

/** hh:mm:ss,mmm for SRT */
function srtTime(t: number): string {
  const ms = Math.max(0, Math.round(t * 1000));
  const h = Math.floor(ms / 3_600_000);
  const m = Math.floor((ms % 3_600_000) / 60_000);
  const s = Math.floor((ms % 60_000) / 1000);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")},${String(ms % 1000).padStart(3, "0")}`;
}

/** Names line authors only when there is more than one voice. */
function speakerPrefix(lyrics: Lyrics, labels: Record<string, string>) {
  const speakers = new Set(lyrics.segments.map((s) => s.speaker).filter(Boolean));
  return (speaker: string | null | undefined) => (speakers.size > 1 && speaker ? `${labels[speaker] ?? speaker}: ` : "");
}

/** Line-synced LRC, the format karaoke apps and music players read. */
export function toLRC(lyrics: Lyrics, labels: Record<string, string> = {}, title?: string): string {
  const prefix = speakerPrefix(lyrics, labels);
  const head = [title ? `[ti:${title.replace(/[[\]\r\n]/g, " ")}]` : null, `[length:${lrcTime(lyrics.duration_s).slice(0, 5)}]`, "[re:Unmix]"].filter(Boolean);
  const lines = lyrics.segments.map((s) => `[${lrcTime(s.start)}]${prefix(s.speaker)}${s.text.replace(/[\r\n]+/g, " ")}`);
  return [...head, ...lines].join("\n") + "\n";
}

/** SubRip subtitles: each line with its start and end. */
export function toSRT(lyrics: Lyrics, labels: Record<string, string> = {}): string {
  const prefix = speakerPrefix(lyrics, labels);
  return (
    lyrics.segments
      .map((s, i) => `${i + 1}\n${srtTime(s.start)} --> ${srtTime(Math.max(s.end, s.start + 0.2))}\n${prefix(s.speaker)}${s.text.replace(/[\r\n]+/g, " ")}\n`)
      .join("\n")
  );
}

/** Save text as a file (a blob URL: nothing leaves the machine). */
export function downloadText(text: string, filename: string, type = "text/plain"): void {
  const url = URL.createObjectURL(new Blob([text], { type: `${type};charset=utf-8` }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
