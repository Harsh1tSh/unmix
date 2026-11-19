/**
 * Mirror of /shared/api-contract.md. Change the contract, this file and
 * backend/app/schemas.py together.
 */

export type JobStatus = "queued" | "running" | "done" | "error";
export type Mode = "fast" | "quality";
export type Device = "cpu" | "cuda" | "mps";
export type Stage = "queued" | "preprocess" | "voices" | "music" | "speakers" | "lyrics" | "analysis" | "done" | "error";
export type StemParent = "root" | "voices" | "music";

export interface Features {
  /** the isolated ClearerVoice environment is installed on the server */
  speaker_split: boolean;
  lyrics: boolean;
}

/** What runs where on the server (backend/app/accel.py). */
export interface Accelerator {
  /** BS-RoFormer / HT-Demucs */
  torch: Device;
  /** Kim Vocal 2: fast mode on CPU machines, live mode */
  onnx: "cpu" | "cuda" | "directml";
  /** null until lyrics first run */
  whisper?: "cpu" | "cuda" | null;
  gpu?: string | null;
  /** one line for the UI, e.g. "GPU (Tesla T4)" */
  summary: string;
  notes: string[];
}

export interface Health {
  status: "ok" | "loading" | "error";
  device: Device;
  accelerator: Accelerator;
  /** reached through a tunnel with an access key (e.g. the Colab notebook) */
  remote?: boolean;
  models_ready: boolean;
  models_error?: string | null;
  version: string;
  features: Features;
}

export interface JobCreated {
  job_id: string;
}

export interface Stem {
  name: string;
  label: string;
  parent: StemParent;
  color: string;
  url: string;
}

export interface StageInfo {
  name: Stage;
  label: string;
}

export interface Job {
  id: string;
  status: JobStatus;
  stage: Stage;
  progress: number;
  message: string;
  mode: Mode;
  /** the uploaded file's name, without any path */
  filename: string;
  split_voices: boolean;
  lyrics: boolean;
  /** the stages this job runs, in order (depends on split_voices / lyrics) */
  stages: StageInfo[];
  duration_s?: number;
  error?: string;
  stems?: Stem[];
  timings?: Record<string, number>;
  /** which model produced each stage, e.g. { voices: "Kim_Vocal_2.onnx", music: "htdemucs_6s.yaml" } */
  models?: Record<string, string>;
  /** when done and lyrics were requested */
  lyrics_url?: string;
}

export interface JobEvent {
  status: JobStatus;
  stage: Stage;
  progress: number;
  message: string;
  error?: string;
}

export interface Bands {
  names: string[];
  hz_edges: number[];
  /** [frame][band], 0..1 (dB-scaled, relative to the loudest band-frame in the job) */
  frames: number[][];
}

export interface StemAnalysis {
  /** 2000 points, max |sample| per bucket, 0..1 */
  peaks: number[];
  /** linear RMS per analysis frame */
  rms: number[];
  bands: Bands;
  energy_share: number;
  spectrogram_url: string;
}

export interface Analysis {
  duration_s: number;
  sample_rate: number;
  /** analysis frames per second for `rms` and `bands.frames` */
  frame_rate: number;
  stems: Record<string, StemAnalysis>;
}

export interface LyricWord {
  start: number;
  end: number;
  /** includes Whisper's leading space: join words with "" */
  text: string;
  probability: number;
  /** "speaker_1" | "speaker_2" when voices were split */
  speaker?: string | null;
}

export interface LyricSegment {
  start: number;
  end: number;
  text: string;
  words: LyricWord[];
  speaker?: string | null;
}

export interface Lyrics {
  language: string | null;
  language_probability: number;
  model: string;
  duration_s: number;
  segments: LyricSegment[];
  note?: string | null;
}

export interface CleanState {
  status: "queued" | "running" | "done" | "error";
  progress: number;
  error?: string;
  /** when done: the cleaned voice, 48 kHz mono WAV */
  url?: string;
}
