# Unmix API contract

Version 0.4.0. Source of truth for the HTTP interface between `/backend`
(FastAPI) and `/frontend` (React). Whoever changes one side updates this file
and both typed mirrors in the same change:

| Mirror | File |
|---|---|
| Backend response models (pydantic) | `backend/app/schemas.py` |
| Frontend types + client | `frontend/src/api/types.ts`, `frontend/src/api/client.ts` |

Conventions

- Base path: `/api`. Dev server: `http://127.0.0.1:8000`. The Vite dev server proxies `/api` to it, so the frontend always uses relative URLs.
- All bodies are JSON unless stated. Every URL the API returns (`Stem.url`, `spectrogram_url`) is a path beginning with `/api/...`.
- Errors: non-2xx with `{ "detail": string }` (human-readable, safe to show). Request-validation errors from FastAPI (bad `mode`, missing `file`) are `422` with `{ "detail": [{ "loc", "msg", "type" }] }`.
- Job ids are 32 lowercase hex characters.
- CORS: any `http://localhost:<port>` / `http://127.0.0.1:<port>` origin is allowed.
- Every endpoint: a `Host` other than localhost / 127.0.0.1 / [::1] gets `421`. `POST` / `DELETE` with a non-local `Origin` (or `Sec-Fetch-Site: cross-site`) get `403`. Request bodies over the upload limit get `413` (also mid-stream). See SECURITY.md.
- Job data (`stems`, `spectrograms`, `analysis`, `lyrics`) is sent with `Cache-Control: no-store`.
- Remote mode (server started with `UNMIX_REMOTE_KEY`, e.g. by the Colab notebook): every request and WebSocket must carry `X-Unmix-Key: <key>`, else `401` (WebSocket: closed with 1008). The frontend's dev/preview proxy adds the header; browsers never see the key.
- Optional fields are omitted (never `null`) in `Job` and `JobEvent`.

## Types

```ts
type JobStatus = "queued" | "running" | "done" | "error";
type Mode      = "fast" | "quality";
type Device    = "cpu" | "cuda" | "mps";

// Pipeline stage names, in order. "speakers" runs only with split_voices, "lyrics" only with lyrics.
// Job.stages lists the ones a given job runs.
type Stage = "queued" | "preprocess" | "voices" | "music" | "speakers" | "lyrics" | "analysis" | "done" | "error";

interface Stem {
  name: string;                         // id, also the file name: voices | music | drums | bass | guitar | piano | other | speaker_1 | speaker_2
  label: string;                        // display name, e.g. "Voices"
  parent: "root" | "voices" | "music";  // tree position
  color: string;                        // "#RRGGBB", the stem's accent colour
  url: string;                          // GET -> audio/wav
}
```

### Stem tree and the sum invariant

```
root
├── voices            everything sung or spoken
│   ├── speaker_1     (only when split_voices=true) the louder of the two voices
│   └── speaker_2     voices − speaker_1
└── music             mix − voices
    ├── drums
    ├── bass
    ├── guitar        (mode=fast only)
    ├── piano         (mode=fast only)
    └── other         music − (the stems above)
```

All stems of a job have the **same sample rate (44 100 Hz), channel count (2) and
exact sample length**, and children sum to their parent:
`voices + music = mix`, `drums + bass + guitar + piano + other = music` and
`speaker_1 + speaker_2 = voices`
(to within 16-bit rounding). A player may therefore either play a parent stem or
the sum of its children, never both.

## Endpoints

### `GET /api/health`

```ts
interface Health {
  status: "ok" | "loading" | "error";   // loading = models still downloading/loading
  device: Device;                        // = accelerator.torch (kept for older clients)
  accelerator: {                         // what runs where (backend/app/accel.py)
    torch: Device;                       // BS-RoFormer, HT-Demucs
    onnx: "cpu" | "cuda" | "directml";   // Kim Vocal 2: fast mode on CPU machines, live mode
    whisper?: "cpu" | "cuda" | null;     // null until lyrics first run
    gpu?: string | null;                 // e.g. "Tesla T4"
    summary: string;                     // one line for the UI, e.g. "GPU (Tesla T4)"
    notes: string[];                     // why a GPU is or is not used; GPU failures
  };
  remote: boolean;                       // started with an access key (reached through a tunnel)
  models_ready: boolean;
  models_error?: string | null;          // set when status = "error"
  version: string;
  features: {
    speaker_split: boolean;  // the isolated ClearerVoice environment is installed
    lyrics: boolean;
  };
}
```

Jobs may be submitted while `status = "loading"`; they wait in the queue.

### `POST /api/jobs` → `201`

`multipart/form-data`:

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | required | wav, mp3, m4a, webm, ogg, opus, flac, aac, mp4, aiff. Accepted when the extension is in that list **or** the part's content type is `audio/*`, `video/webm` or `video/mp4` (browser recordings). |
| `mode` | `"fast"` \| `"quality"` | `"fast"` | Selects the models, see "Modes" below. |
| `split_voices` | bool | `false` | Also split `voices` into `speaker_1` / `speaker_2` (MossFormer2). Slow on CPU (about 17× the clip length). Rejected with `422` when `Health.features.speaker_split` is false. |
| `lyrics` | bool | `true` | Transcribe the voices with word timing (faster-whisper). Adds about 0.25× the clip length on CPU. |

Response: `{ "job_id": string }`

#### Modes

| | Stage 1: voices vs. music | Stage 2: music split | Music stems |
|---|---|---|---|
| `fast`, GPU (cuda/mps) | BS-RoFormer | `htdemucs_6s` | drums, bass, guitar, piano, other |
| `fast`, CPU only | MDX-Net `Kim_Vocal_2` (about 6× faster than BS-RoFormer, about 3 dB lower SDR) | `htdemucs_6s` | drums, bass, guitar, piano, other |
| `quality`, any device | BS-RoFormer | `htdemucs_ft` (bag of 4 models) | drums, bass, other |

The models actually used are reported in `Job.models`. The stem list of a job is
always read from `Job.stems`; clients must not assume a fixed set.

The file is decoded and length-checked before the response, so bad input fails here, not later:

| Status | When |
|---|---|
| `400` | empty file |
| `413` | larger than 200 MB |
| `429` | 3 jobs are already queued or running |
| `415` | not an audio type |
| `422` | not one of the accepted containers (checked by content, not extension); cannot be decoded; longer than 6 min; shorter than 1 s; invalid `mode`; `split_voices` requested but not installed |
| `503` | models failed to load |

### `GET /api/jobs/{id}` → `Job`

```ts
interface Job {
  id: string;
  status: JobStatus;
  stage: Stage;
  progress: number;          // 0..1, monotonic
  message: string;           // human-readable current activity
  mode: Mode;
  filename: string;          // the uploaded file's name, without any path
  split_voices: boolean;
  lyrics: boolean;
  stages: { name: Stage; label: string }[];  // the stages this job runs, in order (always present)
  duration_s?: number;
  error?: string;            // when status = "error"
  stems?: Stem[];            // when status = "done"; ordered parents-first
  timings?: Record<string, number>;  // when done: seconds per stage + "total"
  models?: Record<string, string>;   // when done: { voices: <model file>, music: <model file> }
  lyrics_url?: string;               // when done and lyrics were requested
}
```

`404` unknown id.

### `GET /api/jobs/{id}/events` → `text/event-stream`

Unnamed SSE messages (use `EventSource.onmessage`), each `data:` line is:

```ts
interface JobEvent {
  status: JobStatus;
  stage: Stage;
  progress: number;   // 0..1
  message: string;
  error?: string;
}
```

- The first event is the job's current state, so connecting late (or to a finished job) still works.
- The stream ends after an event with `status` `"done"` or `"error"`. **Close the `EventSource` on that event**, otherwise the browser auto-reconnects.
- `: keepalive` comment lines are sent every 15 s of silence.
- `404` unknown id.

### `GET /api/jobs/{id}/stems/{stem}.wav` → `audio/wav`

16-bit PCM, 44.1 kHz, stereo. Supports HTTP range requests (`Accept-Ranges: bytes`, `206`, `416`).
`{stem}` is a `Stem.name` of the job, or `mix` (the loudness-normalised input, 32-bit float).
`?download=true` adds `Content-Disposition: attachment` with a friendly file name.
With `UNMIX_STEM_TRANSPORT=flac` (remote mode) the same URL serves `audio/flac` instead: bit-identical for the 16-bit stems and about half the size (the float `mix` becomes 24-bit). `?download=true` still returns WAV.
`404` unknown job/stem · `409` job not finished or failed.

### `GET /api/jobs/{id}/spectrograms/{stem}.png` → `image/png`

Mel spectrogram (96 mel bands, 20 Hz–16 kHz, low frequencies at the bottom; up to 1200 px wide, 20 columns per second before downscaling), −80…0 dB relative to the loudest point of any stem in the job, tinted with the stem colour on near-black. `404` / `409` as above.

### `GET /api/jobs/{id}/analysis` → `Analysis`

```ts
interface Analysis {
  duration_s: number;
  sample_rate: number;       // 44100
  frame_rate: number;        // analysis frames per second (20) for `rms` and `bands.frames`
  stems: Record<string, StemAnalysis>;   // keyed by Stem.name
}

interface StemAnalysis {
  peaks: number[];           // 2000 points over the whole duration, max |sample| per bucket, 0..1
  rms: number[];             // one per frame, linear RMS amplitude 0..1
  bands: {
    names: string[];         // ["sub-bass","bass","low-mid","mid","high-mid","presence","brilliance"]
    hz_edges: number[];      // [20,60,250,500,2000,4000,6000,20000] (names.length + 1 edges)
    frames: number[][];      // [frame][band], 0..1
  };
  energy_share: number;      // stem energy / mix energy (0..1; siblings sum to ≈ parent's share)
  spectrogram_url: string;
}
```

`bands.frames[t][b]` is the energy in band `b` at frame `t`, in dB, mapped linearly
from −60 dB → 0 to 0 dB → 1 **relative to the loudest band-frame of any stem in the
job**. Stems are therefore comparable: a quiet stem has small meters. Frame `t`
is centred on time `t / frame_rate` seconds.

`404` unknown id · `409` job not finished or failed.

### `GET /api/jobs/{id}/lyrics` → `Lyrics`

Word-timed transcription of the `voices` stem (faster-whisper, model `small` by default, language auto-detected).

```ts
interface Lyrics {
  language: string | null;         // ISO 639-1, e.g. "en"; null when nothing was transcribed
  language_probability: number;
  model: string;                   // e.g. "faster-whisper/small"
  duration_s: number;
  segments: LyricSegment[];        // in time order; [] when no voices were found
  note?: string | null;            // why segments is empty, when it is
}
interface LyricSegment {
  start: number; end: number;      // seconds, = first word start / last word end
  text: string;
  words: LyricWord[];              // never empty
  speaker?: "speaker_1" | "speaker_2" | null;  // majority speaker, when voices were split
}
interface LyricWord {
  start: number; end: number;      // seconds on the same clock as the stems
  text: string;                    // keeps Whisper's leading space: join words with ""
  probability: number;             // 0..1
  speaker?: "speaker_1" | "speaker_2" | null;  // the speaker stem loudest during the word
}
```

`404` unknown id, or lyrics were not requested · `409` job not finished or failed.

### Voice clean-up: `POST /api/jobs/{id}/clean/{stem}` → `202 CleanState`

Removes background noise and room sound from a voice stem (`voices`, `speaker_1`, `speaker_2`) with MossFormer2 SE 48K, on demand. The result is an extra file, not a stem (enhancement changes the audio, so it would break the sum invariant). Runs on the same queue as jobs, one task at a time.

```ts
interface CleanState {
  status: "queued" | "running" | "done" | "error";
  progress: number;          // 0..1
  error?: string;
  url?: string;              // when done: /api/jobs/{id}/clean/{stem}.wav (48 kHz mono, attachment)
}
```

- `POST` again returns the current state instead of starting over; a finished clean-up is returned as `done`.
- `GET /api/jobs/{id}/clean/{stem}` → `CleanState` (`404` if never requested).
- `GET /api/jobs/{id}/clean/{stem}.wav` → the file (`404` until done). Deleted with the job.
- `404` not a voice stem of this job · `409` job not finished · `422` voice models not installed · `429` too many clean-ups waiting.

### Live mode: `WebSocket /api/live`

Near-real-time voices/music separation. One session at a time; the batch queue waits while it runs. The handshake must carry a local `Origin` and `Host` (else closed with code 1008).

```
client → {"type":"start","sample_rate":44100,"channels":1|2}
server → {"type":"ready","sample_rate":44100,"block_frames":253440,"context_frames":3840}
client → binary frames: interleaved little-endian float32, at most 1 MiB each (send ~4096 frames at a time)
server → per block: {"type":"block","index":i,"start":sample,"frames":n,"compute_s":t,"backlog":b}
         followed by one binary message: float32 planar [voices L | voices R | music L | music R], each n long
server → {"type":"behind","skipped_frames":k}   when more than 3 blocks behind (input skipped)
client → {"type":"stop"}                          the server flushes the last partial block
server → {"type":"end","frames":total}
server → {"type":"error","message":...}           then closes
```

Blocks are 253 440 samples (5.75 s) of new audio; `start` is the position in the input stream, so a client schedules block *i* at `base + start / 44100`. Within a session `voices + music` equals the input exactly. Sessions are limited to 10 minutes; nothing is stored.

### `GET /api/models/Kim_Vocal_2.onnx`

The ONNX model used by the browser-only mode. Only this file is served; it is checked against its pinned SHA-256 first. `Cache-Control: public, max-age=31536000, immutable`. `404` if not downloaded yet.

### `DELETE /api/jobs/{id}` → `204`

Deletes the job and its files. A queued job is skipped; a running job is abandoned at its next progress update. `404` unknown id.

## Lifecycle

- Jobs run one at a time, in submission order.
- Files and the job record are deleted 24 h after creation.
- If the server restarts, unfinished jobs become `status: "error"`.

## Changelog

- 0.4.0: `Health.accelerator` and `Health.remote`; `X-Unmix-Key` access key in remote mode (`401`); FLAC stem transport. All additive.
- 0.3.0: live mode (`/api/live` WebSocket); `/api/models/Kim_Vocal_2.onnx` for the browser-only mode.
- 0.2.1: voice clean-up (`/clean/{stem}`); `Job.filename`; security responses (`403`, `421`, `429`, mid-stream `413`); `no-store` caching for job data.
- 0.2.0: lyrics (`POST lyrics`, `Job.lyrics`, `Job.lyrics_url`, `GET /lyrics`); voice splitting is live (`speaker_1`/`speaker_2`, `Health.features`); `Job.stages`; stage `lyrics`. All additive.
- 0.1.0: initial contract. `mode` also selects the Stage 1 model on CPU-only machines (see Modes). Additions to the original sketch: `Health.models_error`/`version`; `Job.message`/`mode`/`split_voices`/`duration_s`/`timings`/`models`; `JobEvent.status`/`error`; `Analysis.frame_rate`; the `/spectrograms` and `stems/mix.wav` routes; `?download=true`.
