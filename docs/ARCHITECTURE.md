# Architecture

How Unmix is put together and why. For the signal processing behind it (STFT, masks, SDR) read [NOTES.md](../NOTES.md); for the interface design system, [DESIGN.md](../DESIGN.md); for the security model, [SECURITY.md](../SECURITY.md); for the HTTP API, [shared/api-contract.md](../shared/api-contract.md).

## Overview

```
browser (React, Web Audio, WebGL)  --/api-->  FastAPI backend  --> worker thread --> pipeline steps
   upload / record / live                         SQLite job state     preprocess -> voices -> music
   stem board, sample-locked playback             SSE progress         -> speakers -> lyrics -> analysis
```

- `backend/`: FastAPI service. One background worker runs jobs one at a time through a list of pluggable steps (`app/pipeline.py`). Models come from `audio-separator` (BS-RoFormer, MDX-Net Kim Vocal 2, HT-Demucs), ClearerVoice MossFormer2 (in its own virtualenv) and faster-whisper.
- `frontend/`: React + TypeScript + Vite. All stems are decoded into `AudioBuffer`s and played on one `AudioContext` clock, so they stay sample-locked; hover-to-listen is a gain crossfade.
- `shared/api-contract.md`: the contract both sides follow.
- `colab/unmix_gpu.ipynb`: optional, runs the backend on a free Colab GPU behind a tunnel with an access key; the frontend stays local.

## Key decisions and why

1. **Stems are residuals, so they sum to the mix.** `music = mix − voices` and `other = music − (named stems)`, computed in `pipeline.py` rather than taken from the models. audio-separator's own outputs do not sum (it peak-normalises each stem separately, disabled here with `normalization_threshold=1.0`). Exact summation is what makes "hover solos with a crossfade and returns to the full mix" correct with nothing but gains. Demucs's own "vocals" and "other" guesses on the music are folded into `other`.

2. **Only leaf stems are played.** The engine never plays `music` together with its children (that would double it). A parent card's solo/mute/volume act on its descendants (`frontend/src/audio/mixer.ts`, pure and unit-tested). This also means `music.wav` is never downloaded by the player.

3. **Sync by construction.** One `AudioBufferSourceNode` per leaf, all started with the same `(when, offset)` on the `AudioContext` clock, mixed with `GainNode`s only. The context runs at 44.1 kHz so stems are not resampled. No `<audio>` elements, which is also why wavesurfer.js was not used: it wants to own playback. Waveforms are drawn on canvas from the API's peaks.

4. **Stage 1 model depends on device and mode.** Measured here (`eval/vocal_model_bench.md`): BS-RoFormer 19.6× realtime at 12.5 dB vocal SDR, Kim Vocal 2 3.3× realtime at 9.4 dB. On CPU, `fast` uses Kim Vocal 2 and `quality` uses BS-RoFormer; with a GPU both use BS-RoFormer. This departs from the original spec ("Stage 1: BS-RoFormer") for CPU fast mode only, and is reversible with `UNMIX_VOCAL_MODEL_CPU_FAST=""`. Policy lives in `AudioSeparatorBackend.vocal_model_for`.

5. **Quality models load lazily.** Only the fast-mode models load at startup; BS-RoFormer (on CPU) and `htdemucs_ft` load on the first quality job. On an 8 GB machine holding 2 GB of unused weights is costly. `/api/health.models_ready` therefore means "fast mode is ready".

6. **Decode and validate inside `POST /jobs`.** ffmpeg decodes the upload before the response, so empty, corrupt, too-long and too-short files get a 4xx immediately instead of a job that fails later. The pipeline's preprocess step then only loudness-normalises. Browser WebM recordings often have no duration in the header, so probing alone would not be enough.

7. **Static-gain loudness normalisation** (pyloudnorm, −14 LUFS, −1 dBFS ceiling) rather than ffmpeg's dynamic `loudnorm`, which would alter the mix.

8. **`FakeBackend`.** A filter bank with the same interface as the real backend. It makes the API tests and Playwright tests run in seconds without weights. It is deliberately the wrong way to separate audio.

9. **Analysis values are normalised per job**, not per stem, so quiet stems look quiet. Band meters are dB mapped from [−60, 0] to [0, 1] against the loudest band-frame of any stem.

10. **CPU inference settings.** RoFormer overlap 2 (4 on GPU) and Demucs shifts 0 (2 on GPU): each halves compute for a small quality cost. In `AudioSeparatorBackend._new_separator`.

11. **Per-frame UI updates bypass React.** The engine exposes `onFrame(cb)`; waveforms, playheads, meters and the seek bar write to the DOM or canvas directly. React state changes only on discrete events (`useSyncExternalStore` over the engine snapshot).

12. **No Redis, no ORM.** A `queue.Queue`, one thread and raw `sqlite3`, as specified. Schema changes need an entry in `ADDED_COLUMNS` in `db.py` (a test covers the migration).

13. **Contract additions** beyond the original sketch are listed in the changelog at the bottom of `shared/api-contract.md`. All are additive (now version 0.2.0).

14. **A separate venv for ClearerVoice.** `clearvoice` pins `numpy<2` and `rotary_embedding_torch==0.8.3`; audio-separator needs `numpy>=2` and `<0.7`. They cannot share an environment. A subprocess per job also returns MossFormer2's memory to the OS after every split, which matters on 8 GB.

15. **Lyrics per speaker, not on the voice mix.** On the voices stem Whisper transcribed all of the speech and none of the singing (it follows the dominant voice). Transcribing each speaker stem recovered the singer. Cost: one more Whisper pass (~10 s per 20 s of audio).

16. **Lyrics on by default, voice splitting off by default.** Lyrics add little time (~0.5× the clip length); splitting adds ~17× on CPU.

## Where things are

| Concern | File |
|---|---|
| Contract | `shared/api-contract.md` ↔ `backend/app/schemas.py` ↔ `frontend/src/api/types.ts` |
| Routes | `backend/app/main.py` |
| Pipeline steps (add stages here) | `backend/app/pipeline.py` |
| Model choice, inference, progress hook | `backend/app/separation.py` |
| Analysis and spectrogram rendering | `backend/app/analysis.py` |
| Queue, SSE fan-out, cleanup | `backend/app/worker.py` |
| Stem names, labels, colours, band edges, settings | `backend/app/config.py` |
| Playback engine | `frontend/src/audio/engine.ts` |
| Solo/mute/hover logic | `frontend/src/audio/mixer.ts` |
| Screen state machine | `frontend/src/App.tsx` |
| Card (tilt, glow, extra visual slots) | `frontend/src/components/StemCard.tsx` |
| Landing orb, mic recording | `frontend/src/components/Orb.tsx`, `frontend/src/audio/recorder.ts` |
| Split animation | `frontend/src/components/Split.tsx` |
| Lyrics: backend / display / timing logic | `backend/app/lyrics.py` / `frontend/src/components/LyricsPanel.tsx` / `frontend/src/lib/lyrics.ts` |
| Voice splitting / clean-up | `backend/app/speakers.py`, `backend/speakers/run_split.py`, `run_enhance.py` (own venv) |
| GPU choice, measurement, fallback | `backend/scripts/pick_accel.py` (install time), `backend/app/accel.py` (run time) |
| Remote GPU (Colab) | `colab/unmix_gpu.ipynb`, `start.ps1 -Remote`, `GuardMiddleware` access key, `audio_io.flac_copy` |
| Live mode | `backend/app/live.py`, `frontend/src/live/`, `frontend/src/components/LiveScreen.tsx` |
| Browser-only mode | `frontend/src/local/` (FFT, MDX STFT, analysis, worker, runLocal) |
| Request guard, input sniffing, error hygiene | `backend/app/security.py` |
| Model hash pins | `backend/app/integrity.py`, `backend/app/model_hashes.json` |
| Design tokens, component classes | `frontend/src/index.css` (system described in `DESIGN.md`) |
| Space scene: stars + sound sphere, one WebGL2 context (+ 2D fallback) | `frontend/src/visual/SpaceBackdrop.tsx`, `sphere.ts`, `space.ts`, `sceneStore.ts`, `SoundSphere.tsx` (anchor), `fallbackOrb.ts` |

## Known issues and limits

- **CPU speed.** BS-RoFormer costs about 20× the clip length on this laptop. That is why `fast` on CPU uses Kim Vocal 2 instead (see decisions). Quality mode on CPU is impractical beyond short clips. Voice splitting is the slowest stage of all on CPU (~17× the clip length): a 3-minute song takes close to an hour with it on.
- **First use of each option downloads weights** (Whisper small 460 MB, MossFormer2 640 MB) inside that job, with no separate progress for the download. On a slow connection the stage looks stuck for minutes the first time.
- **Lyrics quality.** Speech is near word-perfect; singing is approximate (stretched vowels, effects). Whisper can also invent a line over instrumental passages; low-confidence "no speech" segments are dropped, but not all.
- **Speaker split with only one voice.** If there is only one voice (no talking, or no singing), the model still produces two stems; the second is mostly bleed and artefacts.
- **Recording in the browser** needs `localhost` or HTTPS (getUserMedia). WebM/Opus from MediaRecorder is decoded by ffmpeg on the server and by the browser for the progress waveform.
- **Memory in the browser.** Every leaf stem is decoded to an `AudioBuffer`: 6 stems × 6 min × stereo float32 ≈ 760 MB at the maximum length. Fine on desktop, risky on phones. Streaming playback would need a different engine design.
- **Memory on the server.** Whole stems are held as float32 arrays; peak is a few hundred MB at 6 min, plus the models (BS-RoFormer ≈ 2 GB resident).
- **Cancel is coarse.** A running job only stops at its next progress update, i.e. after the current model chunk.
- **Hover on first load.** Browsers only start audio after a user gesture. After an upload there has been one. After opening a `#job=` link directly, hover does nothing until the user clicks once; the transport bar says "Click play once to enable audio".
- **Touch devices** have no hover; Solo is the equivalent.
- **Piano stem.** `htdemucs_6s`'s piano is weak, and a song without piano yields a near-silent piano card (energy share `<1%`). It is not hidden.
- **Energy shares** of siblings sum to their parent only approximately, because stems are correlated.
- **PCM16 stems** are clipped at ±1.0 on write. The input is normalised to −1 dBFS peak so this is rare, but a clipped stem would break the exact sum for those samples.
- **audio-separator internals.** `separation.py` replaces `tqdm` inside three of its modules to get progress, and sets `output_dir` on its model instance per job. An upgrade past 0.48.0 can break either; the `slow` test and a look at the progress events will show it.
- **Windows quirks** (both in README troubleshooting): `npm run x -- --flag` drops flags in PowerShell 5; Vitest fails when the cwd has a lower-case drive letter, which `npm test` works around via `scripts/vitest.mjs`.
- `backend/eval/data/` (MUSDB sample, test songs) and `backend/data/` (models, jobs) are git-ignored and must be re-downloaded on a new machine. `tests/test_real_models.py` skips if its clip is missing.

## Measured quality

From `backend/eval/results.md`: 5 clips of the MUSDB18 7-second preview set, `fast` mode on CPU (Kim Vocal 2 + `htdemucs_6s`), museval SDR, median over 1 s frames.

| | vocals | drums | bass | other |
|---|---|---|---|---|
| median SDR (dB) | 10.92 | 8.07 | 7.85 | 4.52 |
| mean SDR (dB) | 10.03 | 8.31 | 7.19 | 3.85 |

Read with care: 5 clips of 7 s each, with lossy AAC references. Treat the numbers as a smoke test that the pipeline is wired correctly (all stems clearly above 0 dB, vocals near the published range for this model), not as a benchmark. `other` is low and varies a lot between clips (0.5 to 7.7 dB). That is expected, because it is a residual that collects every model error, and `htdemucs_6s`'s guitar/piano split adds to it.

Stage 1 comparison on this CPU (`backend/eval/vocal_model_bench.md`, 2 clips, plain whole-clip SDR):

| model | × realtime | vocals SDR |
|---|---|---|
| BS-RoFormer (viperx 1297) | 19.6 | 12.5 dB |
| Kim Vocal 2 (MDX-Net) | 3.3 | 9.4 dB |

Quality mode, real run (7 s MUSDB clip, CPU, first quality job so including model loading): BS-RoFormer 136 s + `htdemucs_ft` 55 s = 191 s, i.e. 28× the clip length. Stems aligned, `voices + music − mix` and `drums + bass + other − music` both peak at −84 dBFS.
