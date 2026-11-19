"""Objective quality check: SDR per stem on MUSDB18 clips, measured with museval.

Usage
  # 5 clips from the free 7-second MUSDB18 preview set (run fetch_musdb_sample.py first)
  uv run --group eval python eval/run_eval.py

  # 5 x 30 s clips from MUSDB18-HQ (download from https://zenodo.org/records/3338373)
  uv run --group eval python eval/run_eval.py --hq-root /path/to/musdb18hq --clip-seconds 30

The same separation backend as the server is used (Stage 1 + Stage 2, without
loudness normalisation so estimates are directly comparable with the reference
stems). MUSDB has four reference stems, so our stems are mapped as
  vocals = voices,  drums = drums,  bass = bass,  other = music - drums - bass
(i.e. guitar + piano + other in fast mode).

Results are written to eval/results.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import audio_io  # noqa: E402
from app.config import settings  # noqa: E402
from app.separation import create_backend  # noqa: E402

HERE = Path(__file__).resolve().parent
TARGETS = ["vocals", "drums", "bass", "other"]
SR = 44100


def load_sample_clips(n: int) -> list[tuple[str, np.ndarray, dict[str, np.ndarray]]]:
    import musdb

    db = musdb.DB(root=str(HERE / "data" / "musdb18-7s"), subsets="test")
    out = []
    for track in list(db)[:n]:
        refs = {t: track.targets[t].audio.astype(np.float32) for t in TARGETS}
        out.append((track.name, track.audio.astype(np.float32), refs))
    return out


def load_hq_clips(root: Path, n: int, clip_seconds: float) -> list[tuple[str, np.ndarray, dict[str, np.ndarray]]]:
    test_dir = root / "test" if (root / "test").is_dir() else root
    out = []
    for track_dir in sorted(p for p in test_dir.iterdir() if (p / "mixture.wav").is_file())[:n]:
        vocals, sr = sf.read(track_dir / "vocals.wav", dtype="float32", always_2d=True)
        assert sr == SR, f"{track_dir.name}: expected 44.1 kHz"
        clip = int(clip_seconds * SR)
        # Pick the window where the vocals are most active, so every stem is scored.
        starts = range(0, max(1, vocals.shape[0] - clip), SR * 5)
        start = max(starts, key=lambda s: float(np.sum(vocals[s : s + clip] ** 2)))
        sl = slice(start, start + clip)
        refs = {"vocals": vocals[sl]}
        for t in TARGETS[1:]:
            refs[t] = sf.read(track_dir / f"{t}.wav", dtype="float32", always_2d=True)[0][sl]
        mix = sf.read(track_dir / "mixture.wav", dtype="float32", always_2d=True)[0][sl]
        out.append((f"{track_dir.name} [{start / SR:.0f}s+{clip_seconds:.0f}s]", mix, refs))
    return out


def separate(backend, mix: np.ndarray, mode: str, work: Path) -> dict[str, np.ndarray]:
    mix_path = work / "mix.wav"
    audio_io.write_wav(mix_path, mix, SR, subtype="FLOAT")
    voices = audio_io.match_length(backend.separate_voices(mix_path, work, mode, lambda p: None), mix.shape[0])
    music = mix - voices
    music_path = work / "music.wav"
    audio_io.write_wav(music_path, music, SR, subtype="FLOAT")
    est = backend.separate_music(music_path, work, mode, lambda p: None)
    drums = audio_io.match_length(est["drums"], mix.shape[0])
    bass = audio_io.match_length(est["bass"], mix.shape[0])
    return {"vocals": voices, "drums": drums, "bass": bass, "other": music - drums - bass}


def score(refs: dict[str, np.ndarray], ests: dict[str, np.ndarray]) -> dict[str, float]:
    """Median over 1-second frames of BSSEval v4 SDR (the MUSDB/SiSEC convention)."""
    import museval

    ref = np.stack([refs[t] for t in TARGETS])
    est = np.stack([ests[t] for t in TARGETS])
    sdr, _isr, _sir, _sar = museval.evaluate(ref, est, win=SR, hop=SR)
    return {t: float(np.nanmedian(sdr[i])) for i, t in enumerate(TARGETS)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hq-root", type=Path, help="path to MUSDB18-HQ (contains test/<track>/mixture.wav)")
    parser.add_argument("--clips", type=int, default=5)
    parser.add_argument("--clip-seconds", type=float, default=30.0, help="clip length for --hq-root")
    parser.add_argument("--mode", choices=["fast", "quality"], default="fast")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.hq_root:
        clips, dataset = load_hq_clips(args.hq_root, args.clips, args.clip_seconds), f"MUSDB18-HQ test, {args.clip_seconds:.0f} s clips"
    else:
        clips, dataset = load_sample_clips(args.clips), "MUSDB18 7-second preview set (test subset, lossy AAC stems) - not MUSDB18-HQ"

    backend = create_backend(settings)
    backend.load()

    rows, per_stem = [], {t: [] for t in TARGETS}
    t_start = time.perf_counter()
    audio_s = 0.0
    for name, mix, refs in clips:
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.perf_counter()
            ests = separate(backend, mix, args.mode, Path(tmp))
            took = time.perf_counter() - t0
        scores = score(refs, ests)
        audio_s += mix.shape[0] / SR
        for t in TARGETS:
            per_stem[t].append(scores[t])
        rows.append(f"| {name} | " + " | ".join(f"{scores[t]:.2f}" for t in TARGETS) + f" | {took:.0f} |")
        print(rows[-1], flush=True)
    total = time.perf_counter() - t_start

    models = backend.models_for(args.mode)
    vocal_model, music_model = models["voices"], models["music"]
    md = [
        "# Separation quality (SDR)",
        "",
        f"- Date: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- Dataset: {dataset}",
        f"- Clips: {len(clips)} ({audio_s:.0f} s of audio), mode `{args.mode}`, device `{backend.device}`",
        f"- Models: Stage 1 `{vocal_model}`, Stage 2 `{music_model}`",
        "- Metric: SDR in dB (BSSEval v4 via `museval`, median over 1 s frames). Higher is better;",
        "  roughly: < 3 dB poor, 5-7 dB usable, > 8 dB good. `other` = everything that is not vocals/drums/bass.",
        f"- Wall time: {total:.0f} s including scoring",
        "",
        "| clip | vocals | drums | bass | other | separation s |",
        "|---|---|---|---|---|---|",
        *rows,
        "| **median** | " + " | ".join(f"**{np.median(per_stem[t]):.2f}**" for t in TARGETS) + " | |",
        "| **mean** | " + " | ".join(f"**{np.mean(per_stem[t]):.2f}**" for t in TARGETS) + " | |",
        "",
        "Reproduce: see the docstring of `eval/run_eval.py`.",
        "",
    ]
    (HERE / "results.md").write_text("\n".join(md), encoding="utf-8")
    print("written", HERE / "results.md")


if __name__ == "__main__":
    main()
