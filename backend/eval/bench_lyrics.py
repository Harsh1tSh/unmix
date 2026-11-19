"""Compare Whisper sizes / settings for lyric transcription on a separated vocal.

Usage:  uv run python eval/bench_lyrics.py <voices.wav> [model ...]

Prints speed (x realtime), word count and the text, so the default in
app/config.py (UNMIX_LYRICS_MODEL) can be chosen by ear and by speed.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_16k_mono(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return resample_poly(mono, 160, 441).astype(np.float32) if sr == 44100 else mono


def main() -> None:
    path = sys.argv[1]
    models = sys.argv[2:] or ["base", "small"]
    audio = load_16k_mono(path)
    secs = audio.shape[0] / 16000
    from faster_whisper import WhisperModel

    for name in models:
        t0 = time.perf_counter()
        model = WhisperModel(name, device="cpu", compute_type="int8")
        load = time.perf_counter() - t0
        for vad in (False, True):
            t0 = time.perf_counter()
            segments, info = model.transcribe(
                audio, beam_size=5, word_timestamps=True, vad_filter=vad, condition_on_previous_text=False
            )
            segments = list(segments)
            took = time.perf_counter() - t0
            words = sum(len(s.words or []) for s in segments)
            text = " ".join(s.text.strip() for s in segments)
            print(f"== {name} vad={vad} load {load:.1f}s  infer {took:.1f}s ({took / secs:.2f}x rt)  lang {info.language} {info.language_probability:.2f}  words {words}")
            print("   ", text[:400])
        del model


if __name__ == "__main__":
    main()
