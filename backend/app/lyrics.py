"""Lyrics: word-timed transcription of the voices stem with faster-whisper.

Why the voices stem and not the mix: Whisper is trained on speech, so the
backing track mostly adds errors. On the separated vocal it hears (nearly) only
the voices. eval/bench_lyrics.py compares settings; the defaults are the result:

* model "small" (int8 on CPU): noticeably better on singing than "base" at
  about the same speed after loading; ~0.25x realtime on a 4-core laptop CPU.
* vad_filter=False: Silero VAD is tuned for speech and clips sung phrases.
* condition_on_previous_text=False: stops the classic Whisper failure of one
  hallucinated line repeating through a whole song.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Settings

log = logging.getLogger("unmix.lyrics")

WHISPER_SR = 16000
# Segments Whisper itself thinks are probably not speech AND has low confidence
# in: typically "Thank you for watching" style hallucinations over instrumentals.
NO_SPEECH_MAX = 0.6
LOGPROB_MIN = -1.0
SILENT_VOICES_RMS = 1e-3  # below this the voices stem is effectively empty


def to_whisper_input(data: np.ndarray, sample_rate: int) -> np.ndarray:
    """(samples, channels) float -> 16 kHz mono float32."""
    from scipy.signal import resample_poly

    mono = data.mean(axis=1) if data.ndim == 2 else data
    if sample_rate != WHISPER_SR:
        g = int(np.gcd(WHISPER_SR, sample_rate))
        mono = resample_poly(mono, WHISPER_SR // g, sample_rate // g)
    return np.ascontiguousarray(mono, dtype=np.float32)


def empty_lyrics(model: str, reason: str) -> dict:
    return {"language": None, "language_probability": 0.0, "model": model, "segments": [], "note": reason}


class WhisperTranscriber:
    def __init__(self, settings: Settings, device: str):
        self.settings = settings
        self.device = "cuda" if device == "cuda" else "cpu"  # CTranslate2 has no MPS backend
        self.note = ""  # why the GPU was not used, if it was not
        self._model = None
        self._path: str | None = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from faster_whisper.utils import download_model

                from .integrity import WHISPER_SMALL_REVISION, verify_whisper_snapshot

                t0 = time.perf_counter()
                root = self.settings.models_dir / "whisper"
                root.mkdir(parents=True, exist_ok=True)
                size = self.settings.lyrics_model
                # Pin the exact Hugging Face revision for the default model and
                # check its files before CTranslate2 reads them.
                revision = WHISPER_SMALL_REVISION if size == "small" else None
                path = download_model(size, cache_dir=str(root), revision=revision)
                verify_whisper_snapshot(Path(path), size, self.settings.models_dir)
                self._path = path
                if self.device == "cuda":
                    # CTranslate2 aborts the whole process when a CUDA library is
                    # missing, so the GPU is checked in a child process first.
                    from .accel import prepare_cuda_libs, whisper_cuda_works

                    ok, error = whisper_cuda_works(Path(path), self.settings.models_dir / "accel.json")
                    if ok:
                        prepare_cuda_libs()
                    else:
                        log.warning("whisper cannot use the GPU here, using the CPU: %s", error)
                        self.note = "Lyrics run on the CPU: Whisper could not use the GPU here (see the server log)."
                        self.device = "cpu"
                self._model = self._make(path)
                log.info("loaded whisper %s on %s in %.1fs", self.settings.lyrics_model, self.device, time.perf_counter() - t0)
            return self._model

    def _make(self, path: str):
        from faster_whisper import WhisperModel

        return WhisperModel(path, device=self.device, compute_type="float16" if self.device == "cuda" else "int8")

    def _to_cpu(self, exc: BaseException) -> None:
        log.error("whisper failed on the GPU, switching to the CPU: %s: %s", type(exc).__name__, exc)
        with self._lock:
            self.device = "cpu"
            self.note = "Lyrics moved to the CPU after a GPU error (see the server log)."
            self._model = self._make(self._path) if self._path else None

    def transcribe(self, voices: np.ndarray, sample_rate: int, progress: Callable[[float], None]) -> dict:
        try:
            return self._transcribe(voices, sample_rate, progress)
        except Exception as exc:
            from .accel import looks_like_gpu_failure

            if self.device != "cuda" or not looks_like_gpu_failure(exc):
                raise
            self._to_cpu(exc)
            return self._transcribe(voices, sample_rate, progress)

    def _transcribe(self, voices: np.ndarray, sample_rate: int, progress: Callable[[float], None]) -> dict:
        model_name = f"faster-whisper/{self.settings.lyrics_model}"
        audio = to_whisper_input(voices, sample_rate)
        duration = audio.shape[0] / WHISPER_SR
        if float(np.sqrt(np.mean(audio**2))) < SILENT_VOICES_RMS:
            return empty_lyrics(model_name, "No voices were found to transcribe.")

        model = self._load()
        progress(0.05)
        segments, info = model.transcribe(
            audio,
            language=self.settings.lyrics_language or None,
            beam_size=5,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        out = []
        for seg in segments:  # the generator does the work; report as it goes
            progress(0.05 + 0.95 * min(1.0, seg.end / max(duration, 1e-6)))
            if seg.no_speech_prob > NO_SPEECH_MAX and seg.avg_logprob < LOGPROB_MIN:
                continue
            words = [
                {"start": round(w.start, 3), "end": round(w.end, 3), "text": w.word, "probability": round(w.probability, 3)}
                for w in (seg.words or [])
                if w.word.strip()
            ]
            if not words:
                continue
            out.append({"start": words[0]["start"], "end": words[-1]["end"], "text": seg.text.strip(), "words": words})
        return {
            "language": info.language,
            "language_probability": round(float(info.language_probability), 3),
            "model": model_name,
            "segments": out,
        }


MAX_LINE_WORDS = 10
LINE_GAP_S = 0.6  # a pause this long starts a new line


def split_lines(segment: dict) -> list[dict]:
    """Break a Whisper segment into display lines: at pauses, and at about
    MAX_LINE_WORDS words (preferring a break after punctuation). Whisper often
    returns one long segment for sung passages."""
    words = segment["words"]
    lines: list[list[dict]] = [[]]
    for i, w in enumerate(words):
        cur = lines[-1]
        if cur:
            gap = w["start"] - cur[-1]["end"]
            ends_clause = cur[-1]["text"].rstrip()[-1:] in ",.;:!?"
            if gap >= LINE_GAP_S or len(cur) >= MAX_LINE_WORDS or (ends_clause and len(cur) >= MAX_LINE_WORDS // 2):
                lines.append([])
        lines[-1].append(w)
    out = []
    for ws in lines:
        if ws:
            out.append({**{k: v for k, v in segment.items() if k not in ("words", "text", "start", "end")},
                        "start": ws[0]["start"], "end": ws[-1]["end"], "text": "".join(w["text"] for w in ws).strip(), "words": ws})
    return out


def _confidence(seg: dict) -> float:
    return sum(w["probability"] for w in seg["words"]) / max(1, len(seg["words"]))


def drop_bleed(segments: list[dict]) -> list[dict]:
    """Separated stems are never perfectly clean: a loud voice can leak into the
    other speaker's stem and get transcribed twice. When two lines from
    different speakers overlap in time and read alike, keep the more confident
    one."""
    from difflib import SequenceMatcher

    keep = [True] * len(segments)
    for i, a in enumerate(segments):
        for j in range(i + 1, len(segments)):
            b = segments[j]
            if a.get("speaker") == b.get("speaker") or b["start"] >= a["end"] or a["start"] >= b["end"]:
                continue
            if SequenceMatcher(None, a["text"].lower(), b["text"].lower()).ratio() > 0.6:
                if _confidence(a) >= _confidence(b):
                    keep[j] = False
                else:
                    keep[i] = False
    return [seg for seg, k in zip(segments, keep) if k]


def merge_transcripts(parts: dict[str | None, dict], model: str) -> dict:
    """Combine per-source transcripts (key = speaker stem name, or None for the
    unsplit voices) into one time-ordered lyrics payload."""
    segments: list[dict] = []
    for speaker, part in parts.items():
        for seg in part["segments"]:
            if speaker:
                seg["speaker"] = speaker
                for w in seg["words"]:
                    w["speaker"] = speaker
            segments.extend(split_lines(seg))
    segments = drop_bleed(segments)
    segments.sort(key=lambda sg: (sg["start"], sg.get("speaker") or ""))
    # Language of the source with the most transcribed words.
    best = max(parts.values(), key=lambda pt: sum(len(sg["words"]) for sg in pt["segments"]), default=None)
    out = {
        "language": best["language"] if best and best["segments"] else None,
        "language_probability": best["language_probability"] if best and best["segments"] else 0.0,
        "model": model,
        "segments": segments,
    }
    if not segments:
        out["note"] = next((pt.get("note") for pt in parts.values() if pt.get("note")), "No words were recognised.")
    return out


class FakeTranscriber:
    """Deterministic placeholder words spread over the parts of the stem that
    have energy. Used by the tests and UNMIX_FAKE_MODELS=1."""

    WORDS = "la la la this is a test of the lyrics view".split()
    # The fake speaker split is a 1 kHz low/high split; give the brighter stem its
    # own words so the two speakers' fake lyrics do not look like bleed.
    WORDS_BRIGHT = "oh here comes the shiny upper part again tonight".split()

    def __init__(self, settings: Settings):
        self.settings = settings

    def transcribe(self, voices: np.ndarray, sample_rate: int, progress: Callable[[float], None]) -> dict:
        duration = voices.shape[0] / sample_rate
        if float(np.sqrt(np.mean(voices**2))) < SILENT_VOICES_RMS:
            return empty_lyrics("fake", "No voices were found to transcribe.")
        mono = voices.mean(axis=1) if voices.ndim == 2 else voices
        zero_crossings = float(np.mean(np.abs(np.diff(np.signbit(mono).astype(np.int8)))))
        vocab = self.WORDS_BRIGHT if zero_crossings > 0.08 else self.WORDS
        segments, t, i = [], 0.2, 0
        while t + 0.4 < duration:
            words = []
            for _ in range(4):
                if t + 0.35 > duration:
                    break
                words.append({"start": round(t, 3), "end": round(t + 0.3, 3), "text": " " + vocab[i % len(vocab)], "probability": 0.9})
                t += 0.35
                i += 1
            if words:
                segments.append({"start": words[0]["start"], "end": words[-1]["end"], "text": "".join(w["text"] for w in words).strip(), "words": words})
            t += 0.3
        progress(1.0)
        return {"language": "en", "language_probability": 1.0, "model": "fake", "segments": segments}


def lyrics_model_files_present(settings: Settings) -> bool:
    root: Path = settings.models_dir / "whisper"
    return root.exists() and any(root.rglob("model.bin"))
