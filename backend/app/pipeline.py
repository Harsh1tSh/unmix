"""The separation pipeline as an ordered list of pluggable steps.

To add a stage: subclass ``Step``, read/write audio through ``PipelineContext``
and add it to ``build_pipeline``. Progress
weights are relative; the runner maps them onto the job's overall 0..1.

Stem bookkeeping lives on the context:
    ctx.audio[name]   float32 (samples, 2) arrays for stems still needed in memory
    ctx.stems         ordered list of stem names that were written to disk
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import analysis, audio_io
from .config import STEMS, Settings
from .separation import SeparationBackend, music_stem_names

log = logging.getLogger("unmix.pipeline")

ReportFn = Callable[[str, float, str], None]  # (stage, overall_progress, message)


@dataclass
class PipelineContext:
    job_id: str
    job_dir: Path
    mode: str
    split_voices: bool
    settings: Settings
    backend: SeparationBackend
    lyrics: bool = False
    audio: dict[str, np.ndarray] = field(default_factory=dict)
    stems: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    mix_energy: float = 0.0
    features: dict[str, analysis.StemFeatures] = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def decoded_path(self) -> Path:
        return self.job_dir / "decoded.wav"

    @property
    def mix_path(self) -> Path:
        return self.job_dir / "mix.wav"

    @property
    def stems_dir(self) -> Path:
        return self.job_dir / "stems"

    @property
    def work_dir(self) -> Path:
        return self.job_dir / "work"

    def save_stem(self, name: str, data: np.ndarray, *, keep: bool = False) -> None:
        """Write a finished stem (16-bit WAV), analyse it, optionally keep it in RAM."""
        audio_io.write_wav(self.stems_dir / f"{name}.wav", data, self.settings.sample_rate)
        self.features[name] = analysis.extract_features(data, self.settings)
        if name not in self.stems:
            self.stems.append(name)
        if keep:
            self.audio[name] = data


class Step:
    name: str = "step"  # machine name, sent to clients as `stage`
    label: str = "Working"  # human message
    weight: float = 1.0  # relative share of total runtime (for the progress bar)

    def cost(self, ctx: PipelineContext) -> float:
        """Expected runtime relative to the other steps; drives the progress bar.

        Default: `weight`. Steps whose cost depends on mode or device override
        this with seconds-per-second-of-audio measured on a 4-core laptop CPU
        (see README "How long it takes"). Only the ratios matter.
        """
        return self.weight

    def enabled(self, ctx: PipelineContext) -> bool:
        return True

    def run(self, ctx: PipelineContext, progress: Callable[[float, str | None], None]) -> None:
        raise NotImplementedError


class PreprocessStep(Step):
    """Stage 0. The upload was already decoded to 44.1 kHz stereo float WAV (and
    length-checked) by the API so bad files fail fast; here we loudness-normalise."""

    name = "preprocess"
    label = "Normalising loudness"
    weight = 0.02

    def run(self, ctx, progress):
        s = ctx.settings
        data, sr = audio_io.read_wav(ctx.decoded_path)
        assert sr == s.sample_rate, f"decoded file has sample rate {sr}"
        data, info = audio_io.loudness_normalize(data, sr, target_lufs=s.target_lufs, peak_ceiling=s.peak_ceiling)
        ctx.info["loudness"] = info
        audio_io.write_wav(ctx.mix_path, data, sr, subtype="FLOAT")
        ctx.audio["mix"] = data
        ctx.duration_s = data.shape[0] / sr
        ctx.mix_energy = float(np.sum(data.astype(np.float64) ** 2))
        ctx.decoded_path.unlink(missing_ok=True)


class VoicesStep(Step):
    """Stage 1: voices vs. everything else (BS-RoFormer; a lighter MDX-Net model
    for mode=fast on CPU-only machines, see AudioSeparatorBackend.vocal_model_for)."""

    name = "voices"
    label = "Separating voices from music"

    def cost(self, ctx):
        # BS-RoFormer ~20x realtime on CPU, Kim Vocal 2 ~2x (vocal_model_bench.md)
        return 20.0 if "roformer" in ctx.backend.models_for(ctx.mode)["voices"].lower() else 2.0

    def run(self, ctx, progress):
        mix = ctx.audio["mix"]
        ctx.info["models"] = ctx.backend.models_for(ctx.mode)
        voices = ctx.backend.separate_voices(ctx.mix_path, ctx.work_dir, ctx.mode, lambda p: progress(p * 0.97, None))
        voices = audio_io.match_length(voices, mix.shape[0])
        music = mix - voices  # residual => voices + music == mix, exactly
        progress(0.98, "Writing voice and music stems")
        # Stage 3 and the lyrics step both read the voices stem again.
        ctx.save_stem("voices", voices, keep=ctx.split_voices or ctx.lyrics)
        ctx.save_stem("music", music, keep=True)
        audio_io.write_wav(ctx.work_dir / "music-f32.wav", music, ctx.settings.sample_rate, subtype="FLOAT")
        ctx.audio.pop("mix", None)


class MusicStep(Step):
    """Stage 2: split the music (HT-Demucs 6-stem, or htdemucs_ft in quality mode)."""

    name = "music"
    label = "Splitting music into instruments"

    def cost(self, ctx):
        return 8.0 if ctx.mode == "quality" else 1.4  # htdemucs_ft is a bag of 4 models

    def run(self, ctx, progress):
        music = ctx.audio["music"]
        est = ctx.backend.separate_music(ctx.work_dir / "music-f32.wav", ctx.work_dir, ctx.mode, lambda p: progress(p * 0.95, None))
        progress(0.96, "Writing instrument stems")
        other = music.copy()
        for name in music_stem_names(ctx.mode):
            stem = audio_io.match_length(est[name], music.shape[0])
            other -= stem
            ctx.save_stem(name, stem)
            del stem
        # "other" is the residual: whatever the named stems did not claim
        # (including the model's own leftover "vocals" guess).
        ctx.save_stem("other", other)
        ctx.audio.pop("music", None)


class SpeakerSplitStep(Step):
    """Stage 3: split `voices` into speaker_1 / speaker_2 (MossFormer2_SS_16K in
    its own venv, see app/speakers.py). The two always sum to `voices`."""

    name = "speakers"
    label = "Telling the voices apart"
    weight = 17.0  # MossFormer2 on CPU: ~17x realtime (342 s for a 20 s clip)

    def run(self, ctx, progress):
        voices = ctx.audio["voices"]
        s1, s2 = ctx.backend.separate_speakers(ctx.stems_dir / "voices.wav", voices, ctx.work_dir, lambda p: progress(p * 0.97, None))
        progress(0.98, "Writing voice stems")
        # The lyrics step transcribes each speaker separately, so keep them in RAM for it.
        ctx.save_stem("speaker_1", audio_io.match_length(s1, voices.shape[0]), keep=ctx.lyrics)
        ctx.save_stem("speaker_2", audio_io.match_length(s2, voices.shape[0]), keep=ctx.lyrics)


class LyricsStep(Step):
    """Word-timed lyrics (faster-whisper, app/lyrics.py).

    Without a speaker split, the voices stem is transcribed. With one, each
    speaker stem is transcribed on its own: Whisper follows only the dominant
    voice in a mixture, so transcribing `voices` would lose the quieter one
    (e.g. the singer under your speech). Words are tagged with their speaker.
    """

    name = "lyrics"
    label = "Transcribing lyrics"
    weight = 0.6  # whisper small int8: ~0.25x realtime per source, plus loading

    def cost(self, ctx):
        return self.weight * (2 if ctx.split_voices else 1)

    def run(self, ctx, progress):
        from .lyrics import merge_transcripts

        speakers = [n for n in ("speaker_1", "speaker_2") if n in ctx.audio]
        sources: dict[str | None, np.ndarray] = {n: ctx.audio.pop(n) for n in speakers} or {None: ctx.audio["voices"]}
        ctx.audio.pop("voices", None)
        parts: dict[str | None, dict] = {}
        model = "unknown"
        for i, (speaker, audio) in enumerate(sources.items()):
            share = 1.0 / len(sources)
            label = f"Transcribing {STEMS[speaker].label.lower()}" if speaker else None
            parts[speaker] = ctx.backend.transcribe(audio, lambda p, _i=i: progress((_i + p) * share, label))
            model = parts[speaker]["model"]
        sources.clear()
        lyrics = merge_transcripts(parts, model)
        lyrics["duration_s"] = round(ctx.duration_s, 3)
        tmp = ctx.job_dir / "lyrics.tmp.json"
        tmp.write_text(json.dumps(lyrics, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        tmp.replace(ctx.job_dir / "lyrics.json")
        ctx.info["lyrics_words"] = sum(len(seg["words"]) for seg in lyrics["segments"])


class AnalysisStep(Step):
    """Stage 4: turn per-stem features into analysis.json + spectrogram PNGs."""

    name = "analysis"
    label = "Analysing frequency content"
    weight = 0.05

    def run(self, ctx, progress):
        base = f"/api/jobs/{ctx.job_id}"
        payload = analysis.build_analysis(
            {name: ctx.features[name] for name in ctx.stems},
            mix_energy=ctx.mix_energy,
            duration_s=ctx.duration_s,
            settings=ctx.settings,
            colors={name: STEMS[name].color for name in ctx.stems},
            spectrogram_dir=ctx.job_dir / "spectrograms",
            spectrogram_url={name: f"{base}/spectrograms/{name}.png" for name in ctx.stems},
        )
        tmp = ctx.job_dir / "analysis.tmp.json"
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(ctx.job_dir / "analysis.json")


def build_pipeline(split_voices: bool, lyrics: bool) -> list[Step]:
    """The steps a job with these options runs, in order."""
    steps: list[Step] = [PreprocessStep(), VoicesStep(), MusicStep()]
    if split_voices:
        steps.append(SpeakerSplitStep())
    if lyrics:
        steps.append(LyricsStep())
    steps.append(AnalysisStep())
    return steps


def planned_stages(split_voices: bool, lyrics: bool) -> list[dict[str, str]]:
    return [{"name": s.name, "label": s.label} for s in build_pipeline(split_voices, lyrics)]


def run_pipeline(ctx: PipelineContext, report: ReportFn) -> dict[str, float]:
    """Run every enabled step in order. Returns wall-clock seconds per stage."""
    steps = [s for s in build_pipeline(ctx.split_voices, ctx.lyrics) if s.enabled(ctx)]
    costs = [s.cost(ctx) for s in steps]
    total_weight = sum(costs)
    timings: dict[str, float] = {}
    done_weight = 0.0

    for step, cost in zip(steps, costs):
        start_frac = done_weight / total_weight
        span = cost / total_weight

        def progress(p: float, message: str | None = None, *, _step=step, _start=start_frac, _span=span) -> None:
            report(_step.name, _start + _span * max(0.0, min(1.0, p)), message or _step.label)

        progress(0.0)
        t0 = time.perf_counter()
        step.run(ctx, progress)
        timings[step.name] = round(time.perf_counter() - t0, 3)
        log.info("job %s | stage %-10s | %7.2fs", ctx.job_id, step.name, timings[step.name])
        done_weight += cost

    shutil.rmtree(ctx.work_dir, ignore_errors=True)
    total = sum(timings.values())
    timings["total"] = round(total, 3)
    rtf = total / ctx.duration_s if ctx.duration_s else 0.0
    log.info("job %s | total %.2fs for %.1fs of audio (%.2fx realtime)", ctx.job_id, total, ctx.duration_s, rtf)
    return timings
