"""Response models. Keep in sync with /shared/api-contract.md and
/frontend/src/api/types.ts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

JobStatus = Literal["queued", "running", "done", "error"]
Mode = Literal["fast", "quality"]
Device = Literal["cpu", "cuda", "mps"]


class Features(BaseModel):
    speaker_split: bool  # the isolated ClearerVoice venv is installed
    lyrics: bool


class Accelerator(BaseModel):
    """What runs where (app/accel.py)."""

    torch: Device  # BS-RoFormer / HT-Demucs
    onnx: Literal["cpu", "cuda", "directml"]  # Kim Vocal 2: fast mode on CPU machines, live mode
    whisper: Literal["cpu", "cuda"] | None = None  # None until lyrics first run
    gpu: str | None = None
    summary: str  # one line for the UI, e.g. "GPU (Tesla T4)"
    notes: list[str] = []


class Health(BaseModel):
    status: Literal["ok", "loading", "error"]
    device: Device  # = accelerator.torch (kept for older clients)
    accelerator: Accelerator
    remote: bool = False  # reached through a tunnel with an access key (e.g. Colab)
    models_ready: bool
    models_error: str | None = None
    version: str
    features: Features


class JobCreated(BaseModel):
    job_id: str


class Stem(BaseModel):
    name: str
    label: str
    parent: Literal["root", "voices", "music"]
    color: str
    url: str


class StageInfo(BaseModel):
    name: str
    label: str


class JobView(BaseModel):
    id: str
    status: JobStatus
    stage: str
    progress: float
    message: str
    mode: Mode
    filename: str  # the uploaded file's name (no path)
    split_voices: bool
    lyrics: bool
    stages: list[StageInfo]
    duration_s: float | None = None
    error: str | None = None
    stems: list[Stem] | None = None
    timings: dict[str, float] | None = None
    models: dict[str, str] | None = None
    lyrics_url: str | None = None


class JobEvent(BaseModel):
    status: JobStatus
    stage: str
    progress: float
    message: str
    error: str | None = None


class Bands(BaseModel):
    names: list[str]
    hz_edges: list[float]
    frames: list[list[float]]


class StemAnalysis(BaseModel):
    peaks: list[float]
    rms: list[float]
    bands: Bands
    energy_share: float
    spectrogram_url: str


class Analysis(BaseModel):
    duration_s: float
    sample_rate: int
    frame_rate: int
    stems: dict[str, StemAnalysis]


class LyricWord(BaseModel):
    start: float
    end: float
    text: str  # includes Whisper's leading space; join words with "" not " "
    probability: float
    speaker: str | None = None  # "speaker_1" / "speaker_2" when voices were split


class LyricSegment(BaseModel):
    start: float
    end: float
    text: str
    words: list[LyricWord]
    speaker: str | None = None


class Lyrics(BaseModel):
    language: str | None
    language_probability: float
    model: str
    duration_s: float
    segments: list[LyricSegment]
    note: str | None = None


class CleanState(BaseModel):
    status: Literal["queued", "running", "done", "error"]
    progress: float = 0.0
    error: str | None = None
    url: str | None = None  # when done: 48 kHz mono WAV
