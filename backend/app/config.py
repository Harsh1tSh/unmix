"""Central configuration. Every knob can be overridden with an UNMIX_* env var."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return os.environ.get(f"UNMIX_{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class StemMeta:
    name: str
    label: str
    parent: str  # "root" | "voices" | "music"
    color: str


# Single source of truth for stem identity. The frontend receives these through
# the API, so colours/labels only ever need to change here.
STEMS: dict[str, StemMeta] = {
    s.name: s
    for s in [
        StemMeta("voices", "Voices", "root", "#FF6B9A"),
        StemMeta("music", "Music", "root", "#A78BFA"),
        StemMeta("drums", "Drums", "music", "#F59E0B"),
        StemMeta("bass", "Bass", "music", "#EF4444"),
        StemMeta("guitar", "Guitar", "music", "#34D399"),
        StemMeta("piano", "Piano", "music", "#38BDF8"),
        StemMeta("other", "Other", "music", "#94A3B8"),
        # Phase 3 (speaker split) children of "voices":
        StemMeta("speaker_1", "Voice 1", "voices", "#FDA4AF"),
        StemMeta("speaker_2", "Voice 2", "voices", "#C084FC"),
    ]
}

# Extra files that may be fetched through /stems/{name}.wav but are not stems.
EXTRA_AUDIO = {"mix"}

BAND_NAMES = ["sub-bass", "bass", "low-mid", "mid", "high-mid", "presence", "brilliance"]
BAND_EDGES_HZ = [20, 60, 250, 500, 2000, 4000, 6000, 20000]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(BACKEND_DIR / "data"))))

    # --- audio ---
    sample_rate: int = 44100
    target_lufs: float = float(_env("TARGET_LUFS", "-14"))
    peak_ceiling: float = 0.891  # -1 dBFS, leaves headroom so separated stems do not clip
    min_duration_s: float = 1.0
    max_duration_s: float = float(_env("MAX_DURATION_S", "360"))
    max_upload_bytes: int = int(_env("MAX_UPLOAD_MB", "200")) * 1024 * 1024

    # --- models (audio-separator filenames) ---
    vocal_model: str = _env("VOCAL_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
    vocal_model_fallback: str = _env("VOCAL_MODEL_FALLBACK", "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt")
    # CPU-only machines use this lighter model for mode=fast ("" = always BS-RoFormer).
    # See AudioSeparatorBackend.vocal_model_for and eval/vocal_model_bench.md.
    vocal_model_cpu_fast: str = _env("VOCAL_MODEL_CPU_FAST", "Kim_Vocal_2.onnx")
    music_model_fast: str = _env("MUSIC_MODEL_FAST", "htdemucs_6s.yaml")
    music_model_quality: str = _env("MUSIC_MODEL_QUALITY", "htdemucs_ft.yaml")
    # Lyrics (faster-whisper on the voices stem). See app/lyrics.py for why these defaults.
    lyrics_model: str = _env("LYRICS_MODEL", "small")
    lyrics_language: str = _env("LYRICS_LANGUAGE", "")  # "" = auto-detect, else e.g. "en"
    # Stage 3: python of the isolated ClearerVoice venv ("" = backend/speakers/.venv).
    speakers_python: str = _env("SPEAKERS_PYTHON", "")
    # Fake DSP "models" for tests / frontend development without the heavy weights.
    fake_models: bool = _env_bool("FAKE_MODELS", False)
    fake_delay_s: float = float(_env("FAKE_DELAY_S", "0"))

    # --- analysis ---
    analysis_fps: int = 20
    analysis_n_fft: int = 4096
    peaks_points: int = 2000
    spectrogram_mels: int = 96
    spectrogram_max_width: int = 1200

    # --- security ---
    max_waiting_jobs: int = int(_env("MAX_WAITING_JOBS", "3"))  # queued + running; more -> 429
    allowed_hosts: tuple[str, ...] = tuple(
        h.strip().lower() for h in _env("ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],testserver").split(",") if h.strip()
    )
    api_docs: bool = _env_bool("API_DOCS", False)  # /docs loads Swagger UI from a CDN, so it is opt-in
    # Remote GPU mode (colab/): when set, every request must carry this key in
    # X-Unmix-Key. The local frontend's proxy adds it; browsers never see it.
    remote_key: str = _env("REMOTE_KEY", "")
    # "flac" serves stems losslessly at about half the size (for a remote server
    # behind a slow link); downloads stay WAV. "wav" = no re-encoding.
    stem_transport: str = _env("STEM_TRANSPORT", "wav")

    # --- housekeeping ---
    job_ttl_hours: float = float(_env("JOB_TTL_HOURS", "24"))
    cors_origins: tuple[str, ...] = tuple(
        o.strip()
        for o in _env(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
        ).split(",")
        if o.strip()
    )

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def models_dir(self) -> Path:
        return Path(_env("MODELS_DIR", str(self.data_dir / "models")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "unmix.sqlite3"


settings = Settings()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".oga", ".opus", ".flac", ".aac", ".mp4", ".aiff", ".aif"}
