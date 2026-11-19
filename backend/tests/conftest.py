from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio_io import ffmpeg_path  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

SR = 44100


def make_tone_mix(seconds: float, sr: int = SR) -> np.ndarray:
    """A deterministic stereo "song": bass note, mid chord, hi-hat-ish noise and a
    wobbling "voice" tone, so every fake stem has some energy."""
    t = np.arange(int(seconds * sr)) / sr
    rng = np.random.default_rng(0)
    bass = 0.30 * np.sin(2 * np.pi * 80 * t)
    chord = 0.15 * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 554 * t))
    voice = 0.25 * np.sin(2 * np.pi * (900 + 200 * np.sin(2 * np.pi * 3 * t)) * t)
    hats = 0.05 * rng.standard_normal(t.shape[0]) * (np.sin(2 * np.pi * 4 * t) > 0.7)
    left = bass + chord + voice + hats
    right = bass + 0.8 * chord + 1.1 * voice + hats
    return np.stack([left, right], axis=1).astype(np.float32)


@pytest.fixture(scope="session")
def audio_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("audio")
    sf.write(d / "song.wav", make_tone_mix(3.0), SR, subtype="PCM_16")
    sf.write(d / "long.wav", make_tone_mix(8.0), SR, subtype="PCM_16")
    sf.write(d / "short.wav", make_tone_mix(0.4), SR, subtype="PCM_16")
    sf.write(d / "mono.wav", make_tone_mix(2.0)[:, 0], 22050, subtype="PCM_16")
    sf.write(d / "silent.wav", np.zeros((SR * 2, 2), dtype=np.float32), SR, subtype="PCM_16")
    for target, codec in (("song.mp3", ["-c:a", "libmp3lame", "-b:a", "128k"]), ("song.webm", ["-c:a", "libopus", "-b:a", "96k"])):
        subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-i", str(d / "song.wav"), *codec, str(d / target)], check=True)
    (d / "empty.wav").write_bytes(b"")
    (d / "garbage.wav").write_bytes(b"this is definitely not a wav file" * 100)
    (d / "notes.txt").write_text("hello")
    return d


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "data", fake_models=True, max_duration_s=6.0)


@pytest.fixture()
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def upload(client: TestClient, path: Path, *, content_type: str = "audio/wav", **form) -> "object":
    with path.open("rb") as fh:
        return client.post("/api/jobs", files={"file": (path.name, fh, content_type)}, data=form)


def wait_for_job(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s: {body}")


@pytest.fixture()
def done_job(client, audio_dir) -> dict:
    r = upload(client, audio_dir / "song.wav")
    assert r.status_code == 201, r.text
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    return job
