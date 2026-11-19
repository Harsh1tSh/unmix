"""End-to-end test with the real neural models. Slow (minutes on CPU) and needs
the weights, so it is excluded by default:

    uv run pytest -m slow -s
"""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import settings as real_settings
from app.main import create_app

from .conftest import upload, wait_for_job

CLIP = Path(__file__).resolve().parents[1] / "eval" / "data" / "songs" / "al_james_schoolboy_7s.wav"


@pytest.mark.slow
def test_real_song_end_to_end(tmp_path):
    if not CLIP.is_file():
        pytest.skip(f"{CLIP} not found (see README: 'Real-song check')")
    settings = replace(real_settings, data_dir=tmp_path / "data", fake_models=False)
    # Reuse the already-downloaded weights instead of the temp data dir.
    import os

    os.environ.setdefault("UNMIX_MODELS_DIR", str(real_settings.models_dir))
    with TestClient(create_app(settings)) as client:
        r = upload(client, CLIP)
        assert r.status_code == 201, r.text
        job = wait_for_job(client, r.json()["job_id"], timeout=1800)
        assert job["status"] == "done", job
        assert [s["name"] for s in job["stems"]] == ["voices", "music", "drums", "bass", "guitar", "piano", "other"]
        assert job["models"]["music"] == "htdemucs_6s.yaml"

        def read(name: str) -> np.ndarray:
            return sf.read(io.BytesIO(client.get(f"/api/jobs/{job['id']}/stems/{name}.wav").content), dtype="float32", always_2d=True)[0]

        mix, voices, music = read("mix"), read("voices"), read("music")
        assert np.max(np.abs(voices + music - mix)) < 4 / 32768
        # A real vocal model must put real energy in both stems of a song with singing.
        shares = {n: s["energy_share"] for n, s in client.get(f"/api/jobs/{job['id']}/analysis").json()["stems"].items()}
        assert shares["voices"] > 0.05 and shares["music"] > 0.05, shares
        # Lyrics are on by default: Whisper must have produced word-timed output (the clip is sung).
        lyrics = client.get(job["lyrics_url"]).json()
        assert lyrics["model"].startswith("faster-whisper/")
        words = [w for seg in lyrics["segments"] for w in seg["words"]]
        assert words and all(0 <= w["start"] < w["end"] <= lyrics["duration_s"] + 0.05 for w in words)
        print("timings:", job["timings"], "models:", job["models"], "shares:", shares)
        print("lyrics:", " / ".join(seg["text"] for seg in lyrics["segments"]))
