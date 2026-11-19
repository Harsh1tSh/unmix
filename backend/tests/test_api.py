"""Endpoint tests. These use the FakeBackend (filters instead of neural models)
so the whole suite runs in seconds; tests/test_real_models.py covers the real
models and is marked `slow`."""

from __future__ import annotations

import io
import json
import time

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import BAND_EDGES_HZ, BAND_NAMES
from app.main import create_app

from .conftest import SR, upload, wait_for_job

FAST_STEMS = ["voices", "music", "drums", "bass", "guitar", "piano", "other"]
MISSING_ID = "0" * 32


# ------------------------------------------------------------------- health


def test_health(client):
    deadline = time.time() + 30  # first import of librosa etc. after a fresh install is slow
    while time.time() < deadline:
        body = client.get("/api/health").json()
        if body["models_ready"]:
            break
        time.sleep(0.05)
    assert body["status"] == "ok"
    assert body["device"] in ("cpu", "cuda", "mps")
    assert body["models_ready"] is True
    accel = body["accelerator"]
    assert accel["torch"] == body["device"] and accel["onnx"] in ("cpu", "cuda", "directml")
    assert accel["summary"] and isinstance(accel["notes"], list)
    assert body["remote"] is False


def test_health_reports_model_failure(settings):
    class Broken:
        device = "cpu"

        def load(self):
            raise RuntimeError("no weights")

        def speaker_split_available(self):
            return False

    with TestClient(create_app(settings, backend=Broken())) as c:
        deadline = time.time() + 10
        while time.time() < deadline and c.get("/api/health").json()["status"] == "loading":
            time.sleep(0.05)
        body = c.get("/api/health").json()
        assert body["status"] == "error"
        assert body["models_ready"] is False
        # Generic message with an error id; the exception text stays in the server log.
        assert "Model loading failed" in body["models_error"] and "error " in body["models_error"]
        assert "no weights" not in body["models_error"]


# ------------------------------------------------------------ upload: happy


def test_create_job_and_get_stems(done_job):
    assert done_job["progress"] == 1.0
    assert done_job["stage"] == "done"
    assert [s["name"] for s in done_job["stems"]] == FAST_STEMS
    by_name = {s["name"]: s for s in done_job["stems"]}
    assert by_name["voices"]["parent"] == "root"
    assert by_name["music"]["parent"] == "root"
    assert all(by_name[n]["parent"] == "music" for n in ["drums", "bass", "guitar", "piano", "other"])
    for stem in done_job["stems"]:
        assert stem["color"].startswith("#") and len(stem["color"]) == 7
        assert stem["url"] == f"/api/jobs/{done_job['id']}/stems/{stem['name']}.wav"
        assert stem["label"]
    assert set(done_job["timings"]) >= {"preprocess", "voices", "music", "analysis", "total"}
    assert set(done_job["models"]) == {"voices", "music"}
    assert abs(done_job["duration_s"] - 3.0) < 0.05


def test_quality_mode_has_four_stem_layout(client, audio_dir):
    r = upload(client, audio_dir / "song.wav", mode="quality")
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert job["mode"] == "quality"
    assert [s["name"] for s in job["stems"]] == ["voices", "music", "drums", "bass", "other"]


def test_split_voices_flag_is_accepted(client, audio_dir):
    r = upload(client, audio_dir / "song.wav", split_voices="true")
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert job["split_voices"] is True


def test_compressed_and_mono_inputs_decode(client, audio_dir):
    for name, ctype in (("song.mp3", "audio/mpeg"), ("song.webm", "audio/webm;codecs=opus"), ("mono.wav", "audio/wav")):
        r = upload(client, audio_dir / name, content_type=ctype)
        assert r.status_code == 201, (name, r.text)
        job = wait_for_job(client, r.json()["job_id"])
        assert job["status"] == "done", (name, job)
        wav = client.get(job["stems"][0]["url"])
        info = sf.info(io.BytesIO(wav.content))
        assert (info.samplerate, info.channels) == (SR, 2), name


def test_browser_recording_without_extension_is_accepted(client, audio_dir):
    """MediaRecorder blobs often arrive as 'blob' with only a content type."""
    with (audio_dir / "song.webm").open("rb") as fh:
        r = client.post("/api/jobs", files={"file": ("blob", fh, "audio/webm")})
    assert r.status_code == 201, r.text


# -------------------------------------------------------------- upload: bad


def test_empty_file_rejected(client, audio_dir):
    r = upload(client, audio_dir / "empty.wav")
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_garbage_file_rejected(client, audio_dir):
    r = upload(client, audio_dir / "garbage.wav")
    assert r.status_code == 422
    assert "supported audio file" in r.json()["detail"].lower()


def test_wrong_type_rejected(client, audio_dir):
    r = upload(client, audio_dir / "notes.txt", content_type="text/plain")
    assert r.status_code == 415
    assert "unsupported" in r.json()["detail"].lower()


def test_too_long_rejected(client, audio_dir):
    r = upload(client, audio_dir / "long.wav")  # 8 s, limit is 6 s in the test settings
    assert r.status_code == 422
    assert "too long" in r.json()["detail"].lower()


def test_too_short_rejected(client, audio_dir):
    r = upload(client, audio_dir / "short.wav")
    assert r.status_code == 422
    assert "too short" in r.json()["detail"].lower()


def test_too_large_rejected(settings, audio_dir):
    from dataclasses import replace

    with TestClient(create_app(replace(settings, max_upload_bytes=1000))) as c:
        r = upload(c, audio_dir / "song.wav")
        assert r.status_code == 413
        assert "too large" in r.json()["detail"].lower()


def test_bad_mode_rejected(client, audio_dir):
    r = upload(client, audio_dir / "song.wav", mode="ludicrous")
    assert r.status_code == 422


def test_missing_file_field_rejected(client):
    assert client.post("/api/jobs", data={"mode": "fast"}).status_code == 422


def test_rejected_upload_leaves_no_files(client, settings, audio_dir):
    upload(client, audio_dir / "garbage.wav")
    upload(client, audio_dir / "long.wav")
    assert list(settings.jobs_dir.iterdir()) == []


def test_silent_audio_fails_with_clear_error(client, audio_dir):
    r = upload(client, audio_dir / "silent.wav")
    assert r.status_code == 201
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "error"
    assert "silent" in job["error"].lower()


# ------------------------------------------------------------------- job


def test_unknown_job_is_404(client):
    for path in ("", "/events", "/analysis", "/stems/voices.wav", "/spectrograms/voices.png"):
        assert client.get(f"/api/jobs/{MISSING_ID}{path}").status_code == 404, path
    assert client.get("/api/jobs/not-a-valid-id").status_code == 404
    assert client.delete(f"/api/jobs/{MISSING_ID}").status_code == 404


def test_events_stream_until_done(client, audio_dir):
    r = upload(client, audio_dir / "song.wav")
    job_id = r.json()["job_id"]
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert events, "no events received"
    assert events[-1]["status"] == "done"
    assert events[-1]["progress"] == 1.0
    for e in events:
        assert set(e) >= {"status", "stage", "progress", "message"}
        assert 0.0 <= e["progress"] <= 1.0
    progresses = [e["progress"] for e in events]
    assert progresses == sorted(progresses), "progress must never go backwards"


def test_events_for_finished_job_returns_final_state(client, done_job):
    with client.stream("GET", f"/api/jobs/{done_job['id']}/events") as resp:
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
    assert len(lines) == 1
    assert json.loads(lines[0][6:])["status"] == "done"


def test_delete_job(client, settings, done_job):
    job_id = done_job["id"]
    assert (settings.jobs_dir / job_id).is_dir()
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert not (settings.jobs_dir / job_id).exists()


def test_files_not_available_before_done(client, settings, audio_dir):
    """A queued job answers 409 (not 404) for its outputs."""
    from dataclasses import replace

    slow = replace(settings, fake_delay_s=1.5, data_dir=settings.data_dir.parent / "slow")
    with TestClient(create_app(slow)) as c:
        job_id = upload(c, audio_dir / "song.wav").json()["job_id"]
        assert c.get(f"/api/jobs/{job_id}/analysis").status_code == 409
        assert c.get(f"/api/jobs/{job_id}/stems/voices.wav").status_code == 409
        body = c.get(f"/api/jobs/{job_id}").json()
        assert body["status"] in ("queued", "running")
        assert "stems" not in body
        assert c.delete(f"/api/jobs/{job_id}").status_code == 204


# ----------------------------------------------------------------- stems


def _read(client, url) -> np.ndarray:
    r = client.get(url)
    assert r.status_code == 200, r.text
    data, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=True)
    assert sr == SR
    return data


def test_stem_audio_format_and_alignment(client, done_job):
    lengths = set()
    for stem in done_job["stems"]:
        r = client.get(stem["url"])
        assert r.headers["content-type"] == "audio/wav"
        info = sf.info(io.BytesIO(r.content))
        assert (info.samplerate, info.channels, info.subtype) == (SR, 2, "PCM_16")
        lengths.add(info.frames)
    assert len(lengths) == 1, f"stems must be sample-aligned, got lengths {lengths}"


def test_stems_sum_back_to_mix(client, done_job):
    """The core invariant the synced player relies on:
    voices + music == mix, and the music children sum to music."""
    urls = {s["name"]: s["url"] for s in done_job["stems"]}
    mix = _read(client, f"/api/jobs/{done_job['id']}/stems/mix.wav")
    voices, music = _read(client, urls["voices"]), _read(client, urls["music"])
    tol = 4 / 32768  # a few LSBs of 16-bit rounding
    assert np.max(np.abs(voices + music - mix)) < tol
    children = sum(_read(client, urls[n]) for n in ["drums", "bass", "guitar", "piano", "other"])
    assert np.max(np.abs(children - music)) < 6 / 32768


def test_stem_range_request(client, done_job):
    url = done_job["stems"][0]["url"]
    full = client.get(url)
    assert full.headers.get("accept-ranges") == "bytes"
    part = client.get(url, headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 100-199/{len(full.content)}"
    assert part.content == full.content[100:200]
    assert client.get(url, headers={"Range": f"bytes={len(full.content) + 10}-"}).status_code == 416


def test_stem_download_disposition(client, done_job):
    r = client.get(done_job["stems"][0]["url"], params={"download": "true"})
    assert "attachment" in r.headers["content-disposition"]
    # Starlette percent-encodes non-token characters (RFC 5987).
    assert "song%20-%20voices.wav" in r.headers["content-disposition"]


def test_unknown_stem_is_404(client, done_job):
    base = f"/api/jobs/{done_job['id']}/stems"
    assert client.get(f"{base}/kazoo.wav").status_code == 404
    assert client.get(f"{base}/voices.mp3").status_code == 404
    assert client.get(f"{base}/..%2F..%2Funmix.sqlite3").status_code == 404


# -------------------------------------------------------------- analysis


def test_analysis_shape(client, done_job):
    r = client.get(f"/api/jobs/{done_job['id']}/analysis")
    assert r.status_code == 200
    a = r.json()
    assert abs(a["duration_s"] - 3.0) < 0.05
    assert a["sample_rate"] == SR
    assert a["frame_rate"] == 20
    assert list(a["stems"]) == FAST_STEMS
    n_frames = None
    for name, s in a["stems"].items():
        assert len(s["peaks"]) == 2000, name
        assert 0.0 <= min(s["peaks"]) and max(s["peaks"]) <= 1.0
        assert s["bands"]["names"] == BAND_NAMES
        assert s["bands"]["hz_edges"] == BAND_EDGES_HZ
        frames = s["bands"]["frames"]
        n_frames = n_frames or len(frames)
        assert len(frames) == n_frames == len(s["rms"]), name
        assert abs(n_frames - a["duration_s"] * a["frame_rate"]) <= 2
        assert all(len(f) == 7 for f in frames)
        flat = [v for f in frames for v in f]
        assert 0.0 <= min(flat) and max(flat) <= 1.0
        assert 0.0 <= s["energy_share"] <= 1.5
        assert s["spectrogram_url"] == f"/api/jobs/{done_job['id']}/spectrograms/{name}.png"


def test_analysis_bands_reflect_content(client, done_job):
    """The fake 'bass' stem is a 200 Hz low-pass, so its energy must sit in the
    low bands -- checks the band edges are wired to the right FFT bins."""
    a = client.get(f"/api/jobs/{done_job['id']}/analysis").json()
    mean = np.mean(np.array(a["stems"]["bass"]["bands"]["frames"]), axis=0)
    assert int(np.argmax(mean)) == BAND_NAMES.index("bass")  # the 80 Hz note
    assert mean[BAND_NAMES.index("bass")] > mean[BAND_NAMES.index("brilliance")] + 0.3
    shares = {n: s["energy_share"] for n, s in a["stems"].items()}
    assert abs(shares["voices"] + shares["music"] - 1.0) < 0.35


def test_spectrogram_png(client, done_job):
    from PIL import Image

    r = client.get(f"/api/jobs/{done_job['id']}/spectrograms/bass.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content))
    assert img.height == 96 and 10 < img.width <= 1200
    assert client.get(f"/api/jobs/{done_job['id']}/spectrograms/kazoo.png").status_code == 404


# ------------------------------------------------------------------ misc


def test_cors_allows_vite_origin(client):
    r = client.options("/api/jobs", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    r = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_expired_jobs_are_cleaned_up(client, settings, done_job):
    manager = client.app.state.manager
    assert manager.cleanup_expired() == 0
    manager.store.update(done_job["id"])  # no-op, keeps created_at
    with manager.store._conn() as conn:
        conn.execute("UPDATE jobs SET created_at = created_at - ? WHERE id = ?", (25 * 3600, done_job["id"]))
    assert manager.cleanup_expired() == 1
    assert client.get(f"/api/jobs/{done_job['id']}").status_code == 404
    assert not (settings.jobs_dir / done_job["id"]).exists()


def test_jobs_interrupted_by_restart_are_marked_failed(settings, audio_dir):
    from dataclasses import replace

    slow = replace(settings, fake_delay_s=30.0)
    with TestClient(create_app(slow)) as c:
        job_id = upload(c, audio_dir / "song.wav").json()["job_id"]
    with TestClient(create_app(settings)) as c:
        body = c.get(f"/api/jobs/{job_id}").json()
        assert body["status"] == "error"
        assert "restart" in body["error"].lower()


def test_database_from_older_version_is_migrated(settings, audio_dir):
    """A jobs table created before a column was added must keep working."""
    import sqlite3

    settings.data_dir.mkdir(parents=True)
    conn = sqlite3.connect(settings.db_path)
    conn.execute(
        "CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL DEFAULT 'queued', progress REAL NOT NULL DEFAULT 0,"
        " message TEXT NOT NULL DEFAULT '', error TEXT, mode TEXT NOT NULL, split_voices INTEGER NOT NULL DEFAULT 0, original_name TEXT NOT NULL DEFAULT '',"
        " duration_s REAL, stems_json TEXT, timings_json TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL)"
    )
    conn.commit()
    conn.close()
    with TestClient(create_app(settings)) as c:
        r = upload(c, audio_dir / "song.wav")
        assert r.status_code == 201, r.text
        assert wait_for_job(c, r.json()["job_id"])["status"] == "done"
