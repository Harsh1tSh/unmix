"""Lyrics and voice-splitting (Stage 3): API behaviour with the fake backend,
plus unit tests of the signal-processing helpers they use."""

from __future__ import annotations

import io
import time

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.lyrics import drop_bleed, merge_transcripts, split_lines, to_whisper_input
from app.main import create_app
from app.separation import FakeBackend
from app.speakers import full_band_split

from .conftest import SR, upload, wait_for_job


def _stem(client, job, name) -> np.ndarray:
    url = next(s["url"] for s in job["stems"] if s["name"] == name)
    return sf.read(io.BytesIO(client.get(url).content), dtype="float32", always_2d=True)[0]


# ------------------------------------------------------------------ lyrics


def test_lyrics_on_by_default(client, done_job):
    assert done_job["lyrics"] is True
    assert done_job["lyrics_url"] == f"/api/jobs/{done_job['id']}/lyrics"
    assert [s["name"] for s in done_job["stages"]] == ["preprocess", "voices", "music", "lyrics", "analysis"]
    assert all(s["label"] for s in done_job["stages"])
    assert "lyrics" in done_job["timings"]

    r = client.get(done_job["lyrics_url"])
    assert r.status_code == 200
    lyr = r.json()
    assert lyr["model"] == "fake"
    assert abs(lyr["duration_s"] - 3.0) < 0.05
    assert lyr["segments"], "fake transcriber always yields words for a non-silent stem"
    last_end = 0.0
    for seg in lyr["segments"]:
        assert seg["words"]
        assert seg["start"] == seg["words"][0]["start"] and seg["end"] == seg["words"][-1]["end"]
        for w in seg["words"]:
            assert 0.0 <= w["start"] < w["end"] <= lyr["duration_s"] + 0.01
            assert w["start"] >= last_end - 1e-6, "words must be in time order"
            last_end = w["end"]
            assert "speaker" not in w or w["speaker"] is None


def test_lyrics_can_be_turned_off(client, audio_dir):
    r = upload(client, audio_dir / "song.wav", lyrics="false")
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert job["lyrics"] is False
    assert "lyrics_url" not in job
    assert "lyrics" not in [s["name"] for s in job["stages"]]
    assert client.get(f"/api/jobs/{job['id']}/lyrics").status_code == 404


def test_lyrics_not_available_before_done(settings, audio_dir):
    from dataclasses import replace

    with TestClient(create_app(replace(settings, fake_delay_s=1.5))) as c:
        job_id = upload(c, audio_dir / "song.wav").json()["job_id"]
        assert c.get(f"/api/jobs/{job_id}/lyrics").status_code == 409
        body = c.get(f"/api/jobs/{job_id}").json()
        assert [s["name"] for s in body["stages"]][-1] == "analysis", "stages are known before the job runs"
        c.delete(f"/api/jobs/{job_id}")


# ---------------------------------------------------------------- speakers


def test_split_voices_adds_speaker_stems_that_sum_to_voices(client, audio_dir):
    r = upload(client, audio_dir / "song.wav", split_voices="true")
    job = wait_for_job(client, r.json()["job_id"])
    assert job["status"] == "done", job
    names = [s["name"] for s in job["stems"]]
    assert names[-2:] == ["speaker_1", "speaker_2"]
    by_name = {s["name"]: s for s in job["stems"]}
    assert by_name["speaker_1"]["parent"] == by_name["speaker_2"]["parent"] == "voices"
    assert [s["name"] for s in job["stages"]] == ["preprocess", "voices", "music", "speakers", "lyrics", "analysis"]

    voices = _stem(client, job, "voices")
    s1, s2 = _stem(client, job, "speaker_1"), _stem(client, job, "speaker_2")
    assert s1.shape == s2.shape == voices.shape
    assert np.max(np.abs(s1 + s2 - voices)) < 4 / 32768

    analysis = client.get(f"/api/jobs/{job['id']}/analysis").json()
    assert {"speaker_1", "speaker_2"} <= set(analysis["stems"])

    lyr = client.get(job["lyrics_url"]).json()
    speakers = {w.get("speaker") for seg in lyr["segments"] for w in seg["words"]}
    assert speakers and speakers <= {"speaker_1", "speaker_2"}, speakers
    assert all(seg.get("speaker") in ("speaker_1", "speaker_2") for seg in lyr["segments"])


def test_split_voices_rejected_when_not_installed(settings, audio_dir):
    class NoSpeakers(FakeBackend):
        def speaker_split_available(self):
            return False

    with TestClient(create_app(settings, backend=NoSpeakers(settings))) as c:
        r = upload(c, audio_dir / "song.wav", split_voices="true")
        assert r.status_code == 422
        assert "not installed" in r.json()["detail"]
        assert c.get("/api/health").json()["features"] == {"speaker_split": False, "lyrics": True}
        assert upload(c, audio_dir / "song.wav").status_code == 201, "plain jobs still work"


def test_health_reports_features(client):
    deadline = time.time() + 10
    while time.time() < deadline and not client.get("/api/health").json()["models_ready"]:
        time.sleep(0.05)
    assert client.get("/api/health").json()["features"] == {"speaker_split": True, "lyrics": True}


# ------------------------------------------------------------- unit tests


def test_full_band_split_follows_the_estimates_and_sums_exactly():
    """Two 'speakers' in different time spans. The 16 kHz estimates say who is
    where; the full-band split must follow them and add back up exactly,
    including content above 8 kHz that the estimates cannot contain."""
    rng = np.random.default_rng(1)
    n = SR * 2
    t = np.arange(n) / SR
    a = np.zeros(n, np.float32)
    b = np.zeros(n, np.float32)
    a[: n // 2] = (0.3 * np.sin(2 * np.pi * 220 * t[: n // 2]) + 0.05 * rng.standard_normal(n // 2)).astype(np.float32)
    b[n // 2 :] = (0.2 * np.sin(2 * np.pi * 330 * t[n // 2 :]) + 0.05 * rng.standard_normal(n - n // 2)).astype(np.float32)
    voices = np.stack([a + b, a + b], axis=1)

    from scipy.signal import resample_poly

    e1 = resample_poly(a, 160, 441).astype(np.float32)
    e2 = resample_poly(b, 160, 441).astype(np.float32)
    s1, s2 = full_band_split(voices, SR, e1, e2)

    assert s1.shape == s2.shape == voices.shape
    assert np.max(np.abs(s1 + s2 - voices)) < 1e-5
    first, second = slice(0, n // 2 - SR // 10), slice(n // 2 + SR // 10, n)
    # speaker_1 is the louder one (a), and it holds a's half, including its >8 kHz noise.
    assert np.mean(s1[first] ** 2) > 20 * np.mean(s2[first] ** 2)
    assert np.mean(s2[second] ** 2) > 20 * np.mean(s1[second] ** 2)


def _w(start, end, text, p=0.9):
    return {"start": start, "end": end, "text": text, "probability": p}


def test_split_lines_breaks_at_pauses_and_length():
    words = [_w(i * 0.3, i * 0.3 + 0.25, f" w{i}") for i in range(14)] + [_w(10.0, 10.3, " after"), _w(10.4, 10.6, " pause")]
    lines = split_lines({"start": 0, "end": 10.6, "text": "...", "words": words, "speaker": "speaker_2"})
    assert [len(line["words"]) for line in lines] == [10, 4, 2]
    assert all(line["speaker"] == "speaker_2" for line in lines)
    assert lines[2]["text"] == "after pause" and lines[2]["start"] == 10.0


def test_merge_tags_speakers_orders_and_drops_bleed():
    you = {"language": "en", "language_probability": 0.99, "model": "m", "segments": [
        {"start": 1.0, "end": 2.0, "text": "hello there", "words": [_w(1.0, 1.4, " hello", 0.95), _w(1.5, 2.0, " there", 0.95)]},
    ]}
    singer = {"language": "en", "language_probability": 0.9, "model": "m", "segments": [
        {"start": 0.2, "end": 0.9, "text": "la la", "words": [_w(0.2, 0.5, " la"), _w(0.6, 0.9, " la")]},
        # faint leak of the speech into the singer stem: same words, same time, less confident
        {"start": 1.05, "end": 2.0, "text": "hello there", "words": [_w(1.05, 1.4, " hello", 0.4), _w(1.5, 2.0, " there", 0.4)]},
    ]}
    out = merge_transcripts({"speaker_1": you, "speaker_2": singer}, "m")
    assert [(sg["speaker"], sg["text"]) for sg in out["segments"]] == [("speaker_2", "la la"), ("speaker_1", "hello there")]
    assert all(w["speaker"] == sg["speaker"] for sg in out["segments"] for w in sg["words"])
    assert out["language"] == "en"


def test_drop_bleed_keeps_different_lines_that_overlap():
    a = {"start": 0, "end": 2, "text": "talking about my day", "speaker": "speaker_1", "words": [_w(0, 2, " x")]}
    b = {"start": 1, "end": 3, "text": "and into the thin air", "speaker": "speaker_2", "words": [_w(1, 3, " y")]}
    assert len(drop_bleed([a, b])) == 2


def test_whisper_input_is_16k_mono():
    x = np.zeros((SR, 2), np.float32)
    y = to_whisper_input(x, SR)
    assert y.dtype == np.float32 and y.ndim == 1 and y.shape[0] == 16000


def test_silent_voices_give_empty_lyrics(client, audio_dir, tmp_path):
    """A clip with no vocal content: the fake 'voices' (a band-pass) of an
    80 Hz tone is near silent, so lyrics must be empty with a note."""
    t = np.arange(SR * 2) / SR
    bass_only = np.stack([0.5 * np.sin(2 * np.pi * 50 * t)] * 2, axis=1).astype(np.float32)
    path = tmp_path / "bass_only.wav"
    sf.write(path, bass_only, SR, subtype="PCM_16")
    job = wait_for_job(client, upload(client, path).json()["job_id"])
    assert job["status"] == "done", job
    lyr = client.get(job["lyrics_url"]).json()
    assert lyr["segments"] == []
    assert lyr["note"]


# ----------------------------------------------------------- voice clean-up


def _wait_clean(client, job_id, stem, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get(f"/api/jobs/{job_id}/clean/{stem}").json()
        if state["status"] in ("done", "error"):
            return state
        time.sleep(0.05)
    raise AssertionError(state)


def test_clean_voice_lifecycle(client, done_job, settings):
    job_id = done_job["id"]
    assert client.get(f"/api/jobs/{job_id}/clean/voices").status_code == 404, "not requested yet"
    r = client.post(f"/api/jobs/{job_id}/clean/voices")
    assert r.status_code == 202
    assert r.json()["status"] in ("queued", "running", "done")
    state = _wait_clean(client, job_id, "voices")
    assert state["status"] == "done", state
    assert state["url"] == f"/api/jobs/{job_id}/clean/voices.wav"
    wav = client.get(state["url"])
    assert wav.status_code == 200 and wav.headers["cache-control"] == "no-store"
    assert "attachment" in wav.headers["content-disposition"] and "clean" in wav.headers["content-disposition"]
    info = sf.info(io.BytesIO(wav.content))
    assert (info.samplerate, info.channels) == (48000, 1)
    assert abs(info.duration - done_job["duration_s"]) < 0.05
    # Asking again returns the finished result instead of redoing it.
    assert client.post(f"/api/jobs/{job_id}/clean/voices").json()["status"] == "done"
    # The stem itself is untouched: the sum-to-mix invariant still holds.
    assert client.get(f"/api/jobs/{job_id}/stems/voices.wav").status_code == 200
    # Deleting the job deletes the cleaned voice too.
    client.delete(f"/api/jobs/{job_id}")
    assert not (settings.jobs_dir / job_id).exists()


def test_clean_only_voice_stems(client, done_job):
    for stem in ("drums", "music", "mix", "../stems/voices", "kazoo"):
        # "../stems/voices" is resolved by the client to the stems route (405): never a clean-up.
        assert client.post(f"/api/jobs/{done_job['id']}/clean/{stem}").status_code in (404, 405), stem


def test_clean_rejected_when_not_installed(settings, audio_dir):
    class NoVoiceModels(FakeBackend):
        def speaker_split_available(self):
            return False

    with TestClient(create_app(settings, backend=NoVoiceModels(settings))) as c:
        job = wait_for_job(c, upload(c, audio_dir / "song.wav").json()["job_id"])
        r = c.post(f"/api/jobs/{job['id']}/clean/voices")
        assert r.status_code == 422
        assert "not installed" in r.json()["detail"]
