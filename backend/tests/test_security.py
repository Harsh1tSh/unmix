"""Security regression tests: each one replays an attack found in the review."""

from __future__ import annotations

import io
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.audio_io import ffmpeg_path
from app.integrity import IntegrityError, verify
from app.main import create_app
from app.security import scrub, sniff_audio_format

from .conftest import SR, upload, wait_for_job


@pytest.fixture()
def secret(tmp_path) -> Path:
    """A private file elsewhere on disk that an upload must never be able to read."""
    path = tmp_path / "private_recording.wav"
    t = np.arange(SR * 3) / SR
    sf.write(path, (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SR)
    return path


# ----------------------------------------------------- malicious media files


@pytest.mark.parametrize(
    "name,body",
    [
        ("evil.wav", "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10.0,\nfile:{secret}\n#EXT-X-ENDLIST\n"),
        ("evil.m3u8", "#EXTM3U\n#EXT-X-TARGETDURATION:10\n#EXTINF:10.0,\n{secret}\n#EXT-X-ENDLIST\n"),
        ("evil.mp3", "ffconcat version 1.0\nfile '{secret}'\n"),
        ("evil.webm", "concat:{secret}|{secret}"),
        ("evil.ogg", "<?xml version='1.0'?><smil><body><audio src='{secret}'/></body></smil>"),
    ],
)
def test_playlists_and_references_are_refused(client, secret, name, body):
    """Playlist/concat/reference files must not be decoded at all: they could
    make ffmpeg read other local files and serve them back as 'your' audio."""
    data = body.format(secret=secret.as_posix()).encode()
    r = client.post("/api/jobs", files={"file": (name, data, "audio/wav")})
    assert r.status_code in (415, 422), r.text
    assert "private_recording" not in r.text


def test_unknown_container_with_audio_extension_is_refused(client, tmp_path):
    # A real AVI (not an allowed audio container) renamed to .wav.
    avi = tmp_path / "video.avi"
    subprocess.run([ffmpeg_path(), "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=2", "-y", str(avi)], check=True)
    r = upload(client, avi.rename(tmp_path / "sneaky.wav"))
    assert r.status_code == 422
    assert "supported audio file" in r.json()["detail"]


@pytest.mark.parametrize(
    "header,expected",
    [
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", "wav"),
        (b"FORM\x00\x00\x00\x00AIFFCOMM", "aiff"),
        (b"fLaC\x00\x00\x00\x22" + b"\x00" * 8, "flac"),
        (b"OggS\x00\x02" + b"\x00" * 10, "ogg"),
        (b"\x1a\x45\xdf\xa3\x01\x00\x00\x00" + b"\x00" * 8, "matroska"),
        (b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 8, "mov"),
        (b"ID3\x04\x00\x00" + b"\x00" * 10, "mp3"),
        (b"\xff\xfb\x90\x00" + b"\x00" * 12, "mp3"),
        (b"\xff\xf1\x50\x80" + b"\x00" * 12, "aac"),
        (b"#EXTM3U\n#EXT-X-" + b"\x00" * 8, None),
        (b"ffconcat version 1.0\n", None),
        (b"RIFF\x00\x00\x00\x00AVI LIST", None),
        (b"\x00" * 16, None),
    ],
)
def test_sniffing(tmp_path, header, expected):
    p = tmp_path / "f"
    p.write_bytes(header + b"\x00" * 64)
    assert sniff_audio_format(p) == expected


# ------------------------------------------------------- CSRF / rebinding


def _wav_bytes(seconds=2.0) -> bytes:
    buf = io.BytesIO()
    t = np.arange(int(SR * seconds)) / SR
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), SR, format="WAV")
    return buf.getvalue()


def test_foreign_origin_cannot_create_or_delete_jobs(client, done_job):
    files = {"file": ("a.wav", _wav_bytes(), "audio/wav")}
    r = client.post("/api/jobs", files=files, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.post("/api/jobs", files=files, headers={"Origin": "null"})
    assert r.status_code == 403
    r = client.post("/api/jobs", files=files, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    r = client.delete(f"/api/jobs/{done_job['id']}", headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert client.get(f"/api/jobs/{done_job['id']}").status_code == 200, "the job must survive"


def test_local_origins_and_scripts_are_allowed(client):
    files = {"file": ("a.wav", _wav_bytes(), "audio/wav")}
    assert client.post("/api/jobs", files=files, headers={"Origin": "http://localhost:5173"}).status_code == 201
    assert client.post("/api/jobs", files=files).status_code == 201  # curl: no Origin


def test_dns_rebinding_host_is_refused(client):
    assert client.get("/api/health", headers={"Host": "attacker.example"}).status_code == 421
    assert client.get("/api/health", headers={"Host": "attacker.example:8000"}).status_code == 421
    for ok in ("127.0.0.1:8000", "localhost:5173", "[::1]:8000"):
        assert client.get("/api/health", headers={"Host": ok}).status_code == 200, ok


def test_cors_preflight_from_foreign_origin_gets_no_grant(client):
    r = client.options("/api/jobs/" + "0" * 32, headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "DELETE"})
    assert "access-control-allow-origin" not in r.headers


# ------------------------------------------------------- resource limits


def test_oversized_body_is_refused_before_it_is_stored(settings):
    small = replace(settings, max_upload_bytes=200_000)
    with TestClient(create_app(small)) as c:
        r = c.post("/api/jobs", files={"file": ("big.wav", b"\x00" * 2_000_000, "audio/wav")})
        assert r.status_code == 413

        # Chunked (no Content-Length): the cap is enforced while streaming.
        def chunks():
            for _ in range(40):
                yield b"x" * 100_000

        r = c.post("/api/jobs", content=chunks(), headers={"Content-Type": "multipart/form-data; boundary=x"})
        assert r.status_code == 413
        assert not any(small.jobs_dir.iterdir()) if small.jobs_dir.exists() else True


def test_waiting_queue_is_capped(settings, audio_dir):
    slow = replace(settings, fake_delay_s=5.0, max_waiting_jobs=2)
    with TestClient(create_app(slow)) as c:
        codes = [upload(c, audio_dir / "song.wav").status_code for _ in range(4)]
        assert codes[:2] == [201, 201]
        assert codes[2:] == [429, 429]
        r = upload(c, audio_dir / "song.wav")
        assert "already waiting" in r.json()["detail"]


# ----------------------------------------------------- headers and leaks


def test_security_headers_and_private_caching(client, done_job):
    h = client.get("/api/health").headers
    assert h["x-content-type-options"] == "nosniff"
    assert h["x-frame-options"] == "DENY"
    assert h["referrer-policy"] == "no-referrer"
    assert h["cross-origin-resource-policy"] == "same-origin"
    assert "default-src 'none'" in h["content-security-policy"]
    for url in (done_job["stems"][0]["url"], f"/api/jobs/{done_job['id']}/analysis", done_job["lyrics_url"]):
        assert client.get(url).headers["cache-control"] == "no-store", url


def test_api_docs_are_off_by_default(client):
    """/docs pulls Swagger UI from a CDN; it is opt-in (UNMIX_API_DOCS=1)."""
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_failed_job_does_not_leak_internals(settings, audio_dir):
    from app.separation import FakeBackend

    class Exploding(FakeBackend):
        def separate_music(self, *a, **k):
            raise RuntimeError(r"boom at C:\Users\someone\secret\path\model.ckpt")

    with TestClient(create_app(settings, backend=Exploding(settings))) as c:
        job = wait_for_job(c, upload(c, audio_dir / "song.wav").json()["job_id"])
        assert job["status"] == "error"
        assert "secret" not in job["error"] and "Users" not in job["error"] and "boom" not in job["error"]
        assert "error " in job["error"]  # the reference id to look up in the log


def test_scrub_removes_paths():
    s = scrub(r"C:\Users\name\Music\song.wav: Invalid data; /home/name/x.wav; \\server\share\a")
    assert "name" not in s and "Music" not in s
    assert s.count("<path>") == 3


# ----------------------------------------------------- model integrity


def test_tampered_model_file_is_refused(tmp_path, monkeypatch):
    from app import integrity

    f = tmp_path / "model.onnx"
    f.write_bytes(b"genuine weights")
    good = integrity.sha256(f)
    monkeypatch.setitem(integrity.PINNED, "model.onnx", good)
    verify(f, "model.onnx", tmp_path)  # passes and is cached
    f.write_bytes(b"tampered weights")
    with pytest.raises(IntegrityError):
        verify(f, "model.onnx", tmp_path)


def test_uploaded_filename_is_reduced_to_a_display_name(client, audio_dir):
    data = (audio_dir / "song.wav").read_bytes()
    for raw, expected in [
        (r"C:\Users\someone\Music\my song.wav", "my song.wav"),
        ("../../etc/passwd.wav", "passwd.wav"),
        ("evil\r\nX-Injected: 1.wav", None),  # the client may percent-encode; either way no CR/LF survives
    ]:
        r = client.post("/api/jobs", files={"file": (raw, data, "audio/wav")})
        assert r.status_code == 201, r.text
        job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
        if expected is not None:
            assert job["filename"] == expected
        assert not any(ord(ch) < 32 for ch in job["filename"])
        client.delete(f"/api/jobs/{job['id']}")


def test_safe_filename_strips_control_characters():
    from app.main import _safe_filename

    assert _safe_filename("a\r\nb\x00c.wav") == "abc.wav"
    assert _safe_filename("") == "upload"
    assert _safe_filename(None) == "upload"
    long = _safe_filename("x" * 300 + ".wav")
    assert len(long) <= 120 and long.endswith(".wav")


def test_model_route_serves_only_the_allowlisted_verified_model(settings, monkeypatch):
    from app import integrity

    models = settings.models_dir
    models.mkdir(parents=True, exist_ok=True)
    (models / "Kim_Vocal_2.onnx").write_bytes(b"onnx bytes")
    (models / "model_bs_roformer_ep_317_sdr_12.9755.ckpt").write_bytes(b"other")
    monkeypatch.setitem(integrity.PINNED, "Kim_Vocal_2.onnx", integrity.sha256(models / "Kim_Vocal_2.onnx"))
    with TestClient(create_app(settings)) as c:
        r = c.get("/api/models/Kim_Vocal_2.onnx")
        assert r.status_code == 200 and r.content == b"onnx bytes"
        assert "immutable" in r.headers["cache-control"]
        for bad in ("model_bs_roformer_ep_317_sdr_12.9755.ckpt", "..%2Funmix.sqlite3", ".verified.json"):
            assert c.get(f"/api/models/{bad}").status_code == 404, bad
        (models / "Kim_Vocal_2.onnx").write_bytes(b"tampered")  # swapped after pinning
        r = c.get("/api/models/Kim_Vocal_2.onnx")
        assert r.status_code == 500 and "integrity" in r.json()["detail"]
