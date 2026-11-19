"""Remote GPU mode: the backend runs on Colab behind a public tunnel URL, so
every request must carry the access key; stems travel as lossless FLAC."""

from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import create_app

from .conftest import upload, wait_for_job

KEY = "k3y-for-tests-only-0123456789abcdef"
TUNNEL = "quiet-river-1234.trycloudflare.com"


@pytest.fixture()
def remote(settings):
    s = replace(settings, remote_key=KEY, stem_transport="flac", allowed_hosts=(*settings.allowed_hosts, TUNNEL))
    with TestClient(create_app(s), headers={"X-Unmix-Key": KEY}) as c:
        yield c


def test_requests_without_the_key_are_refused(remote, audio_dir):
    assert remote.get("/api/health").status_code == 200
    assert remote.get("/api/health").json()["remote"] is True
    for headers in ({"X-Unmix-Key": ""}, {"X-Unmix-Key": "wrong"}, {"X-Unmix-Key": KEY[:-1]}):
        r = remote.get("/api/health", headers=headers)
        assert r.status_code == 401, headers
        assert KEY not in r.text
    with (audio_dir / "song.wav").open("rb") as fh:
        r = remote.post("/api/jobs", files={"file": ("a.wav", fh, "audio/wav")}, headers={"X-Unmix-Key": "nope"})
    assert r.status_code == 401


def test_tunnel_host_is_allowed_but_other_protections_stay(remote):
    assert remote.get("/api/health", headers={"Host": TUNNEL}).status_code == 200
    assert remote.get("/api/health", headers={"Host": "other.trycloudflare.com"}).status_code == 421
    # The local proxy forwards the browser's Origin: other sites are still refused.
    r = remote.delete("/api/jobs/" + "0" * 32, headers={"Host": TUNNEL, "Origin": "https://evil.example"})
    assert r.status_code == 403


def test_websocket_needs_the_key(remote):
    with pytest.raises(WebSocketDisconnect):
        with remote.websocket_connect("/api/live", headers={"Origin": "http://localhost:5173", "X-Unmix-Key": "wrong"}) as ws:
            ws.receive_text()


def test_short_keys_are_rejected(settings):
    with pytest.raises(ValueError, match="at least"):
        create_app(replace(settings, remote_key="short"))


def test_stems_are_served_as_lossless_flac(remote, audio_dir):
    job = wait_for_job(remote, upload(remote, audio_dir / "song.wav").json()["job_id"])
    assert job["status"] == "done", job
    url = job["stems"][0]["url"]
    r = remote.get(url)
    assert r.headers["content-type"] == "audio/flac"
    flac, sr = sf.read(io.BytesIO(r.content), dtype="int16")
    wav_bytes = remote.get(url + "?download=true")
    assert wav_bytes.headers["content-type"] == "audio/wav" and "attachment" in wav_bytes.headers["content-disposition"]
    wav, sr2 = sf.read(io.BytesIO(wav_bytes.content), dtype="int16")
    assert sr == sr2 and np.array_equal(flac, wav), "FLAC must be bit-identical to the WAV"
    assert len(r.content) < len(wav_bytes.content)
    # Range requests still work on the FLAC.
    part = remote.get(url, headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100
    # The mix (float WAV) streams as 24-bit FLAC.
    assert remote.get(f"/api/jobs/{job['id']}/stems/mix.wav").headers["content-type"] == "audio/flac"
