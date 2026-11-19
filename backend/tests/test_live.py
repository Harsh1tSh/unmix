"""Phase 4 live mode: the WebSocket protocol, exact reconstruction, security."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import live as L
from app.main import create_app

from .conftest import make_tone_mix

ORIGIN = {"origin": "http://localhost:5173"}


@pytest.fixture()
def client(client):
    """Live sessions need the models loaded (like a real client watching /api/health)."""
    deadline = time.time() + 10
    while not client.app.state.manager.models_ready and time.time() < deadline:
        time.sleep(0.02)
    return client


def run_session(ws, audio: np.ndarray, channels: int = 2, rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Stream `audio` (n, channels) in irregular frames; collect the output."""
    ws.send_json({"type": "start", "sample_rate": L.SR, "channels": channels})
    assert ws.receive_json()["type"] == "ready"
    rng = np.random.default_rng(rng_seed)
    pos = 0
    while pos < len(audio):
        n = int(rng.integers(1000, 40000))
        ws.send_bytes(np.ascontiguousarray(audio[pos : pos + n], dtype="<f4").tobytes())
        pos += n
    ws.send_json({"type": "stop"})
    voices, music, headers = [], [], []
    while True:
        msg = ws.receive()
        if msg.get("text") is not None:
            data = json.loads(msg["text"])
            headers.append(data)
            if data["type"] == "end":
                break
            if data["type"] == "block":
                payload = np.frombuffer(ws.receive_bytes(), dtype="<f4").reshape(4, data["frames"])
                voices.append(payload[:2])
                music.append(payload[2:])
        elif msg["type"] == "websocket.close":
            break
    return np.concatenate(voices, axis=1), np.concatenate(music, axis=1), headers


def test_live_session_reconstructs_the_input_exactly(client):
    audio = make_tone_mix(12.0)  # (n, 2)
    with client.websocket_connect("/api/live", headers=ORIGIN) as ws:
        voices, music, headers = run_session(ws, audio)
    blocks = [h for h in headers if h["type"] == "block"]
    assert len(blocks) == int(np.ceil(len(audio) / L.GEN))
    assert [b["start"] for b in blocks] == [i * L.GEN for i in range(len(blocks))]
    assert headers[-1] == {"type": "end", "frames": len(audio)}
    assert voices.shape == (2, len(audio)) == music.shape
    # Same invariant as the batch pipeline: voices + music == what was sent.
    assert np.max(np.abs(voices + music - audio.T)) < 1e-5
    assert np.mean(voices**2) > 1e-4 and np.mean(music**2) > 1e-4


def test_mono_microphones_are_accepted(client):
    audio = make_tone_mix(3.0)[:, :1]
    with client.websocket_connect("/api/live", headers=ORIGIN) as ws:
        voices, music, _ = run_session(ws, audio, channels=1)
    assert voices.shape == (2, len(audio))
    assert np.max(np.abs(voices + music - np.vstack([audio.T, audio.T]))) < 1e-5


@pytest.mark.parametrize(
    "headers",
    [
        {},  # no Origin: not a browser (a page could still not read it, but we require one)
        {"origin": "https://evil.example"},  # cross-site WebSocket hijacking
        {"origin": "null"},
        {"origin": "http://localhost:5173", "host": "attacker.example"},  # DNS rebinding
    ],
)
def test_cross_site_websockets_are_refused(client, headers):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/api/live", headers=headers) as ws:
            ws.receive_json()
    assert exc.value.code == 1008


def test_bad_start_and_malformed_frames(client):
    with client.websocket_connect("/api/live", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "sample_rate": 48000, "channels": 2})
        assert "sample_rate" in ws.receive_json()["message"]
    with client.websocket_connect("/api/live", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "sample_rate": L.SR, "channels": 2})
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(b"\x00\x00\x00")  # not a whole frame
        assert ws.receive_json() == {"type": "error", "message": "Malformed audio frame."}
    with client.websocket_connect("/api/live", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "sample_rate": L.SR, "channels": 2})
        ws.receive_json()
        ws.send_bytes(np.full(8, np.nan, dtype="<f4").tobytes())
        assert ws.receive_json()["type"] == "error"
    with client.websocket_connect("/api/live", headers=ORIGIN) as ws:
        ws.send_json({"type": "start", "sample_rate": L.SR, "channels": 2})
        ws.receive_json()
        ws.send_bytes(b"\x00" * (L.MAX_MESSAGE_BYTES + 8))
        assert ws.receive_json()["message"] == "Audio frame too large."


def test_only_one_live_session_and_the_queue_waits(client):
    manager = client.app.state.manager
    with client.websocket_connect("/api/live", headers=ORIGIN) as first:
        first.send_json({"type": "start", "sample_rate": L.SR, "channels": 2})
        assert first.receive_json()["type"] == "ready"
        assert manager.live_active.is_set(), "batch jobs pause while live runs"
        with client.websocket_connect("/api/live", headers=ORIGIN) as second:
            assert "Another live session" in second.receive_json()["message"]
        first.send_json({"type": "stop"})
        while first.receive_json()["type"] != "end":
            pass
    deadline = time.time() + 5
    while manager.live_active.is_set() and time.time() < deadline:
        time.sleep(0.05)
    assert not manager.live_active.is_set()


def test_streaming_splitter_handles_irregular_input_and_skip_ahead():
    s = L.StreamingSplitter(lambda w: w * 0.5)
    x = np.random.default_rng(3).standard_normal((2, L.GEN * 5)).astype(np.float32)
    for i in range(0, x.shape[1], 12345):
        s.push(x[:, i : i + 12345])
    assert s.ready_blocks() >= 4
    skipped = s.skip_ahead(2 * L.GEN)
    assert skipped == 2 * L.GEN
    outs = []
    while (o := s.next_block(final=True)) is not None:
        outs.append(o)
    assert outs[0][0] == 2 * L.GEN, "output resumes after the skipped audio"
    assert sum(o[1].shape[1] for o in outs) == x.shape[1] - skipped
    first_start, v, m = outs[0]
    np.testing.assert_allclose(v + m, x[:, first_start : first_start + v.shape[1]], atol=1e-6)
