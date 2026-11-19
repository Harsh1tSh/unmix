"""Phase 4: near-real-time voices/music separation over a WebSocket.

Model: Kim Vocal 2 (MDX-Net, ONNX), the same model fast mode uses on CPU, on the
GPU app/accel.py picked. Measured on a 4-core laptop CPU: one pass covers 5.75 s
of new audio and takes ~5.1 s, i.e. 0.89x real time, so ~11 s of delay (one block
to fill, one to compute) and little headroom. On the same laptop's GeForce MX150
through DirectML a pass takes ~1.3 s (0.27x) and the delay is ~7.5 s.
Instruments (Demucs, 1.4x real time on CPU) are not separated live.

Streaming scheme (the same chunking MDX uses offline, without overlap):

    window = n_fft/2 samples of context | 253 440 new samples | n_fft/2 of context
    the model sees the whole 261 120-sample window, and only the middle part is
    kept, so block edges get full context and join seamlessly.

Privacy: nothing from a live session is written to disk.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings

log = logging.getLogger("unmix.live")

SR = 44100
N_FFT = 7680
HOP = 1024
DIM_F = 3072
DIM_T = 256
CHUNK = HOP * (DIM_T - 1)  # 261 120 samples the model sees
TRIM = N_FFT // 2  # 3 840 samples of context on each side
GEN = CHUNK - 2 * TRIM  # 253 440 new samples per pass (5.75 s)
MAX_MESSAGE_BYTES = 1 << 20  # one binary frame from the browser (~3 s of stereo float32)
MAX_SESSION_S = 10 * 60
MAX_BACKLOG_BLOCKS = 3  # beyond this the session is falling hopelessly behind


class BlockSeparator(Protocol):
    """Maps a (2, CHUNK) float32 window to the voices estimate for it."""

    def __call__(self, window: np.ndarray) -> np.ndarray: ...


class KimVocalLive:
    """Kim Vocal 2 run directly with onnxruntime (no audio-separator overhead).

    `runner` is the process-wide session (app/accel.OnnxRunner) that fast mode
    uses too: it is on the GPU chosen by app/accel.py, serialises calls and
    falls back to the CPU by itself."""

    def __init__(self, runner):
        import torch

        self._torch = torch
        self.runner = runner
        self.window = torch.hann_window(N_FFT, periodic=True)

    def _run(self, model_in: np.ndarray) -> np.ndarray:
        return self.runner(model_in)[0]

    def __call__(self, window: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))  # (2, CHUNK)
        spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, window=self.window, center=True, return_complex=True)  # (2, F, T)
        spec = spec[:, :DIM_F, :]
        model_in = torch.stack([spec[0].real, spec[0].imag, spec[1].real, spec[1].imag])[None]  # (1, 4, F, T)
        model_in[:, :, :3, :] = 0  # as MDX does: drop the lowest bins
        out = self._run(model_in.numpy())  # (4, F, T)
        full = np.zeros((4, N_FFT // 2 + 1, out.shape[-1]), dtype=np.float32)
        full[:, :DIM_F] = out
        cplx = torch.complex(torch.from_numpy(full[[0, 2]]), torch.from_numpy(full[[1, 3]]))  # (2, F, T)
        voices = torch.istft(cplx, n_fft=N_FFT, hop_length=HOP, window=self.window, center=True, length=CHUNK)
        return voices.numpy()


class StreamingSplitter:
    """Feed arbitrary-sized stereo blocks in, get separated blocks out."""

    def __init__(self, separate: BlockSeparator):
        self.separate = separate
        self.buf = np.zeros((2, TRIM), dtype=np.float32)  # leading context = silence
        self.emitted = 0  # samples of output produced so far
        self.received = 0
        # push() runs on the event loop while next_block() runs in a worker
        # thread: every change to buf/emitted/received happens under this lock
        # (the model call itself does not).
        self._lock = threading.Lock()

    def push(self, stereo: np.ndarray) -> None:
        """stereo: (2, n) float32."""
        with self._lock:
            self.buf = np.concatenate([self.buf, stereo.astype(np.float32, copy=False)], axis=1)
            self.received += stereo.shape[1]

    def ready_blocks(self) -> int:
        with self._lock:
            n = self.buf.shape[1]
        return max(0, (n - (CHUNK - GEN)) // GEN) if n >= CHUNK else 0

    def skip_ahead(self, frames: int) -> int:
        """Drop `frames` of unprocessed input (when hopelessly behind)."""
        with self._lock:
            frames = max(0, min(frames, self.buf.shape[1] - CHUNK))
            self.buf = self.buf[:, frames:]
            self.emitted += frames
            return frames

    def next_block(self, final: bool = False) -> tuple[int, np.ndarray, np.ndarray] | None:
        """Process one block if enough audio is buffered (or pad, if final).
        Returns (start_sample, voices (2,n), music (2,n))."""
        with self._lock:
            if self.buf.shape[1] < CHUNK:
                if not final or self.emitted >= self.received:
                    return None
                self.buf = np.pad(self.buf, ((0, 0), (0, CHUNK - self.buf.shape[1])))
            window = self.buf[:, :CHUNK].copy()
            n = min(GEN, self.received - self.emitted)
            start = self.emitted
        voices = self.separate(window)[:, TRIM : TRIM + GEN]  # the slow part, outside the lock
        mix = window[:, TRIM : TRIM + GEN]
        with self._lock:
            self.emitted += n
            self.buf = self.buf[:, GEN:]
        voices = voices[:, :n].astype(np.float32)
        return start, voices, (mix[:, :n] - voices).astype(np.float32)


def kim_model_path(settings: Settings) -> Path:
    return settings.models_dir / "Kim_Vocal_2.onnx"


def encode_block(index: int, start: int, voices: np.ndarray, music: np.ndarray, compute_s: float, backlog: int) -> tuple[str, bytes]:
    """A JSON header message followed by a binary payload: float32 planar
    [voices L, voices R, music L, music R], each `frames` long."""
    header = {"type": "block", "index": index, "start": start, "frames": int(voices.shape[1]), "compute_s": round(compute_s, 3), "backlog": backlog}
    payload = np.concatenate([voices[0], voices[1], music[0], music[1]]).astype("<f4").tobytes()
    return json.dumps(header), payload


def decode_frames(data: bytes, channels: int) -> np.ndarray:
    """Browser frames: interleaved little-endian float32 -> (2, n)."""
    if len(data) % (4 * channels):
        raise ValueError("partial frame")
    x = np.frombuffer(data, dtype="<f4").reshape(-1, channels).T
    if not np.isfinite(x).all():
        raise ValueError("non-finite samples")
    x = np.clip(x, -1.0, 1.0)
    return np.vstack([x, x]) if channels == 1 else x[:2]


def now() -> float:
    return time.perf_counter()
