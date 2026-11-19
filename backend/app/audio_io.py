"""Decoding (ffmpeg), loudness normalisation and WAV I/O helpers.

Arrays are always float32 with shape (samples, channels) -- the soundfile layout.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioError(Exception):
    """A problem with the user's audio that should be reported as a 4xx."""


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg was not found on PATH. Install it (winget install Gyan.FFmpeg / brew install ffmpeg).")
    return exe


def decode_to_wav(src: Path, dst: Path, *, sample_rate: int, max_seconds: float) -> None:
    """Decode an uploaded audio file to stereo float32 WAV.

    Hardening (the file is untrusted):
    * the container is identified from its magic bytes and that demuxer is
      forced with -f, so ffmpeg never auto-probes into playlist/concat/image
      demuxers that can reference other local files or URLs;
    * -protocol_whitelist file: no network or pipe protocols, whatever the file says;
    * decoding stops slightly past ``max_seconds`` so a huge file cannot tie
      up the server; the caller then rejects anything longer than the limit.
    """
    from .security import sniff_audio_format

    demuxer = sniff_audio_format(src)
    if demuxer is None:
        raise AudioError("This does not look like a supported audio file (wav, mp3, m4a, webm, ogg, flac, aac or aiff).")
    cmd = [
        ffmpeg_path(), "-v", "error", "-nostdin", "-y",
        "-protocol_whitelist", "file",
        "-f", demuxer,
        "-i", str(src),
        "-map", "0:a:0", "-vn",
        "-ac", "2", "-ar", str(sample_rate),
        "-c:a", "pcm_f32le",
        "-t", f"{max_seconds + 1.0:.3f}",
        str(dst),
    ]  # fmt: skip
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as exc:
        raise AudioError("Decoding the file took too long.") from exc
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size <= 44:
        from .security import scrub

        detail = (proc.stderr or "").strip().splitlines()
        reason = scrub(detail[-1])[:160] if detail else "unknown decoder error"
        raise AudioError(f"Could not decode this file as audio ({reason}).")


def wav_duration(path: Path) -> float:
    try:
        info = sf.info(str(path))
    except Exception as exc:  # noqa: BLE001
        raise AudioError("Could not decode this file as audio.") from exc
    return info.frames / float(info.samplerate)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data, sr


def write_wav(path: Path, data: np.ndarray, sample_rate: int, *, subtype: str = "PCM_16") -> None:
    """Write atomically so a half-written stem is never served."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.wav")
    if subtype.startswith("PCM"):
        data = np.clip(data, -1.0, 1.0)
    sf.write(str(tmp), data, sample_rate, subtype=subtype)
    tmp.replace(path)


def flac_copy(wav: Path) -> Path:
    """A FLAC twin of `wav` next to it (made once, atomically), for serving
    stems over slow links: lossless for the 16-bit stems and about half the
    size. Float WAVs (the mix) become 24-bit FLAC, which is beyond audible."""
    out = wav.with_suffix(".flac")
    if out.is_file() and out.stat().st_mtime >= wav.stat().st_mtime:
        return out
    info = sf.info(str(wav))
    subtype = "PCM_16" if info.subtype == "PCM_16" else "PCM_24"
    data, sr = sf.read(str(wav), dtype="int16" if subtype == "PCM_16" else "float32", always_2d=True)
    tmp = out.with_name(f"{out.stem}.{uuid.uuid4().hex[:8]}.tmp.flac")
    try:
        sf.write(str(tmp), data if subtype == "PCM_16" else np.clip(data, -1.0, 1.0), sr, format="FLAC", subtype=subtype)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def loudness_normalize(data: np.ndarray, sample_rate: int, *, target_lufs: float, peak_ceiling: float) -> tuple[np.ndarray, dict]:
    """Linear-gain loudness normalisation (ITU-R BS.1770) with a peak ceiling.

    A single static gain is used on purpose: dynamic processing (compressors,
    ffmpeg's one-pass ``loudnorm``) changes the mix in ways the separation
    models were never trained on.
    """
    import pyloudnorm as pyln

    info: dict = {"input_lufs": None, "gain_db": 0.0, "peak_limited": False}
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 1e-6:
        raise AudioError("The recording is silent.")

    gain = 1.0
    try:
        loudness = float(pyln.Meter(sample_rate).integrated_loudness(data.astype(np.float64)))
    except ValueError:  # shorter than one gating block
        loudness = float("-inf")
    if np.isfinite(loudness):
        info["input_lufs"] = round(loudness, 2)
        gain = float(10.0 ** ((target_lufs - loudness) / 20.0))
    else:
        gain = peak_ceiling / peak  # too quiet/short to measure: just peak-normalise

    if peak * gain > peak_ceiling:
        gain = peak_ceiling / peak
        info["peak_limited"] = True

    info["gain_db"] = round(20.0 * np.log10(gain), 2)
    return (data * np.float32(gain)).astype(np.float32, copy=False), info


def match_length(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Trim or zero-pad along time so every stem has exactly ``n_samples``."""
    if data.shape[0] == n_samples:
        return data
    if data.shape[0] > n_samples:
        return data[:n_samples]
    pad = np.zeros((n_samples - data.shape[0], data.shape[1]), dtype=data.dtype)
    return np.concatenate([data, pad], axis=0)
