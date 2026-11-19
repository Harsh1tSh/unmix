"""Stage 3: split the voices stem into two speakers (e.g. you vs. the singer).

The model, MossFormer2_SS_16K from ClearerVoice-Studio, lives in its own
virtualenv (backend/speakers/) because its pins conflict with audio-separator.
This module runs backend/speakers/run_split.py as a subprocess and then turns
its 16 kHz mono estimates into full-band 44.1 kHz stereo stems.

Full band via masking: upsampling the 16 kHz outputs would lose everything
above 8 kHz (sibilance, breath, air). Instead the two estimates define a soft
time-frequency mask that is applied to the real 44.1 kHz voices stem:

    M1[f,t] = |E1|^2 / (|E1|^2 + |E2|^2)        for f below ~7.5 kHz
    M1[f,t] = per-frame energy ratio of 2-7.5 kHz  above it (no estimate there)
    speaker_1 = iSTFT(M1 * V),   speaker_2 = voices - speaker_1

which also keeps speaker_1 + speaker_2 == voices exactly, as the player needs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from . import audio_io
from .config import BACKEND_DIR, Settings

SPEAKERS_DIR = BACKEND_DIR / "speakers"
MODEL_SR = 16000
N_FFT = 2048
HOP = 512
VALID_HZ = 7500.0  # top of the band the 16 kHz model can say anything about
REF_BAND_HZ = (2000.0, 7500.0)  # band whose ratio is extended to the top octave


class SpeakerSplitUnavailable(RuntimeError):
    pass


def speakers_python(settings: Settings) -> Path:
    if settings.speakers_python:
        return Path(settings.speakers_python)
    exe = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return SPEAKERS_DIR / ".venv" / exe


def speaker_split_available(settings: Settings) -> bool:
    return speakers_python(settings).is_file() and (SPEAKERS_DIR / "run_split.py").is_file()


def _run_runner(script: str, args: list[str], settings: Settings, progress: Callable[[float], None], what: str, gpu: bool = False) -> None:
    """Run one of the scripts in backend/speakers/ with that venv's python,
    relaying its "PROGRESS x" lines. Arguments only, never a shell.

    ClearerVoice uses CUDA whenever its torch sees a GPU, so the GPU is hidden
    from it unless the main backend decided the GPU is usable (app/accel.py).
    A run that fails on the GPU is retried once on the CPU."""
    try:
        _run_runner_once(script, args, settings, progress, what, gpu)
    except RuntimeError as exc:
        from .accel import looks_like_gpu_failure

        if not gpu or not looks_like_gpu_failure(exc):
            raise
        import logging

        logging.getLogger("unmix.speakers").warning("%s failed on the GPU; retrying on the CPU: %s", what, exc)
        _run_runner_once(script, args, settings, progress, what, False)


def _run_runner_once(script: str, args: list[str], settings: Settings, progress: Callable[[float], None], what: str, gpu: bool) -> None:
    py = speakers_python(settings)
    if not py.is_file():
        raise SpeakerSplitUnavailable(
            "The voice models are not installed. Run `uv sync` in backend/speakers (see README, 'Voice splitting')."
        )
    cmd = [str(py), str(SPEAKERS_DIR / script), *args]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env, cwd=str(SPEAKERS_DIR))
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("PROGRESS "):
            try:
                progress(float(line.split()[1]))
            except ValueError:
                pass
        elif line:
            tail = (tail + [line])[-15:]
    code = proc.wait()
    if code != 0:
        # Full output goes to the server log; what the user sees is generic (see worker).
        raise RuntimeError(f"{what} exited with code {code}: {' | '.join(tail[-8:])}")


def run_model(voices_path: Path, work_dir: Path, settings: Settings, progress: Callable[[float], None], gpu: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Run run_split.py; returns the two 16 kHz mono estimates."""
    from .integrity import CLEARVOICE_SS_REVISION, PINNED

    out_dir = work_dir / "speakers"
    ckpt = settings.models_dir / "clearvoice"
    expected = PINNED["clearvoice/MossFormer2_SS_16K/last_best_checkpoint.pt"]
    _run_runner("run_split.py", [str(voices_path), str(out_dir), str(ckpt), CLEARVOICE_SS_REVISION, expected], settings, progress, "speaker separation", gpu)
    e1, _ = audio_io.read_wav(out_dir / "spk1_16k.wav")
    e2, _ = audio_io.read_wav(out_dir / "spk2_16k.wav")
    return e1[:, 0], e2[:, 0]


def enhance_voice(src: Path, dst: Path, settings: Settings, progress: Callable[[float], None], gpu: bool = False) -> None:
    """Voice clean-up (MossFormer2_SE_48K): writes a 48 kHz mono 16-bit WAV.

    The result is an extra download, not a stem: enhancement changes the audio,
    so it would break "stems sum to their parent"."""
    from .integrity import CLEARVOICE_SE_REVISION, PINNED

    tmp = dst.with_name(dst.stem + ".f32.wav")
    expected = PINNED["clearvoice/MossFormer2_SE_48K/last_best_checkpoint.pt"]
    ckpt = settings.models_dir / "clearvoice"
    _run_runner("run_enhance.py", [str(src), str(tmp), str(ckpt), CLEARVOICE_SE_REVISION, expected], settings, progress, "voice clean-up", gpu)
    data, sr = audio_io.read_wav(tmp)
    audio_io.write_wav(dst, data, sr)
    tmp.unlink(missing_ok=True)


def full_band_split(voices: np.ndarray, sample_rate: int, e1: np.ndarray, e2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the 16 kHz estimates as a soft mask to the full-band voices.

    voices: (n, channels) float32. Returns (speaker_1, speaker_2), same shape,
    with speaker_1 + speaker_2 == voices. speaker_1 is the louder speaker.
    """
    import librosa
    from scipy.signal import resample_poly

    n = voices.shape[0]
    g = int(np.gcd(sample_rate, MODEL_SR))

    def up(x: np.ndarray) -> np.ndarray:
        y = resample_poly(x, sample_rate // g, MODEL_SR // g).astype(np.float32)
        return np.pad(y, (0, max(0, n - y.shape[0])))[:n]

    p1 = np.abs(librosa.stft(up(e1), n_fft=N_FFT, hop_length=HOP)) ** 2
    p2 = np.abs(librosa.stft(up(e2), n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sample_rate)
    eps = 1e-10

    mask = p1 / (p1 + p2 + eps)
    total = p1 + p2
    # Cells where neither estimate has energy: split evenly rather than guess.
    mask[total < total.max() * 1e-8] = 0.5
    ref = (freqs >= REF_BAND_HZ[0]) & (freqs < REF_BAND_HZ[1])
    frame_ratio = p1[ref].sum(axis=0) / (p1[ref].sum(axis=0) + p2[ref].sum(axis=0) + eps)
    mask[freqs >= VALID_HZ] = frame_ratio[None, :]
    mask = mask.astype(np.float32)

    s1 = np.empty_like(voices)
    for ch in range(voices.shape[1]):
        spec = librosa.stft(np.ascontiguousarray(voices[:, ch]), n_fft=N_FFT, hop_length=HOP)
        frames = min(spec.shape[1], mask.shape[1])
        spec[:, :frames] *= mask[:, :frames]
        s1[:, ch] = librosa.istft(spec, hop_length=HOP, n_fft=N_FFT, length=n)
    s2 = voices - s1
    if float(np.sum(s2.astype(np.float64) ** 2)) > float(np.sum(s1.astype(np.float64) ** 2)):
        s1, s2 = s2, s1
    return s1, s2


def separate_speakers(voices_path: Path, voices: np.ndarray, work_dir: Path, settings: Settings, progress: Callable[[float], None], gpu: bool = False) -> tuple[np.ndarray, np.ndarray]:
    e1, e2 = run_model(voices_path, work_dir, settings, lambda p: progress(0.9 * p), gpu)
    progress(0.92)
    s1, s2 = full_band_split(voices, settings.sample_rate, e1, e2)
    progress(1.0)
    return s1, s2


if __name__ == "__main__":  # manual check: python -m app.speakers <voices.wav> <work_dir>
    from .config import settings as _s

    v, sr = audio_io.read_wav(Path(sys.argv[1]))
    a, b = separate_speakers(Path(sys.argv[1]), v, Path(sys.argv[2]), _s, lambda p: print(f"{p:.2f}", end=" ", flush=True))
    audio_io.write_wav(Path(sys.argv[2]) / "speaker_1.wav", a, sr)
    audio_io.write_wav(Path(sys.argv[2]) / "speaker_2.wav", b, sr)
    print("\nwritten")
