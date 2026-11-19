"""Stage 4: per-stem analysis (what the stem cards draw).

Separation happens by *source* (neural masks). This module then describes the
frequency content of each separated stem: 7-band energy over time, a waveform
overview, an RMS envelope and a mel spectrogram image.

All level-type values are normalised against the loudest thing in the whole
job, so a quiet stem looks quiet next to a loud one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import BAND_EDGES_HZ, BAND_NAMES, Settings

DB_FLOOR = -60.0  # band meters span [-60 dB, 0 dB] relative to the job maximum
SPEC_DB_FLOOR = -80.0


@dataclass
class StemFeatures:
    peaks: np.ndarray  # (peaks_points,) max |x| per bucket, 0..1
    rms: np.ndarray  # (frames,) linear RMS amplitude
    band_energy: np.ndarray  # (frames, 7) mean power per band (linear)
    mel_power: np.ndarray  # (mels, frames) linear
    energy: float  # sum of x^2 over the whole stem


def _power_spectrogram(mono_or_stereo: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Channel-averaged power spectrogram, shape (1 + n_fft/2, frames)."""
    import librosa

    power = None
    for ch in range(mono_or_stereo.shape[1]):
        spec = librosa.stft(np.ascontiguousarray(mono_or_stereo[:, ch]), n_fft=n_fft, hop_length=hop, window="hann", center=True)
        p = (spec.real**2 + spec.imag**2).astype(np.float32)
        power = p if power is None else power + p
        del spec
    assert power is not None
    return power / mono_or_stereo.shape[1]


def compute_peaks(data: np.ndarray, points: int) -> np.ndarray:
    mono = np.max(np.abs(data), axis=1)
    n = mono.shape[0]
    if n == 0:
        return np.zeros(points, dtype=np.float32)
    edges = np.linspace(0, n, points + 1).astype(np.int64)
    starts = np.minimum(edges[:-1], n - 1)
    # reduceat gives max over [starts[i], starts[i+1]) -- one bucket per point.
    return np.maximum.reduceat(mono, starts).astype(np.float32)


def extract_features(data: np.ndarray, settings: Settings) -> StemFeatures:
    import librosa

    sr = settings.sample_rate
    n_fft = settings.analysis_n_fft
    hop = sr // settings.analysis_fps

    power = _power_spectrogram(data, n_fft, hop)  # (bins, frames)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    bands = np.empty((power.shape[1], len(BAND_NAMES)), dtype=np.float32)
    for b, (lo, hi) in enumerate(zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])):
        sel = (freqs >= lo) & (freqs < hi)
        # Sum (not mean) so a band's value is the energy that falls inside it.
        bands[:, b] = power[sel].sum(axis=0)

    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=settings.spectrogram_mels, fmin=20.0, fmax=min(16000.0, sr / 2))
    mel_power = (mel_fb @ power).astype(np.float32)

    frame_rms = librosa.feature.rms(y=data.mean(axis=1), frame_length=n_fft, hop_length=hop, center=True)[0]

    return StemFeatures(
        peaks=compute_peaks(data, settings.peaks_points),
        rms=frame_rms.astype(np.float32),
        band_energy=bands,
        mel_power=mel_power,
        energy=float(np.sum(data.astype(np.float64) ** 2)),
    )


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def render_spectrogram(mel_power: np.ndarray, ref_power: float, color: str, path: Path, max_width: int) -> None:
    """Write a mel spectrogram PNG tinted with the stem colour.

    Intensity ramp: near-black -> stem colour -> white, so the image sits on the
    dark UI without a border and the loudest partials still pop.
    """
    from PIL import Image

    db = 10.0 * np.log10(np.maximum(mel_power, 1e-20) / max(ref_power, 1e-20))
    norm = np.clip((db - SPEC_DB_FLOOR) / -SPEC_DB_FLOOR, 0.0, 1.0)
    norm = norm[::-1]  # low frequencies at the bottom

    r, g, b = _hex_to_rgb(color)
    ramp_in = np.array([0.0, 0.7, 1.0])
    lut = np.stack(
        [np.interp(np.linspace(0, 1, 256), ramp_in, [8, c, 255]) for c in (r, g, b)],
        axis=1,
    ).astype(np.uint8)
    img = Image.fromarray(lut[(norm * 255).astype(np.uint8)], mode="RGB")
    if img.width > max_width:
        img = img.resize((max_width, img.height), Image.Resampling.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", optimize=True)


def _round_list(arr: np.ndarray, digits: int = 3) -> list:
    return np.round(arr.astype(np.float64), digits).tolist()


def build_analysis(
    features: dict[str, StemFeatures],
    *,
    mix_energy: float,
    duration_s: float,
    settings: Settings,
    colors: dict[str, str],
    spectrogram_dir: Path,
    spectrogram_url: dict[str, str],
) -> dict:
    """Normalise features across the job and assemble the /analysis payload."""
    band_ref = max((float(f.band_energy.max()) for f in features.values()), default=0.0) or 1e-20
    mel_ref = max((float(f.mel_power.max()) for f in features.values()), default=0.0) or 1e-20

    stems: dict[str, dict] = {}
    for name, f in features.items():
        band_db = 10.0 * np.log10(np.maximum(f.band_energy, 1e-20) / band_ref)
        band_norm = np.clip((band_db - DB_FLOOR) / -DB_FLOOR, 0.0, 1.0)
        render_spectrogram(f.mel_power, mel_ref, colors[name], spectrogram_dir / f"{name}.png", settings.spectrogram_max_width)
        stems[name] = {
            "peaks": _round_list(f.peaks),
            "rms": _round_list(f.rms, 4),
            "bands": {
                "names": BAND_NAMES,
                "hz_edges": BAND_EDGES_HZ,
                "frames": _round_list(band_norm),
            },
            "energy_share": round(f.energy / mix_energy, 4) if mix_energy > 0 else 0.0,
            "spectrogram_url": spectrogram_url[name],
        }

    return {
        "duration_s": round(duration_s, 3),
        "sample_rate": settings.sample_rate,
        "frame_rate": settings.analysis_fps,
        "stems": stems,
    }
