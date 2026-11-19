"""Sanity-check a finished job on disk: stem alignment, the sum-to-mix invariant
and a summary of the analysis.

Usage:  uv run python scripts/check_job.py <job_id>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import STEMS, settings  # noqa: E402


def db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-12))


def main(job_id: str) -> int:
    job_dir = settings.jobs_dir / job_id
    mix, sr = sf.read(job_dir / "mix.wav", dtype="float32", always_2d=True)
    stems = {p.stem: sf.read(p, dtype="float32", always_2d=True)[0] for p in sorted((job_dir / "stems").glob("*.wav"))}
    print(f"mix: {mix.shape[0] / sr:.2f}s @ {sr} Hz, peak {db(np.abs(mix).max()):.1f} dBFS")

    ok = True
    for name, data in stems.items():
        aligned = data.shape == mix.shape
        ok &= aligned
        print(f"  {name:8s} peak {db(np.abs(data).max()):6.1f} dBFS  rms {db(np.sqrt(np.mean(data**2))):6.1f} dB  {'aligned' if aligned else 'LENGTH MISMATCH ' + str(data.shape)}")

    err_root = np.abs(stems["voices"] + stems["music"] - mix).max()
    print(f"max |voices + music - mix|      = {err_root:.2e}  ({db(err_root):.0f} dBFS)")
    ok &= err_root < 1e-3
    for parent in ("music", "voices"):
        children = [n for n in stems if STEMS[n].parent == parent]
        if not children:
            continue
        err = np.abs(sum(stems[n] for n in children) - stems[parent]).max()
        print(f"max |sum({', '.join(children)}) - {parent}| = {err:.2e}  ({db(err):.0f} dBFS)")
        ok &= err < 1e-3

    analysis = json.loads((job_dir / "analysis.json").read_text(encoding="utf-8"))
    print(f"analysis: duration {analysis['duration_s']}s, {analysis['frame_rate']} fps")
    for name, s in analysis["stems"].items():
        mean_bands = np.mean(np.array(s["bands"]["frames"]), axis=0)
        loudest = s["bands"]["names"][int(np.argmax(mean_bands))]
        print(f"  {name:8s} energy_share {s['energy_share']:6.3f}  frames {len(s['bands']['frames']):5d}  loudest band: {loudest}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
