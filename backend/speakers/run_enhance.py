"""Voice clean-up runner: MossFormer2_SE_48K speech enhancement.

Runs inside backend/speakers/.venv, called by app/speakers.py:

    speakers/.venv/Scripts/python run_enhance.py <in.wav> <out.wav> <checkpoints_dir> <hf_revision> <sha256>

Input: any WAV (downmixed to mono, resampled to 48 kHz, the model's rate).
Output: 48 kHz mono float WAV with background noise and room sound reduced.
Progress lines "PROGRESS <0..1>" go to stdout.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from run_split import fetch_verified  # same pinned-download + hash check

MODEL = "MossFormer2_SE_48K"
SR = 48000


def main() -> int:
    src, dst, ckpt_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    revision, expected = sys.argv[4], sys.argv[5]
    dst.parent.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    data, sr = sf.read(src, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        g = int(np.gcd(SR, sr))
        mono = resample_poly(mono, SR // g, sr // g).astype(np.float32)
    tmp_in = dst.with_name(dst.stem + "_in48k.wav")
    sf.write(tmp_in, mono, SR, subtype="FLOAT")
    print("PROGRESS 0.05", flush=True)

    fetch_verified(ckpt_root, revision, expected, model=MODEL)
    os.chdir(ckpt_root)
    t0 = time.perf_counter()
    from clearvoice import ClearVoice

    cv = ClearVoice(task="speech_enhancement", model_names=[MODEL])
    print(f"LOG model loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    print("PROGRESS 0.2", flush=True)

    t0 = time.perf_counter()
    out = np.asarray(cv(input_path=str(tmp_in), online_write=False), dtype=np.float32).reshape(-1)[: mono.shape[0]]
    print(f"LOG enhanced in {time.perf_counter() - t0:.1f}s", flush=True)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out *= 0.99 / peak
    sf.write(dst, out, SR, subtype="FLOAT")
    tmp_in.unlink(missing_ok=True)
    print("PROGRESS 1.0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
