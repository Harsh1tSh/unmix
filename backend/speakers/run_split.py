"""Stage 3 runner: split a voices stem into two speakers with MossFormer2_SS_16K.

Runs inside backend/speakers/.venv (ClearerVoice-Studio), never imported by the
backend. The backend (app/speakers.py) calls it as a subprocess:

    speakers/.venv/Scripts/python run_split.py <in.wav> <out_dir> <checkpoints_dir> <hf_revision> <sha256>

The checkpoint is fetched at the pinned Hugging Face revision and its SHA-256
is checked before torch loads it.

Input: any WAV. It is downmixed to mono and resampled to 16 kHz (the model's
rate). Output: <out_dir>/spk1_16k.wav and spk2_16k.wav (16 kHz mono float).
The backend then turns these into full-band 44.1 kHz stems by masking (see
app/speakers.py). Progress lines "PROGRESS <0..1>" go to stdout.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODEL = "MossFormer2_SS_16K"
SR = 16000

# Privacy: no Hugging Face telemetry, never send a stored token.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def fetch_verified(ckpt_root: Path, revision: str, expected_sha256: str, model: str = MODEL) -> None:
    """Download the pinned checkpoint revision (if needed) and check its hash.
    Shared with run_enhance.py."""
    from huggingface_hub import snapshot_download

    target = ckpt_root / "checkpoints" / model
    weights = target / "last_best_checkpoint.pt"
    if not weights.is_file():
        snapshot_download(repo_id=f"alibabasglab/{model}", revision=revision, local_dir=target)
    h = hashlib.sha256()
    with open(weights, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != expected_sha256:
        print(f"ERROR checkpoint hash mismatch ({h.hexdigest()[:12]} != {expected_sha256[:12]}); delete it to re-download", flush=True)
        sys.exit(3)


def main() -> int:
    src, out_dir, ckpt_root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    revision, expected = sys.argv[4], sys.argv[5]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    data, sr = sf.read(src, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SR:
        g = np.gcd(SR, sr)
        mono = resample_poly(mono, SR // g, sr // g).astype(np.float32)
    tmp_in = out_dir / "voices_16k.wav"
    sf.write(tmp_in, mono, SR, subtype="FLOAT")
    print("PROGRESS 0.05", flush=True)

    # ClearVoice resolves checkpoints/<model> relative to the working directory
    # and downloads the weights there from Hugging Face on first use.
    fetch_verified(ckpt_root, revision, expected)
    os.chdir(ckpt_root)
    t0 = time.perf_counter()
    from clearvoice import ClearVoice

    cv = ClearVoice(task="speech_separation", model_names=[MODEL])
    print(f"LOG model loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    print("PROGRESS 0.2", flush=True)

    t0 = time.perf_counter()
    result = cv(input_path=str(tmp_in), online_write=False)
    print(f"LOG separated in {time.perf_counter() - t0:.1f}s", flush=True)

    speakers = [np.asarray(s, dtype=np.float32).reshape(-1) for s in result]
    if len(speakers) != 2:
        print(f"ERROR expected 2 speakers, got {len(speakers)}", flush=True)
        return 2
    for i, spk in enumerate(speakers, start=1):
        spk = spk[: mono.shape[0]]
        if spk.shape[0] < mono.shape[0]:
            spk = np.pad(spk, (0, mono.shape[0] - spk.shape[0]))
        sf.write(out_dir / f"spk{i}_16k.wav", spk, SR, subtype="FLOAT")
    tmp_in.unlink(missing_ok=True)
    print("PROGRESS 1.0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
