"""Compare candidate Stage-1 (voices vs. rest) models on this machine:
speed (x realtime) and vocal/accompaniment SDR on MUSDB18 7 s test clips.

Usage:  uv run --group eval python eval/bench_vocal_models.py [--clips 3] [--models a.ckpt b.onnx]

Results are appended to eval/vocal_model_bench.md. This is what the choice of
default vocal model per device in app/config.py is based on.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = [
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "Kim_Vocal_2.onnx",
]


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Plain signal-to-distortion ratio in dB over the whole clip."""
    num = float(np.sum(ref.astype(np.float64) ** 2))
    den = float(np.sum((ref.astype(np.float64) - est.astype(np.float64)) ** 2))
    return 10 * np.log10(max(num, 1e-12) / max(den, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=3)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--overlap", type=int, default=2, help="RoFormer overlap (windows per sample)")
    args = parser.parse_args()

    import musdb
    import torch
    from audio_separator.separator import Separator

    db = musdb.DB(root=str(HERE / "data" / "musdb18-7s"), subsets="test")
    tracks = list(db)[: args.clips]
    clips = [(t.name, t.audio.astype(np.float32), t.targets["vocals"].audio.astype(np.float32)) for t in tracks]
    total_audio = sum(c[1].shape[0] for c in clips) / 44100

    lines = [f"\n### {time.strftime('%Y-%m-%d %H:%M')} - {args.clips} clips ({total_audio:.0f}s), {torch.get_num_threads()} threads, device {('cuda' if torch.cuda.is_available() else 'cpu')}\n",
             "| model | load s | infer s | x realtime | vocals SDR | accompaniment SDR |", "|---|---|---|---|---|---|"]  # fmt: skip

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for model in args.models:
            t0 = time.perf_counter()
            sep = Separator(
                log_level=logging.WARNING, model_file_dir=str(settings.models_dir), output_dir=str(tmp_dir), output_format="WAV",
                normalization_threshold=1.0, output_single_stem="Vocals", use_soundfile=True,
                mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": 1, "overlap": args.overlap, "pitch_shift": 0},
            )  # fmt: skip
            sep.load_model(model_filename=model)
            load_s = time.perf_counter() - t0

            infer_s, v_sdr, a_sdr = 0.0, [], []
            for i, (name, mix, vocals) in enumerate(clips):
                src = tmp_dir / f"clip{i}.wav"
                sf.write(src, mix, 44100, subtype="FLOAT")
                t0 = time.perf_counter()
                sep.separate(str(src), custom_output_names={"Vocals": f"est{i}"})
                infer_s += time.perf_counter() - t0
                est, _ = sf.read(tmp_dir / f"est{i}.wav", dtype="float32", always_2d=True)
                n = min(est.shape[0], mix.shape[0])
                v_sdr.append(sdr(vocals[:n], est[:n]))
                a_sdr.append(sdr(mix[:n] - vocals[:n], mix[:n] - est[:n]))
                print(f"  {model[:40]:40s} {name[:30]:30s} vocals {v_sdr[-1]:5.2f} dB", flush=True)
            row = f"| {model} | {load_s:.1f} | {infer_s:.1f} | {infer_s / total_audio:.2f} | {np.mean(v_sdr):.2f} | {np.mean(a_sdr):.2f} |"
            print(row, flush=True)
            lines.append(row)
            del sep

    out = HERE / "vocal_model_bench.md"
    header = "# Stage 1 vocal model benchmark\n\nx realtime = inference seconds per second of audio (lower is faster). SDR in dB, higher is better.\n" if not out.exists() else ""
    with out.open("a", encoding="utf-8") as fh:
        fh.write(header + "\n".join(lines) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
