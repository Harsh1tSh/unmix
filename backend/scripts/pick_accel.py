"""Choose which accelerator build of torch / onnxruntime to install.

Stdlib only: the start scripts run it *before* the environment exists
(`uv run --no-project python scripts/pick_accel.py`). Prints one JSON line:

    {"group": "cuda12", "reason": "...", "gpu": "Tesla T4"}

Groups (backend/pyproject.toml):
    cuda    NVIDIA Turing or newer (RTX 20+, T4) with a CUDA 13 driver
    cuda12  NVIDIA Maxwell..Hopper with a CUDA 12 driver (torch 2.14.1 + cu126)
    dml     Windows otherwise: DirectML runs the ONNX voice model on any DX12 GPU
    cpu     everything else (macOS uses Apple's GPU through torch either way)

UNMIX_ACCEL=<group> overrides the choice. At run time the server still
measures what actually works (app/accel.py), so a wrong guess here only costs
speed, never correctness.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys

GROUPS = ("cpu", "dml", "cuda", "cuda12")
# torch's CUDA models need about this much GPU memory (see app/accel.py).
MIN_VRAM_MB = 3500


def nvidia() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe and sys.platform == "win32":  # older drivers do not put it on PATH
        legacy = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe")
        exe = legacy if os.path.isfile(legacy) else None
    if not exe:
        return None
    def query(fields: str) -> list[str] | None:
        """First GPU's values, or None if this driver does not know a field."""
        r = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]

    try:
        head = subprocess.run([exe], capture_output=True, text=True, timeout=20).stdout
        base = query("name,memory.total") or []
        cap = query("compute_cap")  # drivers older than ~510 cannot report it
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", head)
    info: dict = {"cuda": (int(m.group(1)), int(m.group(2))) if m else None, "name": base[0] if base and base[0] else "NVIDIA GPU"}
    try:
        info["vram_mb"] = int(float(base[1]))
    except (IndexError, ValueError):
        info["vram_mb"] = None
    try:
        major, minor = cap[0].split(".")  # type: ignore[index]
        info["cap"] = (int(major), int(minor))
    except (TypeError, IndexError, ValueError):
        info["cap"] = None
    return info


def pick() -> dict:
    forced = os.environ.get("UNMIX_ACCEL", "").strip().lower()
    if forced:
        if forced not in GROUPS:
            return {"group": "cpu", "reason": f"UNMIX_ACCEL={forced!r} is not one of {', '.join(GROUPS)}; using cpu."}
        return {"group": forced, "reason": f"UNMIX_ACCEL={forced}"}

    windows = sys.platform == "win32"
    if sys.platform == "darwin":
        reason = "macOS: torch uses the Apple GPU (MPS) when there is one." if platform.machine() == "arm64" else "Intel Mac: CPU."
        return {"group": "cpu", "reason": reason}

    gpu = nvidia()
    fallback = "dml" if windows else "cpu"
    if gpu:
        name, cuda, cap, vram = gpu["name"], gpu["cuda"], gpu["cap"], gpu["vram_mb"]
        if vram is not None and vram < MIN_VRAM_MB:
            why = f"{name} has {vram / 1024:.0f} GB of memory, too little for the CUDA models"
            return {"group": fallback, "gpu": name, "reason": why + ("; DirectML will run the voice model on it." if windows else "; using the CPU.")}
        if cuda is None:
            return {"group": fallback, "gpu": name, "reason": f"Could not read the CUDA version of the {name} driver."}
        if cuda >= (13, 0) and (cap is None or cap >= (7, 5)):
            return {"group": "cuda", "gpu": name, "reason": f"{name}, driver supports CUDA {cuda[0]}.{cuda[1]}."}
        if cuda >= (12, 0) and (cap is None or cap >= (5, 0)):
            return {"group": "cuda12", "gpu": name, "reason": f"{name}, driver supports CUDA {cuda[0]}.{cuda[1]} (CUDA 12 build)."}
        why = f"{name}: driver supports only CUDA {cuda[0]}.{cuda[1]}; updating the NVIDIA driver enables the CUDA build"
        return {"group": fallback, "gpu": name, "reason": why + ("; DirectML meanwhile." if windows else ".")}
    if windows:
        return {"group": "dml", "reason": "Windows: DirectML can run the voice model on the GPU (measured at first start)."}
    return {"group": "cpu", "reason": "No NVIDIA GPU found."}


if __name__ == "__main__":
    print(json.dumps(pick()))
