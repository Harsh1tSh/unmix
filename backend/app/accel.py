"""Hardware acceleration: where each kind of model runs.

Unmix runs three kinds of models, on three runtimes:

* torch      -- BS-RoFormer (voices) and HT-Demucs (instruments)
* ONNX       -- Kim Vocal 2 (fast-mode voices on CPU machines, live mode)
* CTranslate2 -- Whisper (lyrics)

Policy (all automatic; every piece can be overridden with an env var):

* torch: ``cuda`` if a CUDA GPU works with the installed torch build *and* has
  enough memory for BS-RoFormer, else ``mps`` on Apple Silicon, else ``cpu``.
  A GPU that torch lists but cannot run (too old for this build, broken
  driver) is detected with a tiny test kernel, not assumed. ``UNMIX_DEVICE``.
* ONNX: every execution provider this onnxruntime build offers (CUDA,
  DirectML) is *measured* on the real model in a child process, because a
  faulty GPU driver can take the whole process down, and the fastest one whose
  output matches the CPU's wins. The result is cached in
  ``<models>/accel.json`` (``UNMIX_ACCEL_REPROBE=1`` measures again).
  Measured on a 2017 laptop: CPU 5.3 s per window, MX150 via DirectML 1.1 s;
  the Intel iGPU's DirectML driver fails and is skipped.
* Whisper: ``cuda`` (float16) when torch runs on CUDA and a child-process check
  passes, else ``cpu`` (int8).
* Anything that fails on a GPU during a job is retried on the CPU, and the
  server stays on the CPU until restarted (separation.py).

``python -m app.accel`` prints what was chosen and why.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("unmix.accel")

# BS-RoFormer at segment 256 peaks around 2.5 GB with autocast; below this the
# torch models stay on the CPU (the ONNX model may still use the GPU).
MIN_CUDA_VRAM_GB = float(os.environ.get("UNMIX_MIN_CUDA_VRAM_GB", "3.5"))
PROBE_TIMEOUT_S = 240
CPU_PROVIDERS: list = ["CPUExecutionProvider"]
# A GPU provider must beat the CPU by this factor to be worth its quirks.
MIN_SPEEDUP = 1.25


@dataclass
class Accel:
    torch_device: str = "cpu"  # "cuda" | "mps" | "cpu"
    gpu_name: str | None = None  # the CUDA/MPS device torch uses
    onnx: str = "cpu"  # "cuda" | "directml" | "cpu"
    onnx_providers: list = field(default_factory=lambda: list(CPU_PROVIDERS))
    whisper: str | None = None  # "cuda" | "cpu"; None until lyrics first load
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.torch_device == "cuda":
            return f"GPU ({self.gpu_name})" if self.gpu_name else "GPU (CUDA)"
        if self.torch_device == "mps":
            return "GPU (Apple)"
        if self.onnx == "directml":
            return "CPU + GPU for fast voices (DirectML)"
        if self.onnx == "cuda":
            return "CPU + GPU for fast voices (CUDA)"
        return "CPU"

    def as_dict(self) -> dict:
        return {
            "torch": self.torch_device,
            "onnx": self.onnx,
            "whisper": self.whisper,
            "gpu": self.gpu_name,
            "summary": self.summary(),
            "notes": list(self.notes),
        }


_GPU_ERROR_MARKERS = (
    "cuda", "cudnn", "cublas", "out of memory", "no kernel image", "device-side assert",
    "directml", "dmlexecutionprovider", "887a00", "mps", "metal", "hip error",
)


def looks_like_gpu_failure(exc: BaseException | str) -> bool:
    """Heuristic: did this fail because of the GPU (and might work on the CPU)?"""
    if type(exc).__name__ in ("OutOfMemoryError", "AcceleratorError"):
        return True
    text = str(exc).lower()
    return any(m in text for m in _GPU_ERROR_MARKERS)


# --------------------------------------------------------------------------- #
# CUDA libraries for onnxruntime-gpu and CTranslate2
# --------------------------------------------------------------------------- #

_prepared = False


def prepare_cuda_libs() -> None:
    """Make pip-installed CUDA 12 libraries (cuBLAS, cuDNN) visible to
    onnxruntime-gpu and CTranslate2, which look them up by name at run time.
    torch finds its own; the others need help. Safe to call anywhere."""
    global _prepared
    if _prepared:
        return
    _prepared = True
    try:
        import nvidia  # namespace package of the nvidia-* wheels

        roots = [Path(p) for p in nvidia.__path__]
    except ImportError:
        roots = []
    try:
        import torch

        roots_torch = [Path(torch.__file__).parent / "lib"]
    except ImportError:
        roots_torch = []
    if sys.platform == "win32":
        dirs = [d for r in roots for d in r.glob("*/bin") if d.is_dir()] + [d for d in roots_torch if d.is_dir()]
        for d in dirs:
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass
        if dirs:
            os.environ["PATH"] = os.pathsep.join([*map(str, dirs), os.environ.get("PATH", "")])
    elif sys.platform.startswith("linux"):
        import ctypes

        # Load order matters: cuBLASLt before cuBLAS, cuDNN's core before its parts.
        for pattern in ("cublas/lib/libcublasLt.so.*", "cublas/lib/libcublas.so.*", "cudnn/lib/libcudnn.so.*", "cudnn/lib/libcudnn_*.so.*"):
            for r in roots:
                for lib in sorted(r.glob(pattern)):
                    try:
                        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers() and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception as exc:  # noqa: BLE001
        log.debug("onnxruntime preload_dlls failed: %s", exc)


# --------------------------------------------------------------------------- #
# torch
# --------------------------------------------------------------------------- #


def pick_torch_device(notes: list[str]) -> tuple[str, str | None]:
    forced = os.environ.get("UNMIX_DEVICE", "auto").strip().lower()
    import torch

    if forced == "cpu":
        notes.append("UNMIX_DEVICE=cpu: GPU use is switched off.")
        return "cpu", None

    if forced in ("auto", "cuda") and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        built_for = torch.cuda.get_arch_list()
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if built_for and arch not in built_for and not any(a.startswith("compute_") and int(a[8:]) <= major * 10 + minor for a in built_for):
            notes.append(f"{name} ({arch}) is too old for this PyTorch build; the torch models run on the CPU.")
        elif vram < MIN_CUDA_VRAM_GB and forced != "cuda":
            notes.append(f"{name} has {vram:.1f} GB of memory (BS-RoFormer needs about {MIN_CUDA_VRAM_GB:g} GB); the torch models run on the CPU.")
        else:
            try:
                x = torch.arange(1024, device="cuda", dtype=torch.float32)
                if float((x * 2).sum().item()) != 1047552.0:
                    raise RuntimeError("wrong result from a test kernel")
                return "cuda", name
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{name} failed a CUDA test ({type(exc).__name__}); the torch models run on the CPU.")
    elif forced == "cuda":
        notes.append("UNMIX_DEVICE=cuda but this PyTorch build has no working CUDA; using the CPU.")

    if forced in ("auto", "mps"):
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available() and platform.machine() == "arm64":
            return "mps", "Apple GPU"
    return "cpu", None


# --------------------------------------------------------------------------- #
# ONNX Runtime: measure, don't guess
# --------------------------------------------------------------------------- #


def ort_providers(providers: list) -> list:
    """JSON round-trips ("Name", {opts}) tuples as lists; onnxruntime wants tuples."""
    return [tuple(p) if isinstance(p, list) else p for p in providers]


def onnx_candidates() -> list[tuple[str, list]]:
    """(label, providers) for every GPU provider this onnxruntime offers, best guess first."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    out: list[tuple[str, list]] = []
    if "CUDAExecutionProvider" in available:
        out.append(("cuda", [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]))
    if "DmlExecutionProvider" in available:
        # "high_performance" picks the discrete GPU on hybrid laptops; explicit
        # adapter ids are the fallback when that one fails.
        out.append(("directml", [("DmlExecutionProvider", {"performance_preference": "high_performance", "device_filter": "gpu"}), "CPUExecutionProvider"]))
        for i in range(3):
            out.append(("directml", [("DmlExecutionProvider", {"device_id": i}), "CPUExecutionProvider"]))
    # CoreML is not offered: the MDX model's 3072x256 input is slower there than on Apple CPUs.
    return out


def _probe_child(model: str, providers: list) -> dict:
    """Runs in a child process: time one window of the model on `providers`."""
    import numpy as np

    prepare_cuda_libs()
    import onnxruntime as ort

    x = np.random.default_rng(0).standard_normal((1, 4, 3072, 256)).astype(np.float32)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    session = ort.InferenceSession(model, opts, providers=ort_providers(providers))
    wanted = providers[0][0] if isinstance(providers[0], (list, tuple)) else providers[0]
    if session.get_providers()[0] != wanted:
        return {"ok": False, "error": f"{wanted} could not start (onnxruntime fell back to {session.get_providers()[0]})"}
    out = session.run(None, {"input": x})[0]  # warm-up (DirectML compiles shaders here)
    t0 = time.perf_counter()
    out = session.run(None, {"input": x})[0]
    seconds = time.perf_counter() - t0
    w = np.random.default_rng(1).standard_normal(out.size).astype(np.float64)
    return {"ok": True, "seconds": seconds, "abs": float(np.abs(out).sum()), "dot": float(out.astype(np.float64).ravel() @ w)}


def _run_probe(model: Path, providers: list) -> dict:
    cmd = [sys.executable, "-m", "app.accel", "--probe-onnx", str(model), json.dumps(providers)]
    try:
        r = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timed out"}
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("{")), None)
    if r.returncode != 0 or line is None:
        tail = (r.stderr or "").strip().splitlines()[-1:] or [f"exit code {r.returncode}"]
        return {"ok": False, "error": tail[0][:300]}
    return json.loads(line)


def _cache_key() -> dict:
    import onnxruntime as ort

    return {"onnxruntime": ort.__version__, "providers": ort.get_available_providers(), "platform": platform.platform(), "python": sys.version.split()[0]}


def _close(a: dict, b: dict) -> bool:
    """Same output as the CPU, up to float16-ish rounding."""
    rel = lambda x, y: abs(x - y) / max(abs(y), 1e-6)  # noqa: E731
    return rel(a["abs"], b["abs"]) < 1e-2 and rel(a["dot"], b["dot"]) < 5e-2


def pick_onnx(model: Path, cache_file: Path, notes: list[str]) -> tuple[str, list]:
    """Measure the GPU providers on `model` (once; cached) and return the winner."""
    forced = os.environ.get("UNMIX_ONNX_PROVIDER", "auto").strip().lower()
    if forced == "cpu" or os.environ.get("UNMIX_DEVICE", "").strip().lower() == "cpu":
        return "cpu", list(CPU_PROVIDERS)
    candidates = onnx_candidates()
    if forced in ("cuda", "directml"):
        candidates = [c for c in candidates if c[0] == forced]
    if not candidates:
        return "cpu", list(CPU_PROVIDERS)

    key = _cache_key()
    if cache_file.is_file() and os.environ.get("UNMIX_ACCEL_REPROBE", "") not in ("1", "true"):
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("key") == key and forced == cached.get("forced", "auto"):
                choice = cached["onnx"]
                notes.extend(cached.get("notes", []))
                return choice["label"], ort_providers(choice["providers"])
        except (OSError, ValueError, KeyError):
            pass

    t0 = time.perf_counter()
    log.info("measuring ONNX Runtime providers on %s (first start only)...", model.name)
    cpu = _run_probe(model, list(CPU_PROVIDERS))
    found: list[str] = []
    best: tuple[str, list, float] | None = None
    good: set[str] = set()  # providers with a configuration that already beats the CPU
    for label, providers in candidates:
        if label in good:
            continue
        result = _run_probe(model, providers)
        opts = providers[0][1] if isinstance(providers[0], (list, tuple)) else {}
        what = f"{label} {opts}"
        if not result.get("ok"):
            log.info("  %s: unavailable (%s)", what, result.get("error"))
            continue
        if cpu.get("ok") and not _close(result, cpu):
            log.warning("  %s: output differs from the CPU's; not used", what)
            found.append(f"{label} gave different results from the CPU and is not used.")
            continue
        log.info("  %s: %.2fs per window (CPU %.2fs)", what, result["seconds"], cpu.get("seconds", float("nan")))
        if not cpu.get("ok") or result["seconds"] * MIN_SPEEDUP < cpu["seconds"]:
            good.add(label)
        if best is None or result["seconds"] < best[2]:
            best = (label, providers, result["seconds"])
    chosen: tuple[str, list] = ("cpu", list(CPU_PROVIDERS))
    if best is not None and (not cpu.get("ok") or best[2] * MIN_SPEEDUP < cpu["seconds"]):
        chosen = (best[0], best[1])
        if cpu.get("ok"):
            found.append(f"Fast voice model: {best[0]} {cpu['seconds'] / best[2]:.1f}x faster than the CPU ({best[2]:.1f}s vs {cpu['seconds']:.1f}s per window).")
    elif best is not None:
        found.append(f"{best[0]} was not faster than the CPU for the voice model; using the CPU.")
    log.info("ONNX provider: %s (measured in %.0fs)", chosen[0], time.perf_counter() - t0)
    notes.extend(found)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": key, "forced": forced, "onnx": {"label": chosen[0], "providers": chosen[1]}, "notes": found}
        cache_file.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    except OSError:
        pass
    return chosen[0], ort_providers(chosen[1])


class OnnxRunner:
    """The one onnxruntime session for a model in this process.

    Fast mode and live mode both run Kim Vocal 2. They share this session
    because a second DirectML session in the same process hangs the GPU on
    small cards (measured on an MX150: "887A0006 The GPU will not respond"),
    and one copy of the weights is enough anyway. Calls are serialised (DirectML
    does not allow concurrent runs on a session). If the GPU fails, the session
    is rebuilt on the CPU and the call retried; `on_fallback` is told why.
    """

    def __init__(self, path: Path, providers: list, on_fallback=None):
        import threading

        self.path = Path(path)
        self.providers = ort_providers(providers)
        self.on_fallback = on_fallback
        self._lock = threading.Lock()
        try:
            self.session = self._session(self.providers)
        except Exception as exc:  # noqa: BLE001
            self._to_cpu(exc)

    @property
    def on_gpu(self) -> bool:
        return self.providers != CPU_PROVIDERS

    def _session(self, providers: list):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        return ort.InferenceSession(str(self.path), opts, providers=providers)

    def _to_cpu(self, exc: BaseException) -> None:
        log.error("%s failed on the GPU, switching to the CPU: %s", self.path.name, exc)
        self.providers = list(CPU_PROVIDERS)
        self.session = self._session(self.providers)
        if self.on_fallback:
            self.on_fallback(exc)

    def __call__(self, x):
        """x: (1, 4, F, T) float32 -> the model's first output."""
        with self._lock:
            try:
                return self.session.run(None, {"input": x})[0]
            except Exception as exc:
                if not self.on_gpu:
                    raise
                self._to_cpu(exc)
                return self.session.run(None, {"input": x})[0]


# --------------------------------------------------------------------------- #
# Whisper (CTranslate2)
# --------------------------------------------------------------------------- #


def _whisper_child(model_dir: str) -> dict:
    import numpy as np

    prepare_cuda_libs()
    from faster_whisper import WhisperModel

    model = WhisperModel(model_dir, device="cuda", compute_type="float16")
    audio = (0.05 * np.random.default_rng(0).standard_normal(16000 * 2)).astype(np.float32)
    list(model.transcribe(audio, beam_size=1, word_timestamps=True, vad_filter=False)[0])
    return {"ok": True}


def whisper_cuda_works(model_dir: Path, cache_file: Path) -> tuple[bool, str]:
    """Check (in a child process, cached) that Whisper runs on CUDA here.
    CTranslate2 aborts the process outright when a CUDA library is missing,
    so this must never be tried in the server process first."""
    key = {**_cache_key(), "model": model_dir.name}
    cache = {}
    if cache_file.is_file():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
            w = cache.get("whisper")
            if w and w.get("key") == key:
                return bool(w["ok"]), w.get("error", "")
        except (OSError, ValueError):
            cache = {}
    cmd = [sys.executable, "-m", "app.accel", "--probe-whisper", str(model_dir)]
    try:
        r = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        ok = r.returncode == 0 and '"ok": true' in r.stdout
        error = "" if ok else ((r.stderr or "").strip().splitlines()[-1:] or [f"exit code {r.returncode}"])[0][:300]
    except subprocess.TimeoutExpired:
        ok, error = False, "timed out"
    try:
        cache["whisper"] = {"key": key, "ok": ok, "error": error}
        cache_file.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    except OSError:
        pass
    return ok, error


# --------------------------------------------------------------------------- #


def _main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[0] == "--probe-onnx":
        print(json.dumps(_probe_child(argv[1], json.loads(argv[2]))))
        return 0
    if len(argv) >= 2 and argv[0] == "--probe-whisper":
        print(json.dumps(_whisper_child(argv[1])))
        return 0
    # Human-readable report: python -m app.accel
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .config import settings
    from .live import kim_model_path

    notes: list[str] = []
    device, gpu = pick_torch_device(notes)
    print(f"torch models : {device}" + (f" ({gpu})" if gpu else ""))
    model = kim_model_path(settings)
    if model.is_file():
        label, _ = pick_onnx(model, settings.models_dir / "accel.json", notes)
        print(f"ONNX model   : {label}")
    else:
        print("ONNX model   : not downloaded yet (start the server once)")
    for n in notes:
        print(" -", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
