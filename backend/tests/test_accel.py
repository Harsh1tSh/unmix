"""GPU selection and CPU fallback (app/accel.py, scripts/pick_accel.py).

No GPU is needed: probes and failures are simulated. The real measurements
are covered by `python -m app.accel` and the Colab notebook's self-check."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from app import accel
from app.config import Settings

BACKEND = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ torch


def test_unmix_device_cpu_switches_the_gpu_off(monkeypatch):
    monkeypatch.setenv("UNMIX_DEVICE", "cpu")
    notes: list[str] = []
    assert accel.pick_torch_device(notes) == ("cpu", None)
    assert any("switched off" in n for n in notes)


def test_gpu_failure_heuristic():
    for msg in ("CUDA out of memory. Tried to allocate 2.00 GiB", "CUBLAS_STATUS_NOT_INITIALIZED", "no kernel image is available for execution on the device",
                "[ONNXRuntimeError] : 1 : FAIL : DmlExecutionProvider ... 887A0005", "MPS backend out of memory"):
        assert accel.looks_like_gpu_failure(RuntimeError(msg)), msg
    for msg in ("Could not decode this file as audio.", "division by zero", "No such file or directory"):
        assert not accel.looks_like_gpu_failure(RuntimeError(msg)), msg


# ------------------------------------------------------------------- ONNX


@pytest.fixture()
def model(tmp_path) -> Path:
    p = tmp_path / "Kim_Vocal_2.onnx"
    p.write_bytes(b"onnx")
    return p


DML_HP = [("DmlExecutionProvider", {"performance_preference": "high_performance", "device_filter": "gpu"}), "CPUExecutionProvider"]
DML_0 = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
DML_1 = [("DmlExecutionProvider", {"device_id": 1}), "CPUExecutionProvider"]
SAME = {"abs": 1000.0, "dot": 50.0}


def _fake_probes(monkeypatch, results: dict[str, dict]) -> list[str]:
    """results: provider description -> probe result; returns the call log."""
    calls: list[str] = []

    def fake(model, providers):
        key = json.dumps(accel.ort_providers(providers), default=str)
        calls.append(key)
        for k, v in results.items():
            if k in key:
                return v
        return {"ok": False, "error": "not simulated"}

    monkeypatch.setattr(accel, "_run_probe", fake)
    return calls


def test_fastest_matching_gpu_provider_wins_and_is_cached(monkeypatch, model, tmp_path):
    monkeypatch.setattr(accel, "onnx_candidates", lambda: [("directml", DML_HP), ("directml", DML_0), ("directml", DML_1)])
    monkeypatch.setattr(accel, "_cache_key", lambda: {"k": 1})
    calls = _fake_probes(monkeypatch, {
        '["CPUExecutionProvider"]': {"ok": True, "seconds": 5.3, **SAME},
        "high_performance": {"ok": False, "error": "887A0005"},  # like this laptop's Intel iGPU
        '"device_id": 0': {"ok": True, "seconds": 1.1, "abs": 1000.4, "dot": 50.2},
    })
    cache = tmp_path / "accel.json"
    notes: list[str] = []
    label, providers = accel.pick_onnx(model, cache, notes)
    assert label == "directml" and providers[0] == ("DmlExecutionProvider", {"device_id": 0})
    assert any("faster than the CPU" in n for n in notes)
    assert not any('"device_id": 1' in c for c in calls), "a working DirectML config makes further ones unnecessary"

    # Second start: read from the cache, nothing is measured again.
    monkeypatch.setattr(accel, "_run_probe", lambda *a: pytest.fail("probed despite the cache"))
    assert accel.pick_onnx(model, cache, []) == (label, providers)


def test_gpu_with_different_output_or_no_speedup_is_not_used(monkeypatch, model, tmp_path):
    monkeypatch.setattr(accel, "onnx_candidates", lambda: [("directml", DML_HP)])
    monkeypatch.setattr(accel, "_cache_key", lambda: {"k": 2})
    _fake_probes(monkeypatch, {'["CPUExecutionProvider"]': {"ok": True, "seconds": 5.0, **SAME}, "high_performance": {"ok": True, "seconds": 0.5, "abs": 1400.0, "dot": 50.0}})
    notes: list[str] = []
    assert accel.pick_onnx(model, tmp_path / "a.json", notes)[0] == "cpu"
    assert any("different results" in n for n in notes)

    _fake_probes(monkeypatch, {'["CPUExecutionProvider"]': {"ok": True, "seconds": 5.0, **SAME}, "high_performance": {"ok": True, "seconds": 4.5, **SAME}})
    notes = []
    assert accel.pick_onnx(model, tmp_path / "b.json", notes)[0] == "cpu"
    assert any("not faster" in n for n in notes)


def test_cpu_only_onnxruntime_measures_nothing(monkeypatch, model, tmp_path):
    monkeypatch.setattr(accel, "onnx_candidates", lambda: [])
    monkeypatch.setattr(accel, "_run_probe", lambda *a: pytest.fail("nothing to measure"))
    assert accel.pick_onnx(model, tmp_path / "c.json", []) == ("cpu", ["CPUExecutionProvider"])


@pytest.fixture()
def relu_model(tmp_path) -> Path:
    """A tiny ONNX model with Kim Vocal 2's input name and shape."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 3072, 256])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 3072, 256])
    graph = helper.make_graph([helper.make_node("Relu", ["input"], ["output"])], "g", [x], [y])
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    path = tmp_path / "relu.onnx"
    onnx.save(m, str(path))
    return path


def test_probe_child_runs_a_real_onnx_model(relu_model):
    """The child-process probe end to end."""
    result = accel._run_probe(relu_model, ["CPUExecutionProvider"])
    assert result["ok"] and result["seconds"] > 0 and result["abs"] > 0


def test_shared_onnx_session_falls_back_to_cpu(relu_model):
    reasons: list[str] = []
    runner = accel.OnnxRunner(relu_model, ["CPUExecutionProvider"], on_fallback=lambda e: reasons.append(str(e)))
    x = np.random.default_rng(0).standard_normal((1, 4, 3072, 256)).astype(np.float32)
    expected = np.maximum(x, 0)

    class HungGpu:
        def run(self, *a, **k):
            raise RuntimeError("887A0006 The GPU will not respond to more commands")

    runner.session, runner.providers = HungGpu(), [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    assert np.array_equal(runner(x), expected)
    assert not runner.on_gpu and reasons and "887A0006" in reasons[0]
    assert np.array_equal(runner(x), expected)  # stays on the CPU


# ------------------------------------------------------- run-time fallback


def test_gpu_failure_mid_job_retries_on_cpu_and_stays_there(tmp_path):
    from app.separation import AudioSeparatorBackend

    b = AudioSeparatorBackend(Settings(data_dir=tmp_path))
    b.accel.torch_device, b.accel.gpu_name = "cuda", "Pretend GPU"
    b.accel.onnx = "cuda"
    attempts: list[str] = []

    def job():
        attempts.append(b.device)
        if b.device == "cuda":
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.50 GiB")
        return "stems"

    assert b._cpu_fallback("voice separation", job) == "stems"
    assert attempts == ["cuda", "cpu"]
    assert b.device == "cpu" and b.accel.onnx == "cpu" and b.accel.onnx_providers == ["CPUExecutionProvider"]
    assert any("GPU failed during voice separation" in n for n in b.accel.notes)
    assert b.accel.summary() == "CPU"


def test_non_gpu_errors_are_not_retried(tmp_path):
    from app.separation import AudioSeparatorBackend

    b = AudioSeparatorBackend(Settings(data_dir=tmp_path))
    b.accel.torch_device = "cuda"
    with pytest.raises(ValueError):
        b._cpu_fallback("x", lambda: (_ for _ in ()).throw(ValueError("bad input")))
    assert b.device == "cuda", "a bad file must not push the server off the GPU"


def test_whisper_falls_back_to_cpu(monkeypatch, tmp_path):
    from app.lyrics import WhisperTranscriber

    t = WhisperTranscriber(Settings(data_dir=tmp_path), "cuda")
    calls: list[str] = []

    def fake(voices, sr, progress):
        calls.append(t.device)
        if t.device == "cuda":
            raise RuntimeError("CUDA failed with error CUBLAS_STATUS_NOT_SUPPORTED")
        return {"segments": []}

    monkeypatch.setattr(t, "_transcribe", fake)
    monkeypatch.setattr(t, "_make", lambda path: object())
    assert t.transcribe(np.zeros((10, 2), np.float32), 44100, lambda p: None) == {"segments": []}
    assert calls == ["cuda", "cpu"] and t.device == "cpu" and t.note


def test_voice_split_retries_on_cpu(monkeypatch):
    from app import speakers

    calls: list[bool] = []

    def once(script, args, settings, progress, what, gpu):
        calls.append(gpu)
        if gpu:
            raise RuntimeError("speaker separation exited with code 1: torch.OutOfMemoryError: CUDA out of memory")

    monkeypatch.setattr(speakers, "_run_runner_once", once)
    speakers._run_runner("run_split.py", [], Settings(), lambda p: None, "speaker separation", gpu=True)
    assert calls == [True, False]


def test_voice_split_hides_the_gpu_unless_told(monkeypatch, tmp_path):
    from app import speakers

    seen = {}

    class P:
        stdout = iter(())

        def __init__(self, cmd, **kw):
            seen.update(kw["env"])

        def wait(self):
            return 0

    py = tmp_path / "python.exe"
    py.write_text("")
    monkeypatch.setattr(speakers, "speakers_python", lambda s: py)
    monkeypatch.setattr(speakers.subprocess, "Popen", P)
    speakers._run_runner("run_split.py", [], Settings(), lambda p: None, "x", gpu=False)
    assert seen["CUDA_VISIBLE_DEVICES"] == ""
    seen.clear()
    speakers._run_runner("run_split.py", [], Settings(), lambda p: None, "x", gpu=True)
    assert "CUDA_VISIBLE_DEVICES" not in seen or seen["CUDA_VISIBLE_DEVICES"] != ""


# ------------------------------------------------- install-time choice


def _pick_accel():
    spec = importlib.util.spec_from_file_location("pick_accel", BACKEND / "scripts" / "pick_accel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "platform,gpu,expected",
    [
        ("linux", {"name": "Tesla T4", "cuda": (13, 0), "cap": (7, 5), "vram_mb": 15360}, "cuda"),
        ("linux", {"name": "Tesla T4", "cuda": (12, 8), "cap": (7, 5), "vram_mb": 15360}, "cuda12"),
        ("win32", {"name": "GeForce GTX 1080", "cuda": (12, 4), "cap": (6, 1), "vram_mb": 8192}, "cuda12"),
        ("win32", {"name": "GeForce GTX 1080", "cuda": (13, 1), "cap": (6, 1), "vram_mb": 8192}, "cuda12"),  # Pascal: no CUDA 13 torch
        ("win32", {"name": "GeForce MX150", "cuda": (11, 0), "cap": None, "vram_mb": 2048}, "dml"),  # this laptop
        ("linux", {"name": "GeForce MX150", "cuda": (12, 2), "cap": (6, 1), "vram_mb": 2048}, "cpu"),
        ("win32", {"name": "GeForce RTX 3060", "cuda": (11, 4), "cap": (8, 6), "vram_mb": 12288}, "dml"),  # old driver
        ("win32", None, "dml"),
        ("linux", None, "cpu"),
        ("darwin", None, "cpu"),
    ],
)
def test_pick_accel(monkeypatch, platform, gpu, expected):
    mod = _pick_accel()
    monkeypatch.delenv("UNMIX_ACCEL", raising=False)
    monkeypatch.setattr(mod.sys, "platform", platform)
    monkeypatch.setattr(mod, "nvidia", lambda: gpu)
    choice = mod.pick()
    assert choice["group"] == expected, choice
    assert choice["reason"]


def test_nvidia_smi_parsing_with_an_old_driver(monkeypatch):
    """Drivers before ~510 reject the compute_cap field (this laptop's 451.67 does)."""
    import subprocess as sp

    mod = _pick_accel()

    def fake_run(cmd, **kw):
        q = next((a for a in cmd if a.startswith("--query-gpu=")), None)
        if q is None:
            return sp.CompletedProcess(cmd, 0, "| NVIDIA-SMI 451.67   Driver Version: 451.67   CUDA Version: 11.0 |\n", "")
        if "compute_cap" in q:
            return sp.CompletedProcess(cmd, 2, 'Field "compute_cap" is not a valid field to query.\n', "")
        return sp.CompletedProcess(cmd, 0, "GeForce MX150, 2048\n", "")

    monkeypatch.setattr(mod.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert mod.nvidia() == {"cuda": (11, 0), "name": "GeForce MX150", "vram_mb": 2048, "cap": None}


def _locked(project: Path, group: str) -> dict[str, str]:
    """name -> pinned version for a dependency group, from uv.lock (no network)."""
    import shutil
    import subprocess

    uv = shutil.which("uv")
    if not uv:
        pytest.skip("uv not on PATH")
    out = subprocess.run([uv, "export", "--frozen", "--no-default-groups", "--group", group, "--no-hashes", "--format", "requirements-txt"],
                         cwd=project, capture_output=True, text=True, check=True).stdout
    pins: dict[str, str] = {}
    for line in out.splitlines():
        if line.startswith(("#", " ")) or "==" not in line or "sys_platform != 'linux'" in line:
            continue  # comments, continuation lines, and pins for other platforms
        name, _, rest = line.partition("==")
        pins[name.strip()] = rest.split(";")[0].strip()
    return pins


@pytest.mark.parametrize("project,group,cuda_tag", [
    (BACKEND, "cuda", "cu132"), (BACKEND, "cuda12", "cu126"), (BACKEND / "speakers", "cuda", "cu132"),
])
def test_gpu_builds_have_matching_cuda_versions_on_linux(project, group, cuda_tag):
    """torchvision/torchaudio refuse to load next to a torch with another CUDA major
    version (Colab failed on exactly this). PyPI's Linux builds of them are CUDA 13,
    so a CUDA 12 torch needs them from the same PyTorch index."""
    pins = _locked(project, group)
    assert pins["torch"].endswith("+" + cuda_tag), pins["torch"]
    for lib in ("torchvision", "torchaudio"):
        if lib in pins:
            local = pins[lib].partition("+")[2]
            if cuda_tag.startswith("cu12"):
                assert local == cuda_tag, f"{lib} {pins[lib]} next to torch {pins['torch']}"
            else:
                assert local in ("", cuda_tag), f"{lib} {pins[lib]}: PyPI's Linux build is CUDA 13, else it must be {cuda_tag}"


def test_pick_accel_override(monkeypatch):
    mod = _pick_accel()
    monkeypatch.setenv("UNMIX_ACCEL", "cuda12")
    assert mod.pick()["group"] == "cuda12"
    monkeypatch.setenv("UNMIX_ACCEL", "rocm")
    assert mod.pick()["group"] == "cpu"
