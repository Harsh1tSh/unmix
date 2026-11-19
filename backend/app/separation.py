"""Separation backends.

``SeparationBackend`` is the seam between the pipeline and the neural models:

* ``AudioSeparatorBackend`` -- the real thing (BS-RoFormer + HT-Demucs through
  the ``audio-separator`` package).
* ``FakeBackend`` -- cheap DSP filters with the same interface, used by the test
  suite and for frontend work without multi-GB weights (UNMIX_FAKE_MODELS=1).

Both return float32 arrays shaped (samples, 2). They return only what the model
*estimates*; the pipeline derives the residual stems (music = mix - voices,
other = music - named stems) so that stems always sum back to the mix exactly.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Protocol, TypeVar

import numpy as np

from . import audio_io
from .config import Settings

log = logging.getLogger("unmix.separation")

ProgressFn = Callable[[float], None]
T = TypeVar("T")

# Stems each music model estimates directly. Anything else it produces (its own
# "vocals"/"other" guess) is folded into our residual "other".
MUSIC_STEMS_FAST = ["drums", "bass", "guitar", "piano"]
MUSIC_STEMS_QUALITY = ["drums", "bass"]


def music_stem_names(mode: str) -> list[str]:
    return MUSIC_STEMS_QUALITY if mode == "quality" else MUSIC_STEMS_FAST


def detect_device() -> str:
    """Where the torch models will run (see app/accel.py for the policy)."""
    from .accel import pick_torch_device

    return pick_torch_device([])[0]


class SeparationBackend(Protocol):
    device: str
    accel: "Accel"  # noqa: F821 - app.accel.Accel; what runs where, for /api/health

    def load(self) -> None: ...

    def models_for(self, mode: str) -> dict[str, str]: ...

    def separate_voices(self, mix_path: Path, work_dir: Path, mode: str, progress: ProgressFn) -> np.ndarray: ...

    def separate_music(self, music_path: Path, work_dir: Path, mode: str, progress: ProgressFn) -> dict[str, np.ndarray]: ...

    def speaker_split_available(self) -> bool: ...

    def separate_speakers(self, voices_path: Path, voices: np.ndarray, work_dir: Path, progress: ProgressFn) -> tuple[np.ndarray, np.ndarray]: ...

    def transcribe(self, voices: np.ndarray, progress: ProgressFn) -> dict: ...

    def enhance(self, src: Path, dst: Path, progress: ProgressFn) -> None: ...

    def live_separator(self): ...


# --------------------------------------------------------------------------- #
# tqdm hook: audio-separator only reports progress through tqdm bars.
# --------------------------------------------------------------------------- #


class _ProgressHook:
    """Routes tqdm iteration counts to the active job's progress callback.

    A stage may run several tqdm loops ("passes"): Demucs runs one per model in
    a bag and one per random shift. ``expected_passes`` lets us map them onto a
    single 0..1 range.
    """

    def __init__(self) -> None:
        self._callback: ProgressFn | None = None
        self._expected_passes = 1
        self._passes_done = 0

    def start(self, callback: ProgressFn, expected_passes: int = 1) -> None:
        self._callback = callback
        self._expected_passes = max(1, expected_passes)
        self._passes_done = 0

    def stop(self) -> None:
        self._callback = None

    def tqdm(self, iterable, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003 - mirrors tqdm.tqdm
        try:
            total = len(iterable)
        except TypeError:
            total = 0
        for i, item in enumerate(iterable):
            yield item
            if self._callback and total:
                frac = (self._passes_done + (i + 1) / total) / self._expected_passes
                self._callback(min(1.0, frac))
        self._passes_done += 1


_hook = _ProgressHook()


class _TqdmModuleShim:
    """Stands in for the ``tqdm`` module where code calls ``tqdm.tqdm(...)``."""

    tqdm = staticmethod(lambda iterable, *a, **k: _hook.tqdm(iterable, *a, **k))


def _install_tqdm_hooks() -> None:
    from audio_separator.separator.architectures import mdx_separator, mdxc_separator
    from audio_separator.separator.uvr_lib_v5.demucs import apply as demucs_apply

    mdx_separator.tqdm = _hook.tqdm
    mdxc_separator.tqdm = _hook.tqdm
    demucs_apply.tqdm = _TqdmModuleShim


# --------------------------------------------------------------------------- #
# Real backend
# --------------------------------------------------------------------------- #


class AudioSeparatorBackend:
    def __init__(self, settings: Settings):
        from .accel import Accel, pick_torch_device

        self.settings = settings
        self.accel = Accel()
        self.accel.torch_device, self.accel.gpu_name = pick_torch_device(self.accel.notes)
        self._vocal: dict[str, tuple[object, str]] = {}  # requested model -> (separator, model actually loaded)
        self._music: dict[str, object] = {}
        self._lock = threading.Lock()
        self._transcriber = None
        self._live = None
        self._onnx: dict[str, object] = {}  # model path -> accel.OnnxRunner
        self._onnx_lock = threading.Lock()
        self._onnx_picked = False

    @property
    def device(self) -> str:
        """Where the torch models run: "cuda" | "mps" | "cpu"."""
        return self.accel.torch_device

    @property
    def _demucs_shifts(self) -> int:
        return 0 if self.device == "cpu" else 2

    # -- accelerators -------------------------------------------------------

    def _pick_onnx(self) -> None:
        """Measure the ONNX providers once (needs the Kim Vocal 2 file)."""
        from .accel import onnx_candidates, pick_onnx, prepare_cuda_libs
        from .integrity import verify
        from .live import kim_model_path

        if self._onnx_picked:
            return
        self._onnx_picked = True
        prepare_cuda_libs()
        if not onnx_candidates() or os.environ.get("UNMIX_DEVICE", "").strip().lower() == "cpu":
            return
        path = kim_model_path(self.settings)
        if not path.is_file():
            self._new_separator(single_stem="Vocals").download_model_files(path.name)
        verify(path, path.name, self.settings.models_dir)
        self.accel.onnx, self.accel.onnx_providers = pick_onnx(path, self.settings.models_dir / "accel.json", self.accel.notes)

    def _on_gpu(self) -> bool:
        return self.device != "cpu" or self.accel.onnx != "cpu"

    def _fall_back_to_cpu(self, exc: BaseException, what: str) -> None:
        """A GPU failed mid-job: drop every GPU-loaded model and stay on the CPU
        until the server restarts (a driver that failed once tends to again)."""
        from .accel import CPU_PROVIDERS

        with self._lock:
            if not self._on_gpu():
                return
            log.error("the GPU failed during %s; switching to the CPU: %s: %s", what, type(exc).__name__, exc)
            self.accel.notes.append(f"The GPU failed during {what} ({type(exc).__name__}); everything runs on the CPU until the server restarts.")
            self.accel.torch_device, self.accel.gpu_name = "cpu", None
            self.accel.onnx, self.accel.onnx_providers = "cpu", list(CPU_PROVIDERS)
            self._vocal.clear()
            self._music.clear()
            self._live = None
            self._transcriber = None
            with self._onnx_lock:
                self._onnx.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _cpu_fallback(self, what: str, fn: Callable[[], T]) -> T:
        from .accel import looks_like_gpu_failure

        try:
            return fn()
        except Exception as exc:
            if not self._on_gpu() or not looks_like_gpu_failure(exc):
                raise
            self._fall_back_to_cpu(exc, what)
            return fn()

    # -- model policy -------------------------------------------------------

    def vocal_model_for(self, mode: str) -> str:
        """Which Stage 1 model a job gets.

        BS-RoFormer is the best vocal model we have but costs ~20x realtime on a
        laptop CPU (eval/vocal_model_bench.md). So on CPU, `fast` mode uses a
        lighter MDX-Net model (~3x realtime, about 3 dB lower SDR) and BS-RoFormer
        is kept for `quality`. With a GPU, BS-RoFormer is always used.
        Set UNMIX_VOCAL_MODEL_CPU_FAST="" to force BS-RoFormer everywhere.
        """
        if self.device == "cpu" and mode != "quality" and self.settings.vocal_model_cpu_fast:
            return self.settings.vocal_model_cpu_fast
        return self.settings.vocal_model

    def models_for(self, mode: str) -> dict[str, str]:
        requested = self.vocal_model_for(mode)
        loaded = self._vocal.get(requested)
        return {
            "voices": loaded[1] if loaded else requested,
            "music": self.settings.music_model_quality if mode == "quality" else self.settings.music_model_fast,
        }

    # -- construction -------------------------------------------------------

    def _new_separator(self, *, single_stem: str | None = None):
        import torch
        from audio_separator.separator import Separator

        on_gpu = self.device != "cpu"
        sep = Separator(
            log_level=logging.WARNING,
            model_file_dir=str(self.settings.models_dir),
            output_format="WAV",
            # 1.0 == never rescale. The library otherwise peak-normalises every
            # stem independently, which would break "stems sum to the mix".
            normalization_threshold=1.0,
            amplification_threshold=0.0,
            output_single_stem=single_stem,
            sample_rate=self.settings.sample_rate,
            # soundfile keeps float32 in -> float32 out (the pydub writer goes
            # through int16 and wraps on overshoot).
            use_soundfile=True,
            use_autocast=self.device == "cuda",
            mdxc_params={
                "segment_size": 256,
                "override_model_segment_size": False,
                "batch_size": 1,
                # Number of overlapping windows per sample. 2 halves CPU time
                # for a small quality cost; GPUs can afford 4.
                "overlap": 4 if on_gpu else 2,
                "pitch_shift": 0,
            },
            demucs_params={
                "segment_size": "Default",
                # The "shift trick" averages N randomly shifted passes: ~+0.2 dB
                # SDR for N times the compute. Off on CPU.
                "shifts": self._demucs_shifts,
                "overlap": 0.25,
                "segments_enabled": True,
            },
        )
        # audio-separator picks devices by itself (CUDA whenever torch sees one,
        # DirectML only through the long-stale torch-directml). Use ours instead.
        # Its own ONNX session is only a stand-in on the CPU: _share_onnx swaps
        # in the process-wide GPU session right after loading.
        from .accel import CPU_PROVIDERS

        sep.torch_device = torch.device(self.device)
        sep.onnx_execution_provider = list(CPU_PROVIDERS)
        return sep

    def _onnx_runner(self, path: Path):
        """The one session per ONNX model file (accel.OnnxRunner), on the chosen provider."""
        from .accel import OnnxRunner

        with self._onnx_lock:
            runner = self._onnx.get(str(path))
            if runner is None:
                runner = OnnxRunner(path, self.accel.onnx_providers, on_fallback=self._onnx_failed)
                self._onnx[str(path)] = runner
            return runner

    def _onnx_failed(self, exc: BaseException) -> None:
        from .accel import CPU_PROVIDERS

        self.accel.onnx, self.accel.onnx_providers = "cpu", list(CPU_PROVIDERS)
        note = f"The GPU failed while running the voice model ({type(exc).__name__}); it runs on the CPU until the server restarts."
        if note not in self.accel.notes:
            self.accel.notes.append(note)

    def _share_onnx(self, sep, name: str) -> None:
        """Route an ONNX model loaded by audio-separator through our shared session."""
        mi = getattr(sep, "model_instance", None)
        if not name.endswith(".onnx") or mi is None or getattr(mi, "uses_pytorch_inference", True):
            return
        runner = self._onnx_runner(self.settings.models_dir / name)
        mi.model_run = lambda spek: runner(spek.cpu().numpy())

    def load(self) -> None:
        """Download (first run only) and load the default models."""
        self.settings.models_dir.mkdir(parents=True, exist_ok=True)
        _install_tqdm_hooks()
        if self.device == "cpu":
            # Fast mode will use the ONNX voice model: find its best provider first.
            try:
                self._pick_onnx()
            except Exception as exc:  # noqa: BLE001 - measuring is an optimisation, never fatal
                log.warning("could not measure ONNX providers, using the CPU: %s", exc)

        # Only the default (fast-mode) models are loaded eagerly. Quality-mode
        # models load on first use so an 8 GB laptop is not holding ~2 GB of
        # weights it may never need.
        t0 = time.perf_counter()
        sep, name = self._vocal_separator("fast")
        log.info("vocal model ready: %s (%.1fs)", name, time.perf_counter() - t0)

        t0 = time.perf_counter()
        self._music_separator("fast")
        log.info("music model ready: %s (%.1fs)", self.settings.music_model_fast, time.perf_counter() - t0)
        log.info("accelerators: %s", self.accel.as_dict())

    def _vocal_separator(self, mode: str) -> tuple[object, str]:
        requested = self.vocal_model_for(mode)
        with self._lock:
            if requested not in self._vocal:
                # Fallback chain: requested -> BS-RoFormer -> Mel-Band RoFormer.
                candidates = list(dict.fromkeys([requested, self.settings.vocal_model, self.settings.vocal_model_fallback]))
                last_error: Exception | None = None
                for name in candidates:
                    try:
                        t0 = time.perf_counter()
                        sep = self._new_separator(single_stem="Vocals")
                        self._fetch_and_verify(sep, name)
                        sep.load_model(model_filename=name)
                        self._share_onnx(sep, name)
                        self._vocal[requested] = (sep, name)
                        if name != requested:
                            log.warning("using vocal model %s instead of %s", name, requested)
                        log.info("loaded vocal model %s in %.1fs", name, time.perf_counter() - t0)
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_error = exc
                        log.warning("vocal model %s unavailable (%s)", name, exc)
                else:
                    raise RuntimeError(f"No vocal separation model could be loaded: {last_error}")
            return self._vocal[requested]

    def _music_separator(self, mode: str):
        """The quality model (4 x 80 MB) is loaded lazily on first use."""
        with self._lock:
            if mode not in self._music:
                name = self.settings.music_model_quality if mode == "quality" else self.settings.music_model_fast
                sep = self._new_separator()
                self._fetch_and_verify(sep, name)
                sep.load_model(model_filename=name)
                self._music[mode] = sep
            return self._music[mode]

    def _fetch_and_verify(self, sep, name: str) -> None:
        """Download the model's files if needed, then check every pinned file
        before torch reads any of them (app/integrity.py)."""
        from .integrity import PINNED, verify_separator_files

        sep.download_model_files(name)
        verify_separator_files(self.settings.models_dir, [n for n in PINNED if "/" not in n])

    # -- inference ----------------------------------------------------------

    @staticmethod
    def _run(sep, audio_path: Path, work_dir: Path, names: dict[str, str]) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        sep.output_dir = str(work_dir)
        sep.model_instance.output_dir = str(work_dir)
        sep.separate(str(audio_path), custom_output_names=names)

    def separate_voices(self, mix_path: Path, work_dir: Path, mode: str, progress: ProgressFn) -> np.ndarray:
        def run() -> np.ndarray:
            sep, _name = self._vocal_separator(mode)
            _hook.start(progress)
            try:
                self._run(sep, mix_path, work_dir, {"Vocals": "sep-voices"})
            finally:
                _hook.stop()
            voices, _ = audio_io.read_wav(work_dir / "sep-voices.wav")
            return voices

        return self._cpu_fallback("voice separation", run)

    def separate_music(self, music_path: Path, work_dir: Path, mode: str, progress: ProgressFn) -> dict[str, np.ndarray]:
        def run() -> dict[str, np.ndarray]:
            sep = self._music_separator(mode)
            shifts = max(1, self._demucs_shifts)
            bag = 4 if mode == "quality" else 1  # htdemucs_ft is a bag of four fine-tuned models
            all_names = ["Vocals", "Drums", "Bass", "Other", "Guitar", "Piano"]
            _hook.start(progress, expected_passes=bag * shifts)
            try:
                self._run(sep, music_path, work_dir, {n: f"sep-{n.lower()}" for n in all_names})
            finally:
                _hook.stop()
            out: dict[str, np.ndarray] = {}
            for stem in music_stem_names(mode):
                out[stem], _ = audio_io.read_wav(work_dir / f"sep-{stem}.wav")
            return out

        return self._cpu_fallback("instrument separation", run)


    # -- stage 3 and lyrics ---------------------------------------------------

    def speaker_split_available(self) -> bool:
        from .speakers import speaker_split_available

        return speaker_split_available(self.settings)

    def separate_speakers(self, voices_path: Path, voices: np.ndarray, work_dir: Path, progress: ProgressFn) -> tuple[np.ndarray, np.ndarray]:
        from .speakers import separate_speakers

        return separate_speakers(voices_path, voices, work_dir, self.settings, progress, gpu=self.device == "cuda")

    def live_separator(self):
        """Kim Vocal 2 run directly with onnxruntime for streaming (app/live.py).
        Shares the pinned-hash check with the batch models."""
        from .integrity import verify
        from .live import KimVocalLive, kim_model_path

        try:
            self._pick_onnx()  # no-op if load() already did
        except Exception as exc:  # noqa: BLE001
            log.warning("could not measure ONNX providers, using the CPU: %s", exc)
        with self._lock:
            if self._live is None:
                path = kim_model_path(self.settings)
                if not path.is_file():
                    sep = self._new_separator(single_stem="Vocals")
                    sep.download_model_files(path.name)
                verify(path, path.name, self.settings.models_dir)
                self._live = KimVocalLive(self._onnx_runner(path))
            return self._live

    def enhance(self, src: Path, dst: Path, progress: ProgressFn) -> None:
        from .speakers import enhance_voice

        enhance_voice(src, dst, self.settings, progress, gpu=self.device == "cuda")

    def transcribe(self, voices: np.ndarray, progress: ProgressFn) -> dict:
        from .lyrics import WhisperTranscriber

        with self._lock:
            if self._transcriber is None:
                self._transcriber = WhisperTranscriber(self.settings, self.device)
            transcriber = self._transcriber
        result = transcriber.transcribe(voices, self.settings.sample_rate, progress)
        if self.accel.whisper != transcriber.device:
            self.accel.whisper = transcriber.device
            if transcriber.note and transcriber.note not in self.accel.notes:
                self.accel.notes.append(transcriber.note)
        return result


# --------------------------------------------------------------------------- #
# Fake backend (tests / UI development)
# --------------------------------------------------------------------------- #


class FakeBackend:
    """Frequency filters pretending to be source separation.

    This is exactly the approach the real app must NOT use (see NOTES.md) -- it
    exists only so the API, pipeline plumbing and UI can be exercised in
    milliseconds, with stems that are audibly different from one another.
    """

    device = "cpu"

    def __init__(self, settings: Settings):
        from .accel import Accel

        self.settings = settings
        self.accel = Accel(notes=["Fake models (UNMIX_FAKE_MODELS=1): frequency filters, not neural separation."])

    def load(self) -> None:
        return None

    def _tick(self, progress: ProgressFn, steps: int = 5) -> None:
        for i in range(steps):
            if self.settings.fake_delay_s:
                time.sleep(self.settings.fake_delay_s / steps)
            progress((i + 1) / steps)

    def _band(self, x: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
        from scipy.signal import butter, sosfiltfilt

        nyq = self.settings.sample_rate / 2
        if lo and hi:
            sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        elif hi:
            sos = butter(4, hi / nyq, btype="low", output="sos")
        else:
            sos = butter(4, lo / nyq, btype="high", output="sos")
        return sosfiltfilt(sos, x, axis=0).astype(np.float32)

    def models_for(self, mode: str) -> dict[str, str]:
        return {"voices": "fake-bandpass", "music": "fake-filterbank"}

    def separate_voices(self, mix_path: Path, work_dir: Path, mode: str, progress: ProgressFn) -> np.ndarray:
        mix, _ = audio_io.read_wav(mix_path)
        self._tick(progress)
        return self._band(mix, 300.0, 3400.0)

    def separate_music(self, music_path: Path, work_dir: Path, mode: str, progress: ProgressFn) -> dict[str, np.ndarray]:
        music, _ = audio_io.read_wav(music_path)
        self._tick(progress)
        bands = {"drums": (6000.0, None), "bass": (None, 200.0), "guitar": (200.0, 1200.0), "piano": (1200.0, 6000.0)}
        return {name: self._band(music, *bands[name]) for name in music_stem_names(mode)}

    def speaker_split_available(self) -> bool:
        return True

    def separate_speakers(self, voices_path: Path, voices: np.ndarray, work_dir: Path, progress: ProgressFn) -> tuple[np.ndarray, np.ndarray]:
        self._tick(progress)
        low = self._band(voices, None, 1000.0)
        return low, voices - low

    def live_separator(self):
        band = self._band

        def separate(window: np.ndarray) -> np.ndarray:
            if self.settings.fake_delay_s:
                time.sleep(self.settings.fake_delay_s)
            return band(window.T, 300.0, 3400.0).T  # (2, n) "voices"

        return separate

    def enhance(self, src: Path, dst: Path, progress: ProgressFn) -> None:
        data, sr = audio_io.read_wav(src)
        self._tick(progress)
        # A gentle high-pass stands in for real enhancement (48 kHz mono like the real one).
        from scipy.signal import resample_poly

        mono = self._band(data, 90.0, None).mean(axis=1)
        audio_io.write_wav(dst, resample_poly(mono, 160, 147).astype(np.float32)[:, None], 48000)

    def transcribe(self, voices: np.ndarray, progress: ProgressFn) -> dict:
        from .lyrics import FakeTranscriber

        return FakeTranscriber(self.settings).transcribe(voices, self.settings.sample_rate, progress)


def create_backend(settings: Settings) -> SeparationBackend:
    return FakeBackend(settings) if settings.fake_models else AudioSeparatorBackend(settings)
