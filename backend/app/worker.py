"""Background job runner.

One daemon thread pulls job ids off an in-process queue and runs them strictly
one at a time (a single laptop CPU/GPU cannot usefully run two separations at
once). Job state is persisted in SQLite; live progress is also fanned out to
any SSE subscribers.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import shutil
import threading
import time
from pathlib import Path

from . import audio_io
from .config import Settings
from .db import JobStore
from .pipeline import PipelineContext, run_pipeline
from .security import public_error
from .separation import SeparationBackend

log = logging.getLogger("unmix.worker")

_STOP = object()


class JobManager:
    def __init__(self, settings: Settings, store: JobStore, backend: SeparationBackend):
        self.settings = settings
        self.store = store
        self.backend = backend

        self.models_ready = False
        self.models_error: str | None = None

        self._queue: queue.Queue = queue.Queue()
        self._subscribers: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}
        self._sub_lock = threading.Lock()
        self._cancelled: set[str] = set()
        self._threads: list[threading.Thread] = []
        # Voice clean-up tasks, keyed (job_id, stem): {"status", "progress", "error"}.
        # Finished results are files, so "done" survives restarts; the rest is in memory.
        self._cleans: dict[tuple[str, str], dict] = {}
        self._cleans_lock = threading.Lock()
        # A live session needs the whole CPU: the queue waits while one is active.
        self.live_active = threading.Event()
        self._stopping = threading.Event()

    # ------------------------------------------------------------------ life

    def start(self) -> None:
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        # Anything still marked queued/running belongs to a previous process.
        for job_id in self.store.unfinished_ids():
            self.store.update(job_id, status="error", stage="error", error="The server restarted while this job was in progress. Please upload again.")
        self.cleanup_expired()
        for target, name in ((self._load_models, "unmix-models"), (self._work_loop, "unmix-worker"), (self._cleanup_loop, "unmix-cleanup")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stopping.set()
        self._queue.put(_STOP)

    def _load_models(self) -> None:
        t0 = time.perf_counter()
        try:
            self.backend.load()
            # Pay the slow first imports now rather than inside the first job.
            import librosa  # noqa: F401
            import pyloudnorm  # noqa: F401
            import scipy.signal  # noqa: F401

            self.models_ready = True
            log.info("models ready on %s in %.1fs", self.backend.device, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001
            # Shown in /api/health and the UI: keep paths and internals out of it.
            self.models_error, _ = public_error("Model loading failed", exc)

    # ----------------------------------------------------------------- queue

    def job_dir(self, job_id: str) -> Path:
        return self.settings.jobs_dir / job_id

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def cancel(self, job_id: str) -> None:
        """Best effort: a queued job is skipped; a running one finishes its
        current model call and is then discarded."""
        self._cancelled.add(job_id)

    def _work_loop(self) -> None:
        while not self._stopping.is_set():
            item = self._queue.get()
            if item is _STOP:
                return
            while self.live_active.is_set() and not self._stopping.is_set():
                time.sleep(0.5)  # finish the live session first (jobs keep their place)
            try:
                if isinstance(item, tuple):  # ("clean", job_id, stem)
                    self._run_clean(item[1], item[2])
                else:
                    self._run_job(item)
            except Exception:  # noqa: BLE001
                log.exception("unexpected failure running %s", item)

    # --------------------------------------------------------- voice clean-up

    def clean_path(self, job_id: str, stem: str) -> Path:
        return self.job_dir(job_id) / "clean" / f"{stem}.wav"

    def clean_status(self, job_id: str, stem: str) -> dict | None:
        if self.clean_path(job_id, stem).is_file():
            return {"status": "done", "progress": 1.0}
        with self._cleans_lock:
            state = self._cleans.get((job_id, stem))
            return dict(state) if state else None

    def pending_cleans(self) -> int:
        with self._cleans_lock:
            return sum(1 for s in self._cleans.values() if s["status"] in ("queued", "running"))

    def request_clean(self, job_id: str, stem: str) -> dict:
        current = self.clean_status(job_id, stem)
        if current and current["status"] in ("queued", "running", "done"):
            return current
        with self._cleans_lock:
            self._cleans[(job_id, stem)] = {"status": "queued", "progress": 0.0}
        self._queue.put(("clean", job_id, stem))
        return {"status": "queued", "progress": 0.0}

    def _run_clean(self, job_id: str, stem: str) -> None:
        key = (job_id, stem)

        def set_state(**kw) -> None:
            with self._cleans_lock:
                if key in self._cleans:
                    self._cleans[key].update(kw)

        src = self.job_dir(job_id) / "stems" / f"{stem}.wav"
        if self.store.get(job_id) is None or not src.is_file():
            with self._cleans_lock:
                self._cleans.pop(key, None)
            return
        set_state(status="running")
        dst = self.clean_path(job_id, stem)
        t0 = time.perf_counter()
        try:
            self.backend.enhance(src, dst, lambda p: set_state(progress=round(float(p), 3)))
        except Exception as exc:  # noqa: BLE001
            message, _ = public_error("Voice clean-up failed", exc)
            set_state(status="error", error=message)
            return
        if self.store.get(job_id) is None:  # deleted meanwhile
            shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        with self._cleans_lock:
            self._cleans.pop(key, None)
        log.info("job %s | clean %s | %.2fs", job_id, stem, time.perf_counter() - t0)

    def _wait_for_models(self, job_id: str) -> None:
        announced = False
        while not self.models_ready and not self._stopping.is_set():
            if self.models_error:
                raise RuntimeError(f"Models failed to load: {self.models_error}")
            if not announced:
                self._report(job_id, "queued", 0.0, "Loading models (first run downloads them)", status="queued")
                announced = True
            time.sleep(0.25)

    def _run_job(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None or job_id in self._cancelled:
            self._cancelled.discard(job_id)
            return
        last = {"stage": "", "progress": -1.0, "message": ""}

        def report(stage: str, progress: float, message: str) -> None:
            if self._stopping.is_set():
                raise _Shutdown()
            if job_id in self._cancelled:
                raise _Cancelled()
            # Skip no-op updates so chatty model loops do not hammer SQLite.
            if stage == last["stage"] and message == last["message"] and abs(progress - last["progress"]) < 0.002:
                return
            last.update(stage=stage, progress=progress, message=message)
            self._report(job_id, stage, progress, message, status="running")

        try:
            self._wait_for_models(job_id)
            ctx = PipelineContext(
                job_id=job_id,
                job_dir=self.job_dir(job_id),
                mode=job.mode,
                split_voices=job.split_voices,
                lyrics=job.lyrics,
                settings=self.settings,
                backend=self.backend,
            )
            timings = run_pipeline(ctx, report)
            if job_id in self._cancelled:
                raise _Cancelled()
            self.store.update(job_id, status="done", stage="done", progress=1.0, message="Done", stems=ctx.stems, timings=timings, models=ctx.info.get("models", {}), duration_s=ctx.duration_s)
            self._publish(job_id, {"status": "done", "stage": "done", "progress": 1.0, "message": "Done"})
        except _Shutdown:
            log.info("job %s abandoned: server is shutting down", job_id)
        except _Cancelled:
            log.info("job %s cancelled", job_id)
            self._cancelled.discard(job_id)
            shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        except audio_io.AudioError as exc:
            self._fail(job_id, str(exc))
        except Exception as exc:  # noqa: BLE001
            message, _ = public_error(f"Job {job_id[:8]} failed", exc)
            self._fail(job_id, message)

    def _fail(self, job_id: str, message: str) -> None:
        self.store.update(job_id, status="error", stage="error", error=message, message=message)
        self._publish(job_id, {"status": "error", "stage": "error", "progress": 0.0, "message": message, "error": message})
        shutil.rmtree(self.job_dir(job_id) / "work", ignore_errors=True)

    def _report(self, job_id: str, stage: str, progress: float, message: str, *, status: str) -> None:
        progress = round(float(progress), 4)
        self.store.update(job_id, status=status, stage=stage, progress=progress, message=message)
        self._publish(job_id, {"status": status, "stage": stage, "progress": progress, "message": message})

    # ------------------------------------------------------------------- SSE

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        with self._sub_lock:
            self._subscribers.setdefault(job_id, set()).add((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        with self._sub_lock:
            subs = self._subscribers.get(job_id)
            if subs:
                subs.difference_update({s for s in subs if s[1] is q})
                if not subs:
                    self._subscribers.pop(job_id, None)

    def _publish(self, job_id: str, event: dict) -> None:
        with self._sub_lock:
            subs = list(self._subscribers.get(job_id, ()))
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(_put_drop_oldest, q, event)
            except RuntimeError:  # loop already closed
                pass

    # --------------------------------------------------------------- cleanup

    def delete_job(self, job_id: str) -> bool:
        job = self.store.get(job_id)
        if job is None:
            return False
        if job.status in ("queued", "running"):
            self.cancel(job_id)
        self.store.delete(job_id)
        if job.status != "running":  # a running job removes its own files when it notices the cancel
            shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        self._publish(job_id, {"status": "error", "stage": "error", "progress": 0.0, "message": "Job deleted", "error": "Job deleted"})
        return True

    def cleanup_expired(self) -> int:
        cutoff = time.time() - self.settings.job_ttl_hours * 3600
        removed = 0
        for job_id in self.store.ids_older_than(cutoff):
            self.store.delete(job_id)
            shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
            removed += 1
        # Orphaned folders (no DB row) older than the TTL.
        known = set(self.store.all_ids())
        if self.settings.jobs_dir.exists():
            for d in self.settings.jobs_dir.iterdir():
                if d.is_dir() and d.name not in known and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
        if removed:
            log.info("cleanup removed %d expired job(s)", removed)
        return removed

    def _cleanup_loop(self) -> None:
        while not self._stopping.wait(timeout=1800):
            try:
                self.cleanup_expired()
            except Exception:  # noqa: BLE001
                log.exception("cleanup failed")


class _Cancelled(Exception):
    pass


class _Shutdown(Exception):
    pass


def _put_drop_oldest(q: asyncio.Queue, event: dict) -> None:
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(event)
