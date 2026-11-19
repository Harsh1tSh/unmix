"""FastAPI application. Contract: /shared/api-contract.md"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from . import audio_io, schemas
from .config import ALLOWED_EXTENSIONS, EXTRA_AUDIO, STEMS, Settings, settings as default_settings
from .db import Job, JobStore
from .security import GuardMiddleware, public_error
from .pipeline import planned_stages
from .separation import SeparationBackend, create_backend
from .worker import JobManager

VERSION = "0.4.0"
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SSE_KEEPALIVE_S = 15.0

log = logging.getLogger("unmix.api")


def _configure_logging() -> None:
    if not logging.getLogger("unmix").handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S"))
        root = logging.getLogger("unmix")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _safe_filename(raw: str | None) -> str:
    """The client's file name, reduced to a display name: no directories (either
    separator style, whatever OS the server runs on), no control characters,
    at most 120 characters."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = _CONTROL_RE.sub("", name).strip() or "upload"
    if len(name) > 120:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: 110] + "…" + dot + ext) if dot and len(ext) <= 8 else name[:120]
    return name


def _stem_view(job_id: str, name: str) -> schemas.Stem:
    meta = STEMS[name]
    return schemas.Stem(name=name, label=meta.label, parent=meta.parent, color=meta.color, url=f"/api/jobs/{job_id}/stems/{name}.wav")


def _job_view(job: Job) -> schemas.JobView:
    done = job.status == "done"
    return schemas.JobView(
        id=job.id,
        status=job.status,  # type: ignore[arg-type]
        stage=job.stage,
        progress=job.progress,
        message=job.message,
        mode=job.mode,  # type: ignore[arg-type]
        filename=job.original_name,
        split_voices=job.split_voices,
        lyrics=job.lyrics,
        stages=[schemas.StageInfo(**st) for st in planned_stages(job.split_voices, job.lyrics)],
        duration_s=job.duration_s,
        error=job.error,
        stems=[_stem_view(job.id, n) for n in job.stems] if done else None,
        timings=job.timings if done else None,
        models=(job.models or None) if done else None,
        lyrics_url=f"/api/jobs/{job.id}/lyrics" if done and job.lyrics else None,
    )


def create_app(settings: Settings | None = None, backend: SeparationBackend | None = None) -> FastAPI:
    """App factory. Tests pass their own settings (temp data dir) and a fake backend."""
    settings = settings or default_settings
    _configure_logging()

    store = JobStore(settings.db_path)
    manager = JobManager(settings, store, backend or create_backend(settings))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        manager.start()
        yield
        manager.stop()

    docs = {} if settings.api_docs else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title="Unmix API", version=VERSION, lifespan=lifespan, **docs)
    app.state.manager = manager
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        # Vite picks another port when 5173 is busy; any localhost port is fine for a local tool.
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Range"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Content-Disposition"],
    )

    from .security import MIN_ACCESS_KEY_LENGTH

    if settings.remote_key and len(settings.remote_key) < MIN_ACCESS_KEY_LENGTH:
        # Fail at startup, not at the first request (middleware is built lazily).
        raise ValueError(f"UNMIX_REMOTE_KEY must be at least {MIN_ACCESS_KEY_LENGTH} characters.")

    # Outermost: host/origin checks, body cap and security headers apply to everything.
    # Multipart overhead on top of the file itself is small; 1 MB of slack covers it.
    app.add_middleware(
        GuardMiddleware,
        allowed_hosts=settings.allowed_hosts,
        allowed_origins=settings.cors_origins,
        max_body_bytes=settings.max_upload_bytes + 1024 * 1024,
        access_key=settings.remote_key,
    )

    # Your recordings and their transcripts: never leave copies in the browser's disk cache.
    PRIVATE = {"Cache-Control": "no-store"}

    api = APIRouter(prefix="/api")

    def get_job(job_id: str) -> Job:
        job = store.get(job_id) if JOB_ID_RE.match(job_id) else None
        if job is None:
            raise HTTPException(404, "Job not found.")
        return job

    def get_done_job(job_id: str) -> Job:
        job = get_job(job_id)
        if job.status == "error":
            raise HTTPException(409, f"This job failed: {job.error}")
        if job.status != "done":
            raise HTTPException(409, "This job has not finished yet.")
        return job

    # ---------------------------------------------------------------- health

    @api.get("/health", response_model=schemas.Health)
    def health() -> schemas.Health:
        status = "error" if manager.models_error else ("ok" if manager.models_ready else "loading")
        from .accel import Accel

        backend_accel = getattr(manager.backend, "accel", None) or Accel(torch_device=manager.backend.device)
        accel = backend_accel.as_dict()
        return schemas.Health(
            status=status,
            device=manager.backend.device,  # type: ignore[arg-type]
            accelerator=schemas.Accelerator(**accel),
            remote=bool(settings.remote_key),
            models_ready=manager.models_ready,
            models_error=manager.models_error,
            version=VERSION,
            features=schemas.Features(speaker_split=manager.backend.speaker_split_available(), lyrics=True),
        )

    # ------------------------------------------------------------------ jobs

    @api.post("/jobs", response_model=schemas.JobCreated, status_code=201)
    async def create_job(
        file: UploadFile = File(...),
        mode: schemas.Mode = Form("fast"),
        split_voices: bool = Form(False),
        lyrics: bool = Form(True),
    ) -> schemas.JobCreated:
        filename = _safe_filename(file.filename)
        ext = Path(filename).suffix.lower()
        ctype = (file.content_type or "").split(";")[0].strip().lower()
        looks_like_audio = ext in ALLOWED_EXTENSIONS or ctype.startswith("audio/") or ctype in {"video/webm", "video/mp4"}
        if not looks_like_audio:
            raise HTTPException(415, f"Unsupported file type '{ext or ctype or 'unknown'}'. Upload audio: wav, mp3, m4a, webm, ogg or flac.")
        if split_voices and not manager.backend.speaker_split_available():
            raise HTTPException(
                422, "Splitting voices is not installed on this server. Run `uv sync` in backend/speakers (README: 'Voice splitting'), or upload without it."
            )
        if manager.models_error:
            raise HTTPException(503, f"The separation models failed to load: {manager.models_error}")
        if store.count_unfinished() >= settings.max_waiting_jobs:
            raise HTTPException(429, f"{settings.max_waiting_jobs} jobs are already waiting. Try again when one has finished.")

        job_id = uuid.uuid4().hex
        job_dir = manager.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        upload_path = job_dir / f"upload{ext if ext in ALLOWED_EXTENSIONS else '.bin'}"
        try:
            size = 0
            with upload_path.open("wb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, f"File is too large (limit {settings.max_upload_bytes // (1024 * 1024)} MB).")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "The uploaded file is empty.")

            decoded = job_dir / "decoded.wav"
            try:
                await run_in_threadpool(audio_io.decode_to_wav, upload_path, decoded, sample_rate=settings.sample_rate, max_seconds=settings.max_duration_s)
                duration = audio_io.wav_duration(decoded)
            except audio_io.AudioError as exc:
                raise HTTPException(422, str(exc)) from exc
            if duration > settings.max_duration_s:
                raise HTTPException(422, f"Audio is too long (limit {settings.max_duration_s / 60:g} minutes).")
            if duration < settings.min_duration_s:
                raise HTTPException(422, f"Audio is too short ({duration:.2f} s; minimum {settings.min_duration_s:g} s).")
            upload_path.unlink(missing_ok=True)
            store.create(job_id, mode=mode, split_voices=split_voices, lyrics=lyrics, original_name=filename, duration_s=duration)
        except BaseException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

        manager.enqueue(job_id)
        log.info("job %s queued | %s | %.1fs | mode=%s split_voices=%s lyrics=%s", job_id, filename, duration, mode, split_voices, lyrics)
        return schemas.JobCreated(job_id=job_id)

    @api.get("/jobs/{job_id}", response_model=schemas.JobView, response_model_exclude_none=True)
    def read_job(job_id: str) -> schemas.JobView:
        return _job_view(get_job(job_id))

    @api.delete("/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> Response:
        get_job(job_id)
        manager.delete_job(job_id)
        return Response(status_code=204)

    @api.get("/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        get_job(job_id)

        def snapshot() -> dict | None:
            job = store.get(job_id)
            if job is None:
                return None
            event = schemas.JobEvent(status=job.status, stage=job.stage, progress=job.progress, message=job.message, error=job.error)  # type: ignore[arg-type]
            return event.model_dump(exclude_none=True)

        async def stream() -> AsyncIterator[str]:
            # Subscribe before reading state so no update can fall in between.
            q = manager.subscribe(job_id)
            try:
                event = snapshot()
                last_sent: dict | None = None
                while event is not None:
                    if event != last_sent:
                        yield f"data: {json.dumps(event)}\n\n"
                        last_sent = event
                    if event["status"] in ("done", "error"):
                        return
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=SSE_KEEPALIVE_S)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            return
                        # A comment line keeps proxies/browsers from timing out during
                        # long model calls; re-reading state self-heals a missed update.
                        yield ": keepalive\n\n"
                        event = snapshot()
            finally:
                manager.unsubscribe(job_id, q)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
        )

    # ----------------------------------------------------------------- files

    @api.get("/jobs/{job_id}/stems/{filename}")
    def read_stem(job_id: str, filename: str, download: bool = False) -> FileResponse:
        job = get_done_job(job_id)
        name, dot, ext = filename.rpartition(".")
        if not dot or ext != "wav" or (name not in job.stems and name not in EXTRA_AUDIO):
            raise HTTPException(404, f"Unknown stem '{filename}'. Available: {', '.join(job.stems)}.")
        path = manager.job_dir(job_id) / ("mix.wav" if name == "mix" else f"stems/{name}.wav")
        if not path.is_file():
            raise HTTPException(404, "Stem file is missing (it may have expired).")
        # FileResponse implements HTTP range requests (206 + Content-Range).
        nice = f"{Path(job.original_name).stem or 'unmix'} - {name}.wav"
        if settings.stem_transport == "flac" and not download:
            # Remote server: the player gets lossless FLAC (half the bytes);
            # the URL keeps its .wav name so clients need not care.
            return FileResponse(audio_io.flac_copy(path), media_type="audio/flac", headers=PRIVATE)
        return FileResponse(path, media_type="audio/wav", filename=nice if download else None, headers=PRIVATE)

    @api.get("/jobs/{job_id}/spectrograms/{filename}")
    def read_spectrogram(job_id: str, filename: str) -> FileResponse:
        job = get_done_job(job_id)
        name, dot, ext = filename.rpartition(".")
        if not dot or ext != "png" or name not in job.stems:
            raise HTTPException(404, f"Unknown spectrogram '{filename}'.")
        path = manager.job_dir(job_id) / "spectrograms" / f"{name}.png"
        if not path.is_file():
            raise HTTPException(404, "Spectrogram is missing (it may have expired).")
        return FileResponse(path, media_type="image/png", headers=PRIVATE)

    @api.get("/jobs/{job_id}/analysis", response_model=schemas.Analysis)
    def read_analysis(job_id: str) -> FileResponse:
        get_done_job(job_id)
        path = manager.job_dir(job_id) / "analysis.json"
        if not path.is_file():
            raise HTTPException(404, "Analysis is missing (it may have expired).")
        # Served straight from disk: the payload can be a few MB and was validated when written.
        return FileResponse(path, media_type="application/json", headers=PRIVATE)

    @api.get("/jobs/{job_id}/lyrics", response_model=schemas.Lyrics)
    def read_lyrics(job_id: str) -> FileResponse:
        job = get_done_job(job_id)
        path = manager.job_dir(job_id) / "lyrics.json"
        if not job.lyrics:
            raise HTTPException(404, "Lyrics were not requested for this job.")
        if not path.is_file():
            raise HTTPException(404, "Lyrics are missing (they may have expired).")
        return FileResponse(path, media_type="application/json", headers=PRIVATE)

    # --------------------------------------------------------- voice clean-up

    CLEANABLE = {"voices", "speaker_1", "speaker_2"}

    def clean_view(job_id: str, stem: str, state: dict) -> schemas.CleanState:
        url = f"/api/jobs/{job_id}/clean/{stem}.wav" if state["status"] == "done" else None
        return schemas.CleanState(status=state["status"], progress=state.get("progress", 0.0), error=state.get("error"), url=url)

    def cleanable(job_id: str, stem: str) -> Job:
        job = get_done_job(job_id)
        if stem not in CLEANABLE or stem not in job.stems:
            raise HTTPException(404, "Only voice stems can be cleaned up.")
        return job

    @api.post("/jobs/{job_id}/clean/{stem}", response_model=schemas.CleanState, response_model_exclude_none=True, status_code=202)
    def request_clean(job_id: str, stem: str) -> schemas.CleanState:
        cleanable(job_id, stem)
        if not manager.backend.speaker_split_available():
            raise HTTPException(422, "Voice clean-up is not installed on this server. Run `uv sync` in backend/speakers (README: 'Voice splitting').")
        current = manager.clean_status(job_id, stem)
        if not (current and current["status"] in ("queued", "running", "done")) and manager.pending_cleans() >= settings.max_waiting_jobs:
            raise HTTPException(429, "Several clean-ups are already waiting. Try again when one has finished.")
        return clean_view(job_id, stem, manager.request_clean(job_id, stem))

    @api.get("/jobs/{job_id}/clean/{name}")
    def read_clean(job_id: str, name: str):
        if name.endswith(".wav"):
            stem = name[: -len(".wav")]
            job = cleanable(job_id, stem)
            path = manager.clean_path(job_id, stem)
            if not path.is_file():
                raise HTTPException(404, "This voice has not been cleaned up yet.")
            nice = f"{Path(job.original_name).stem or 'unmix'} - {stem} (clean).wav"
            return FileResponse(path, media_type="audio/wav", filename=nice, headers=PRIVATE)
        cleanable(job_id, name)
        state = manager.clean_status(job_id, name)
        if state is None:
            raise HTTPException(404, "No clean-up was requested for this voice.")
        return clean_view(job_id, name, state).model_dump(exclude_none=True)

    # ------------------------------------------------- browser-only mode model

    BROWSER_MODELS = {"Kim_Vocal_2.onnx"}  # allowlist: nothing else in the models dir is served

    @api.get("/models/{name}")
    def read_model(name: str) -> FileResponse:
        """The ONNX model the browser-only mode runs with onnxruntime-web.
        Verified against its pinned SHA-256 before it is served; cacheable
        forever, since the pinned file never changes."""
        from .integrity import PINNED, IntegrityError, verify

        if name not in BROWSER_MODELS:
            raise HTTPException(404, "Unknown model.")
        path = settings.models_dir / name
        if not path.is_file():
            raise HTTPException(404, "The model is not downloaded yet. Run one normal job first (or scripts/download_models.py).")
        try:
            verify(path, name, settings.models_dir)
        except IntegrityError as exc:
            message, _ = public_error("Model file failed its integrity check", exc)
            raise HTTPException(500, message) from exc
        return FileResponse(
            path,
            media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=31536000, immutable", "ETag": f'"{PINNED.get(name, "")[:32]}"'},
        )

    # ------------------------------------------------------------ live mode

    live_lock = asyncio.Lock()

    @api.websocket("/live")
    async def live(ws: WebSocket) -> None:
        """Near-real-time voices/music split. Protocol (see api-contract.md):
        client: {"type":"start","sample_rate":44100,"channels":1|2} then binary
        interleaved float32 frames, then {"type":"stop"}.
        server: {"type":"ready"}, then per block a JSON header + binary payload,
        {"type":"behind",...} when skipping ahead, {"type":"end"}."""
        from . import live as L

        await ws.accept()
        if live_lock.locked():
            await ws.send_json({"type": "error", "message": "Another live session is running."})
            await ws.close(code=1013)
            return
        async with live_lock:
            manager.live_active.set()
            try:
                await _live_session(ws, L)
            except WebSocketDisconnect:
                pass
            except Exception as exc:  # noqa: BLE001
                message, _ = public_error("Live session failed", exc)
                try:
                    await ws.send_json({"type": "error", "message": message})
                    await ws.close(code=1011)
                except Exception:  # noqa: BLE001
                    pass
            finally:
                manager.live_active.clear()

    async def _live_session(ws: WebSocket, L) -> None:
        start = await asyncio.wait_for(ws.receive_json(), timeout=10)
        if not isinstance(start, dict) or start.get("type") != "start" or start.get("sample_rate") != L.SR or start.get("channels") not in (1, 2):
            await ws.send_json({"type": "error", "message": f"Send {{type: 'start', sample_rate: {L.SR}, channels: 1 or 2}} first."})
            await ws.close(code=1003)
            return
        channels = int(start["channels"])
        if not manager.models_ready:
            await ws.send_json({"type": "error", "message": "Models are still loading. Try again in a moment."})
            await ws.close(code=1013)
            return
        separate = await run_in_threadpool(manager.backend.live_separator)
        splitter = L.StreamingSplitter(separate)
        await ws.send_json({"type": "ready", "sample_rate": L.SR, "block_frames": L.GEN, "context_frames": L.TRIM})

        queue: asyncio.Queue = asyncio.Queue()
        index = 0
        started = L.now()

        async def compute() -> None:
            nonlocal index
            final = False
            while True:
                final = await queue.get() or final
                while splitter.ready_blocks() or final:
                    backlog = splitter.ready_blocks()
                    if backlog > L.MAX_BACKLOG_BLOCKS:
                        # Hopelessly behind (CPU too slow or busy): skip ahead rather than let the delay grow.
                        skipped = splitter.skip_ahead((backlog - 1) * L.GEN)
                        await ws.send_json({"type": "behind", "skipped_frames": skipped})
                    t0 = L.now()
                    out = await run_in_threadpool(splitter.next_block, final)
                    if out is None:
                        break
                    header, payload = L.encode_block(index, out[0], out[1], out[2], L.now() - t0, splitter.ready_blocks())
                    await ws.send_text(header)
                    await ws.send_bytes(payload)
                    index += 1
                if final:
                    await ws.send_json({"type": "end", "frames": splitter.emitted})
                    return

        worker = asyncio.create_task(compute())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    data = msg["bytes"]
                    if len(data) > L.MAX_MESSAGE_BYTES:
                        await ws.send_json({"type": "error", "message": "Audio frame too large."})
                        break
                    try:
                        splitter.push(L.decode_frames(data, channels))
                    except ValueError:
                        await ws.send_json({"type": "error", "message": "Malformed audio frame."})
                        break
                    if splitter.received > L.MAX_SESSION_S * L.SR:
                        await ws.send_json({"type": "error", "message": f"Live sessions are limited to {L.MAX_SESSION_S // 60} minutes."})
                        queue.put_nowait(True)
                        break
                    queue.put_nowait(False)
                elif msg.get("text") is not None:
                    try:
                        text = json.loads(msg["text"])
                    except ValueError:
                        text = {}
                    if text.get("type") == "stop":
                        queue.put_nowait(True)
                        await worker
                        await ws.close()
                        return
            queue.put_nowait(True)
        finally:
            if not worker.done():
                worker.cancel()
        log.info("live session ended after %.0fs (%d frames in)", L.now() - started, splitter.received)

    app.include_router(api)
    return app
