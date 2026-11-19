"""Model file integrity: SHA-256 pins for every weight file Unmix loads.

Model weights are downloaded from third-party hosts (GitHub releases, Meta's
CDN, Hugging Face). torch already refuses to execute code from them
(weights_only / restricted unpickler), but a swapped file could still change
what the app does with your audio. Every file listed in model_hashes.json is
hashed before it is loaded; a mismatch refuses the model.

Hashing 600 MB takes a few seconds, so results are cached per file by
(size, mtime) in <models_dir>/.verified.json.

Files not in the manifest (e.g. the fallback vocal model, or a model you chose
via UNMIX_* settings) are allowed with a warning, unless UNMIX_STRICT_MODELS=1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("unmix.integrity")

MANIFEST = json.loads((Path(__file__).with_name("model_hashes.json")).read_text(encoding="utf-8"))
PINNED: dict[str, str] = MANIFEST["files"]
WHISPER_SMALL_REVISION: str = MANIFEST["whisper_small_revision"]
CLEARVOICE_SS_REVISION: str = MANIFEST["clearvoice_ss_revision"]
CLEARVOICE_SE_REVISION: str = MANIFEST["clearvoice_se_revision"]
_lock = threading.Lock()


class IntegrityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strict() -> bool:
    return os.environ.get("UNMIX_STRICT_MODELS", "0").strip().lower() in {"1", "true", "yes"}


def verify(path: Path, key: str, models_dir: Path) -> None:
    """Raise IntegrityError unless `path` matches the pinned hash for `key`."""
    expected = PINNED.get(key)
    if expected is None:
        if _strict():
            raise IntegrityError(f"Model file {key} is not in the pinned manifest (UNMIX_STRICT_MODELS=1).")
        log.warning("model file %s is not pinned; loading without a hash check", key)
        return
    st = path.stat()
    stamp = f"{st.st_size}:{st.st_mtime_ns}"
    cache_path = models_dir / ".verified.json"
    with _lock:
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cache = {}
        if cache.get(key) == {"stamp": stamp, "sha256": expected}:
            return
        actual = sha256(path)
        if actual != expected:
            raise IntegrityError(f"Model file {key} does not match its pinned SHA-256 (got {actual[:12]}…, expected {expected[:12]}…). Delete it to re-download.")
        cache[key] = {"stamp": stamp, "sha256": expected}
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        tmp.replace(cache_path)


def verify_separator_files(models_dir: Path, names: list[str]) -> None:
    """Verify audio-separator files that sit flat in the models dir."""
    for name in names:
        p = models_dir / name
        if p.is_file():
            verify(p, name, models_dir)


def verify_whisper_snapshot(snapshot_dir: Path, size: str, models_dir: Path) -> None:
    for p in snapshot_dir.iterdir():
        if p.is_file():
            verify(p, f"whisper/{size}/{p.name}", models_dir)
