"""Download (and cache) every model the pipeline can use.

Usage:  uv run python scripts/download_models.py [--quality]

The server also downloads missing weights on startup; this script just lets you
do it ahead of time and see the progress bars.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", action="store_true", help="also fetch the quality-mode models (BS-RoFormer 610 MB, htdemucs_ft 4 x 80 MB)")
    args = parser.parse_args()

    from audio_separator.separator import Separator

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    from app.separation import detect_device

    cpu_only = detect_device() == "cpu"
    # Mirrors AudioSeparatorBackend.vocal_model_for: CPU fast mode uses the light model.
    wanted = [settings.vocal_model_cpu_fast if cpu_only and settings.vocal_model_cpu_fast else settings.vocal_model, settings.music_model_fast]
    if args.quality:
        wanted += [settings.vocal_model, settings.music_model_quality]
    wanted = list(dict.fromkeys(wanted))

    for name in wanted:
        print(f"==> {name}")
        sep = Separator(log_level=logging.WARNING, model_file_dir=str(settings.models_dir))
        try:
            sep.download_model_files(name)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}")
            if name == settings.vocal_model:
                print(f"    trying fallback {settings.vocal_model_fallback}")
                sep.download_model_files(settings.vocal_model_fallback)
    print("done ->", settings.models_dir)


if __name__ == "__main__":
    main()
