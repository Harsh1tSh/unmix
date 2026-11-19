"""Download the free MUSDB18 7-second preview set (~140 MB, 144 tracks).

Usage:  uv run --group eval python eval/fetch_musdb_sample.py

This is NOT MUSDB18-HQ: it is the official lossy (AAC stems) 7 s excerpt of
every track, published by the dataset authors for quick experiments. Use it to
smoke-test eval/run_eval.py; download the full MUSDB18-HQ from
https://zenodo.org/records/3338373 for numbers comparable with papers.
"""

from __future__ import annotations

from pathlib import Path

import musdb

ROOT = Path(__file__).resolve().parent / "data" / "musdb18-7s"

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    db = musdb.DB(root=str(ROOT), download=True, subsets="test")
    print(f"{len(db)} test tracks under {ROOT}")
    for track in list(db)[:5]:
        print("  ", track.name, f"{track.duration:.1f}s")
