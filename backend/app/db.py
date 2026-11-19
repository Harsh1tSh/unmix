"""SQLite job store.

One short-lived connection per call keeps this safe to use from the API event
loop thread and the worker thread at the same time (SQLite serialises writes).
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    status        TEXT NOT NULL,           -- queued | running | done | error
    stage         TEXT NOT NULL DEFAULT 'queued',
    progress      REAL NOT NULL DEFAULT 0,
    message       TEXT NOT NULL DEFAULT '',
    error         TEXT,
    mode          TEXT NOT NULL,
    split_voices  INTEGER NOT NULL DEFAULT 0,
    lyrics        INTEGER NOT NULL DEFAULT 0,
    original_name TEXT NOT NULL DEFAULT '',
    duration_s    REAL,
    stems_json    TEXT,
    timings_json  TEXT,
    models_json   TEXT,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
"""

# column -> type/default, for databases created before the column existed
ADDED_COLUMNS = {
    "models_json": "TEXT",
    "lyrics": "INTEGER NOT NULL DEFAULT 0",
}


@dataclass
class Job:
    id: str
    status: str
    stage: str
    progress: float
    message: str
    error: str | None
    mode: str
    split_voices: bool
    lyrics: bool
    original_name: str
    duration_s: float | None
    stems: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        status=row["status"],
        stage=row["stage"],
        progress=row["progress"],
        message=row["message"],
        error=row["error"],
        mode=row["mode"],
        split_voices=bool(row["split_voices"]),
        lyrics=bool(row["lyrics"]),
        original_name=row["original_name"],
        duration_s=row["duration_s"],
        stems=json.loads(row["stems_json"]) if row["stems_json"] else [],
        timings=json.loads(row["timings_json"]) if row["timings_json"] else {},
        models=json.loads(row["models_json"]) if row["models_json"] else {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class JobStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns introduced after a database was first created.

        CREATE TABLE IF NOT EXISTS never alters an existing table, so every
        column added to SCHEMA later must also be listed here.
        """
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        for column, ddl in ADDED_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {ddl}")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create(self, job_id: str, *, mode: str, split_voices: bool, lyrics: bool = False, original_name: str, duration_s: float) -> Job:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (id, status, stage, progress, message, mode, split_voices, lyrics, original_name,"
                " duration_s, created_at, updated_at) VALUES (?, 'queued', 'queued', 0, 'Waiting in queue', ?, ?, ?, ?, ?, ?, ?)",
                (job_id, mode, int(split_voices), int(lyrics), original_name, duration_s, now, now),
            )
        job = self.get(job_id)
        assert job is not None
        return job

    def get(self, job_id: str) -> Job | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "stems" in fields:
            fields["stems_json"] = json.dumps(fields.pop("stems"))
        for key in ("timings", "models"):
            if key in fields:
                fields[f"{key}_json"] = json.dumps(fields.pop(key))
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def delete(self, job_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cur.rowcount > 0

    def ids_older_than(self, cutoff_ts: float) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id FROM jobs WHERE created_at < ?", (cutoff_ts,)).fetchall()
        return [r["id"] for r in rows]

    def count_unfinished(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')").fetchone()[0])

    def unfinished_ids(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id FROM jobs WHERE status IN ('queued', 'running')").fetchall()
        return [r["id"] for r in rows]

    def all_ids(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id FROM jobs").fetchall()
        return [r["id"] for r in rows]
