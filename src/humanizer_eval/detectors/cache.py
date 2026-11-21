"""SQLite cache of detector responses keyed by (detector, config, sha256(text))."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from humanizer_eval.detectors.base import DetectorScore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detector_cache (
    detector TEXT NOT NULL,
    config TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    version TEXT,
    score REAL NOT NULL,
    candidates TEXT NOT NULL,
    raw TEXT NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (detector, config, text_sha256)
);
"""


class DetectorCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, detector: str, config: str, sha: str, version: str | None = None) -> DetectorScore | None:
        """Return a cached score; if `version` is given, only an entry with that exact version."""
        with self._conn() as c:
            row = c.execute(
                "SELECT version, score, candidates, raw FROM detector_cache"
                " WHERE detector=? AND config=? AND text_sha256=?",
                (detector, config, sha),
            ).fetchone()
        if row is None or (version is not None and row[0] != version):
            return None
        return DetectorScore(
            detector=detector,
            text_sha256=sha,
            score=float(row[1]),
            candidates=json.loads(row[2]),
            version=row[0],
            raw=json.loads(row[3]),
            cached=True,
        )

    def put(self, config: str, s: DetectorScore) -> None:
        if not s.ok:
            return
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO detector_cache VALUES (?,?,?,?,?,?,?,?)",
                (
                    s.detector,
                    config,
                    s.text_sha256,
                    s.version,
                    s.score,
                    json.dumps(s.candidates),
                    json.dumps(s.raw, default=str),
                    time.time(),
                ),
            )

    def invalidate(self, detector: str, keep_version: str | None = None) -> int:
        """Drop entries of a detector (except those of `keep_version`). Returns rows removed."""
        with self._conn() as c:
            if keep_version is None:
                cur = c.execute("DELETE FROM detector_cache WHERE detector=?", (detector,))
            else:
                cur = c.execute(
                    "DELETE FROM detector_cache WHERE detector=? AND (version IS NULL OR version!=?)",
                    (detector, keep_version),
                )
            return cur.rowcount
