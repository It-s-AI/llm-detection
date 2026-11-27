"""Submission queue (SQLite): one row per submission, one worker takes them in payment order.

States::

    awaiting_payment --paid (hold opened)--> queued --claimed--> running --> done | failed
          |                                     ^                   |
          +--> expired (quote ran out)          +---- paused <------+   (judges/detectors down; retry later)
          +--> cancelled (by the miner)

The queue holds what the miner asked for and where it is; scores and records live in the
``SubmissionStore`` (evaluation/pipeline.py), money in ``billing.Accounts``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

ACTIVE = ("awaiting_payment", "queued", "running", "paused")
FINAL = ("done", "failed", "expired", "cancelled")
#: states that use up a registration's one submission (D-66); expired, cancelled and failed (our side,
#: nothing charged) leave it free
CONSUMED = ("queued", "running", "paused", "done")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
  id TEXT PRIMARY KEY,
  hotkey TEXT NOT NULL,
  state TEXT NOT NULL,
  period_id TEXT,
  quote_id TEXT,
  hold_cents INTEGER,
  checkout_session TEXT,
  bundle_url TEXT NOT NULL,
  bundle_sha256 TEXT NOT NULL,
  bundle_bytes INTEGER,
  name TEXT,
  created_ts REAL NOT NULL,
  expires_ts REAL,
  enqueued_ts REAL,
  started_ts REAL,
  finished_ts REAL,
  attempts INTEGER NOT NULL DEFAULT 0,
  pauses INTEGER NOT NULL DEFAULT 0,
  retry_at REAL,
  note TEXT,
  owner_job INTEGER NOT NULL DEFAULT 0,
  reg_block INTEGER,
  hf_repo TEXT,
  hf_revision TEXT,
  commit_block INTEGER,
  secret TEXT,
  archived INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS sub_state ON submissions (state, enqueued_ts);
CREATE INDEX IF NOT EXISTS sub_hotkey ON submissions (hotkey, state);
"""


@dataclass
class Row:
    id: str
    hotkey: str
    state: str
    period_id: str | None
    quote_id: str | None
    hold_cents: int | None
    checkout_session: str | None
    bundle_url: str
    bundle_sha256: str
    bundle_bytes: int | None
    name: str | None
    created_ts: float
    expires_ts: float | None
    enqueued_ts: float | None
    started_ts: float | None
    finished_ts: float | None
    attempts: int
    pauses: int
    retry_at: float | None
    note: str | None
    owner_job: int
    reg_block: int | None = None  # registration block of the hotkey when it submitted
    hf_repo: str | None = None  # the bundle on the miner's Hugging Face repo ...
    hf_revision: str | None = None  # ... at this exact commit (from the on-chain commitment)
    commit_block: int | None = None  # block of that commitment
    secret: str | None = None  # encrypted read token for a private repo; cleared after the evaluation
    archived: int = 0  # the king's bundle is in our archive
    payment_url: str | None = None  # unused since payments moved to TAO (queues made before keep it)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


_COLS = ", ".join(f.name for f in fields(Row))


class QueueError(RuntimeError):
    pass


class SubmissionQueue:
    def __init__(self, path: Path | str, now=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(_SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(submissions)")}
            for col, typ in (("reg_block", "INTEGER"), ("hf_repo", "TEXT"), ("hf_revision", "TEXT"),
                             ("commit_block", "INTEGER"), ("secret", "TEXT"), ("payment_url", "TEXT"),
                             ("archived", "INTEGER NOT NULL DEFAULT 0")):  # fmt: skip
                if col not in cols:  # queues created before these columns
                    c.execute(f"ALTER TABLE submissions ADD COLUMN {col} {typ}")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _tx(self):
        lock, conn = self._lock, self._conn()

        class _Tx:
            def __enter__(self_inner):
                lock.acquire()
                conn.execute("BEGIN IMMEDIATE")
                return conn

            def __exit__(self_inner, et, ev, tb):
                try:
                    conn.execute("ROLLBACK" if et else "COMMIT")
                finally:
                    conn.close()
                    lock.release()
                return False

        return _Tx()

    # --- reads --------------------------------------------------------------------------------------
    def get(self, sid: str) -> Row | None:
        with self._conn() as c:
            r = c.execute(f"SELECT {_COLS} FROM submissions WHERE id = ?", (sid,)).fetchone()  # noqa: S608
        return Row(*r) if r else None

    def by_state(self, *states: str) -> list[Row]:
        q = ",".join("?" * len(states))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT {_COLS} FROM submissions WHERE state IN ({q}) ORDER BY enqueued_ts, created_ts",  # noqa: S608
                states,
            ).fetchall()
        return [Row(*r) for r in rows]

    def active_for(self, hotkey: str) -> list[Row]:
        return [r for r in self.by_state(*ACTIVE) if r.hotkey == hotkey and not r.owner_job]

    def position(self, sid: str) -> int | None:
        """1-based place among submissions waiting to run (queued or paused), owner jobs first."""
        waiting = sorted(self.by_state("queued", "paused"), key=lambda r: (-r.owner_job, r.enqueued_ts or 0))
        for i, r in enumerate(waiting):
            if r.id == sid:
                return i + 1
        return None

    def created_since(self, ts: float, states: tuple[str, ...]) -> int:
        """Miner submissions created at or after ``ts`` that are now in one of ``states``."""
        q = ",".join("?" * len(states))
        with self._conn() as c:
            sql = (
                f"SELECT COUNT(*) FROM submissions WHERE created_ts >= ? AND owner_job = 0 AND state IN ({q})"  # noqa: S608
            )
            return c.execute(sql, (ts, *states)).fetchone()[0]

    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            return dict(c.execute("SELECT state, COUNT(*) FROM submissions GROUP BY state").fetchall())

    # --- writes -------------------------------------------------------------------------------------
    def create(
        self,
        sid: str,
        hotkey: str,
        *,
        bundle_url: str,
        bundle_sha256: str,
        bundle_bytes: int | None,
        name: str = "",
        period_id: str | None = None,
        quote_id: str | None = None,
        hold_cents: int | None = None,
        expires_ts: float | None = None,
        owner_job: bool = False,
        max_waiting: int | None = None,
        reg_block: int | None = None,
        hf_repo: str | None = None,
        hf_revision: str | None = None,
        commit_block: int | None = None,
        secret: str | None = None,
    ) -> Row:
        now = self._now()
        with self._tx() as c:
            if not owner_job and reg_block is not None:
                used = c.execute(
                    f"SELECT id FROM submissions WHERE hotkey = ? AND reg_block = ? AND owner_job = 0 "  # noqa: S608
                    f"AND state IN ({','.join('?' * len(CONSUMED))})",
                    (hotkey, reg_block, *CONSUMED),
                ).fetchone()
                if used:
                    raise QueueError(
                        f"this hotkey registration already made its submission ({used[0]}); "
                        "register a new hotkey to submit again"
                    )
            if not owner_job:
                dup = c.execute(
                    "SELECT id FROM submissions WHERE bundle_sha256 = ? AND hotkey != ? AND owner_job = 0 "
                    "AND state NOT IN ('cancelled', 'expired', 'failed')",
                    (bundle_sha256, hotkey),
                ).fetchone()
                if dup:
                    raise QueueError(f"this exact bundle was already submitted ({dup[0]}); it counts once")
                busy = c.execute(
                    f"SELECT id FROM submissions WHERE hotkey = ? AND owner_job = 0 AND state IN "  # noqa: S608
                    f"({','.join('?' * len(ACTIVE))})",
                    (hotkey, *ACTIVE),
                ).fetchone()
                if busy:
                    raise QueueError(f"hotkey already has an active submission {busy[0]}")
                if max_waiting is not None:  # paid and waiting; an unpaid quote expires on its own
                    n = c.execute(
                        "SELECT COUNT(*) FROM submissions WHERE state IN ('queued','paused')"
                    ).fetchone()[0]
                    if n >= max_waiting:
                        raise QueueError("the queue is full; try again later")
            c.execute(
                "INSERT INTO submissions (id, hotkey, state, period_id, quote_id, hold_cents, bundle_url, "
                "bundle_sha256, bundle_bytes, name, created_ts, expires_ts, owner_job, reg_block, hf_repo, "
                "hf_revision, commit_block, secret) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    hotkey,
                    "awaiting_payment",
                    period_id,
                    quote_id,
                    hold_cents,
                    bundle_url,
                    bundle_sha256,
                    bundle_bytes,
                    name,
                    now,
                    expires_ts,
                    int(owner_job),
                    reg_block,
                    hf_repo,
                    hf_revision,
                    commit_block,
                    secret,
                ),
            )
        row = self.get(sid)
        assert row is not None
        return row

    def update(self, sid: str, *, expect: tuple[str, ...] | None = None, **cols: Any) -> Row:
        """Set columns; with ``expect``, only if the row is in one of those states (else QueueError)."""
        allowed = {f.name for f in fields(Row)} - {"id", "hotkey", "created_ts"}
        bad = set(cols) - allowed
        if bad:
            raise ValueError(f"unknown columns {bad}")
        with self._tx() as c:
            r = c.execute("SELECT state FROM submissions WHERE id = ?", (sid,)).fetchone()
            if r is None:
                raise QueueError(f"unknown submission {sid}")
            if expect is not None and r[0] not in expect:
                raise QueueError(f"submission {sid} is {r[0]}, not {'/'.join(expect)}")
            sets = ", ".join(f"{k} = ?" for k in cols)
            c.execute(f"UPDATE submissions SET {sets} WHERE id = ?", (*cols.values(), sid))  # noqa: S608
        row = self.get(sid)
        assert row is not None
        return row

    def enqueue(self, sid: str) -> Row:
        return self.update(sid, expect=("awaiting_payment",), state="queued", enqueued_ts=self._now())

    def claim(self, max_running: int = 1) -> Row | None:
        """The next submission to run: owner jobs first, then queued and due paused rows in payment order;
        None while ``max_running`` evaluations are already running (each runs on its own pod)."""
        now = self._now()
        with self._tx() as c:
            if (
                c.execute("SELECT COUNT(*) FROM submissions WHERE state = 'running'").fetchone()[0]
                >= max_running
            ):
                return None
            r = c.execute(
                f"SELECT {_COLS} FROM submissions WHERE state = 'queued' "  # noqa: S608
                "OR (state = 'paused' AND (retry_at IS NULL OR retry_at <= ?)) "
                "ORDER BY owner_job DESC, enqueued_ts LIMIT 1",
                (now,),
            ).fetchone()
            if r is None:
                return None
            c.execute(
                "UPDATE submissions SET state = 'running', started_ts = ?, attempts = attempts + 1 "
                "WHERE id = ?",
                (now, r[0]),
            )
        return self.get(r[0])

    def recover(self) -> list[str]:
        """After a restart: a row left 'running' goes back to the queue (its saved outputs, if any, are
        reused; the run itself is not repeated when they exist)."""
        with self._tx() as c:
            ids = [r[0] for r in c.execute("SELECT id FROM submissions WHERE state = 'running'").fetchall()]
            c.execute("UPDATE submissions SET state = 'queued' WHERE state = 'running'")
        return ids

    def note(self, sid: str, **info: Any) -> None:
        self.update(sid, note=json.dumps(info, default=str)[:2000])
