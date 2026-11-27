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

import contextlib
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
                try:
                    conn.execute("BEGIN IMMEDIATE")
                except BaseException:  # never leave the lock held: every later write would hang
                    conn.close()
                    lock.release()
                    raise
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
                # an identical bundle counts once, for the hotkey that committed it on chain FIRST (not the
                # first API call: anyone can copy a public commitment string and race the author); its sha256
                # is unknown ("") while the repo is private: then it is checked when the row is claimed
                # the same pinned source (repo@commit) is the same bundle even before its sha is known
                for dup_id, dup_block in c.execute(
                    "SELECT id, commit_block FROM submissions WHERE hotkey != ? AND owner_job = 0 "
                    # an unpaid quote never blocks: a front-runner copying a public commitment string and
                    # never paying must not lock the author out
                    "AND state NOT IN ('cancelled', 'expired', 'failed', 'awaiting_payment') "
                    "AND ((? != '' AND bundle_sha256 = ?) "
                    "OR (? IS NOT NULL AND hf_repo = ? AND hf_revision = ?))",
                    (hotkey, bundle_sha256, bundle_sha256, hf_repo, hf_repo, hf_revision),
                ).fetchall():
                    if commit_block is None or dup_block is None or dup_block <= commit_block:
                        raise QueueError(f"this exact bundle was already submitted ({dup_id}); counts once")
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
                # a winner waiting for an earlier decided crown goes after every other due row (or it would
                # be claimed before the one it waits for, every time)
                "ORDER BY owner_job DESC, (COALESCE(note, '') LIKE '%earlier decided crown%'), enqueued_ts "
                "LIMIT 1",
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

    def precheck(self, hotkey: str, reg_block: int | None, max_waiting: int | None = None) -> None:
        """The cheap refusals of ``create``, before the caller does expensive work (Hugging Face calls, a
        quote): a used registration, an active submission of the hotkey, a full queue."""
        with self._tx() as c:
            if (
                reg_block is not None
                and c.execute(
                    f"SELECT 1 FROM submissions WHERE hotkey = ? AND reg_block = ? AND owner_job = 0 "  # noqa: S608
                    f"AND state IN ({','.join('?' * len(CONSUMED))})",
                    (hotkey, reg_block, *CONSUMED),
                ).fetchone()
            ):
                raise QueueError("this hotkey registration already made its submission; register a new one")
            busy = c.execute(
                f"SELECT id FROM submissions WHERE hotkey = ? AND owner_job = 0 AND state IN "  # noqa: S608
                f"({','.join('?' * len(ACTIVE))})",
                (hotkey, *ACTIVE),
            ).fetchone()
            if busy:
                raise QueueError(f"hotkey already has an active submission {busy[0]}")
            if max_waiting is not None and (
                c.execute("SELECT COUNT(*) FROM submissions WHERE state IN ('queued','paused')").fetchone()[0]
                >= max_waiting
            ):
                raise QueueError("the queue is full; try again later")

    def unknown_earlier(self, row: Row) -> list[Row]:
        """Other hotkeys' live or finished rows committed before this one whose bundle sha is still unknown
        (their repo was private at submission): read them before deciding this row is not a copy."""
        if row.owner_job or row.commit_block is None:
            return []
        with self._tx() as c:
            q = (
                "SELECT id FROM submissions WHERE hotkey != ? AND owner_job = 0 AND bundle_sha256 = '' "
                "AND hf_repo IS NOT NULL AND commit_block IS NOT NULL AND commit_block < ? "
                "AND state NOT IN ('cancelled', 'expired', 'failed')"
            )
            ids = [r[0] for r in c.execute(q, (row.hotkey, row.commit_block))]
        return [r for r in (self.get(i) for i in ids) if r is not None]

    def committed_earlier(self, row: Row, unpaid: bool = True) -> str | None:
        """Another hotkey's submission of the same bundle committed on chain before this row's commitment
        (the author; this row is the copy): its id, else None. A 'failed' author row counts too (its
        evaluation may have failed on our side). An unpaid author counts while its quote is alive
        (``unpaid``): the caller waits for it a bounded time (AUTHOR_WAIT_S), then asks with unpaid=False,
        so a front-runner who never pays cannot hold the author's run for long."""
        if row.owner_job or row.commit_block is None:
            return None
        same = (
            "((? != '' AND s.bundle_sha256 = ?) OR (? IS NOT NULL AND s.hf_repo = ? AND s.hf_revision = ?))"
        )
        args = (row.bundle_sha256, row.bundle_sha256, row.hf_repo, row.hf_repo, row.hf_revision)
        with self._tx() as c:
            r = c.execute(
                "SELECT s.id FROM submissions s WHERE s.hotkey != ? AND s.owner_job = 0 "  # noqa: S608
                "AND s.commit_block IS NOT NULL AND s.commit_block < ? AND " + same + " AND ("
                "s.state NOT IN ('cancelled', 'expired', 'awaiting_payment') "
                "OR (? AND s.state = 'awaiting_payment' AND COALESCE(s.expires_ts, 0) > ?)) "
                "ORDER BY s.commit_block LIMIT 1",
                (row.hotkey, row.commit_block, *args, int(unpaid), self._now()),
            ).fetchone()
        return r[0] if r else None

    def author_unpaid(self, sid: str) -> bool:
        r = self.get(sid)
        return r is not None and r.state == "awaiting_payment"

    def recover(self) -> list[str]:
        """After a restart: a row left 'running' goes back to the queue (its saved outputs, if any, are
        reused; the run itself is not repeated when they exist)."""
        with self._tx() as c:
            ids = [r[0] for r in c.execute("SELECT id FROM submissions WHERE state = 'running'").fetchall()]
            c.execute("UPDATE submissions SET state = 'queued' WHERE state = 'running'")
        return ids

    #: note keys that outlive a later note (a pause rewrites the note; the bounded author wait must not reset)
    STICKY_NOTE = ("author_wait_since", "crown_waits")

    def note(self, sid: str, **info: Any) -> None:
        cur = self.get(sid)
        with contextlib.suppress(Exception):
            old = json.loads(cur.note or "{}") if cur is not None else {}
            info = {**{k: old[k] for k in self.STICKY_NOTE if k in old and k not in info}, **info}
        self.update(sid, note=json.dumps(info, default=str)[:2000])
