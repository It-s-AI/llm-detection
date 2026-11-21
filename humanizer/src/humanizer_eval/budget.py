"""Local spend ledger with hard per-service caps.

Every client that can spend money (paid detectors, OpenRouter, Lium) must
`reserve()` an upper-bound estimate *before* the request and `commit()` the
actual cost afterwards. A reservation that is never closed (crash) keeps
counting against the cap at its estimate, so a crash can only make us more
conservative, never overspend.

The ledger is a single SQLite file; check-and-reserve runs inside
`BEGIN IMMEDIATE`, so concurrent processes on one machine cannot race past a cap.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class BudgetExceededError(RuntimeError):
    """Raised when a reservation would push a service over its hard cap."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    service TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'committed', 'released')),
    estimate_usd REAL NOT NULL,
    actual_usd REAL,
    ref TEXT,
    meta TEXT
);
CREATE INDEX IF NOT EXISTS ledger_service ON ledger(service, status);
"""


@dataclass
class Reservation:
    ledger: SpendLedger
    id: str
    service: str
    estimate_usd: float
    closed: bool = False

    def commit(self, actual_usd: float, meta: dict | None = None) -> None:
        if self.closed:
            return
        self.ledger._close(self.id, "committed", actual_usd, meta)
        self.closed = True

    def release(self, meta: dict | None = None) -> None:
        """Request failed before any cost was incurred."""
        if self.closed:
            return
        self.ledger._close(self.id, "released", 0.0, meta)
        self.closed = True


class SpendLedger:
    def __init__(self, path: Path | str, caps: dict[str, float]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.caps = dict(caps)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()

    # --- queries -------------------------------------------------------
    def _totals(self, conn: sqlite3.Connection, service: str) -> tuple[float, float]:
        row = conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN status='committed' THEN actual_usd END), 0),"
            "       COALESCE(SUM(CASE WHEN status='open' THEN estimate_usd END), 0)"
            " FROM ledger WHERE service=?",
            (service,),
        ).fetchone()
        return float(row[0]), float(row[1])

    def spent(self, service: str) -> float:
        with self._conn() as c:
            return self._totals(c, service)[0]

    def outstanding(self, service: str) -> float:
        with self._conn() as c:
            return self._totals(c, service)[1]

    def remaining(self, service: str) -> float:
        with self._conn() as c:
            committed, open_ = self._totals(c, service)
        return self.caps.get(service, 0.0) - committed - open_

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        with self._conn() as c:
            services = {r[0] for r in c.execute("SELECT DISTINCT service FROM ledger")} | set(self.caps)
            for s in sorted(services):
                committed, open_ = self._totals(c, s)
                out[s] = {"cap": self.caps.get(s, 0.0), "spent": committed, "reserved": open_}
        return out

    # --- mutations -----------------------------------------------------
    def reserve(self, service: str, estimate_usd: float, ref: str = "") -> Reservation:
        if estimate_usd < 0:
            raise ValueError("estimate must be non-negative")
        if service not in self.caps:
            raise BudgetExceededError(f"no budget configured for service {service!r}")
        rid = uuid.uuid4().hex
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                committed, open_ = self._totals(c, service)
                cap = self.caps[service]
                if committed + open_ + estimate_usd > cap + 1e-9:
                    raise BudgetExceededError(
                        f"{service}: cap ${cap:.2f}, spent ${committed:.2f}, reserved ${open_:.2f},"
                        f" requested ${estimate_usd:.2f}"
                    )
                c.execute(
                    "INSERT INTO ledger (id, ts, service, status, estimate_usd, ref) VALUES (?,?,?,?,?,?)",
                    (rid, time.time(), service, "open", estimate_usd, ref),
                )
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise
        return Reservation(self, rid, service, estimate_usd)

    def _close(self, rid: str, status: str, actual_usd: float, meta: dict | None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE ledger SET status=?, actual_usd=?, meta=? WHERE id=? AND status='open'",
                (status, float(actual_usd), json.dumps(meta or {}, default=str), rid),
            )

    @contextmanager
    def spend(self, service: str, estimate_usd: float, ref: str = ""):
        """Reserve; if the block raises, the reservation is committed at its
        estimate (we cannot know whether the vendor charged us)."""
        res = self.reserve(service, estimate_usd, ref)
        try:
            yield res
        except BaseException:
            if not res.closed:
                res.commit(estimate_usd, {"note": "exception; committed at estimate"})
            raise
        if not res.closed:
            res.commit(estimate_usd, {"note": "not explicitly committed; committed at estimate"})
