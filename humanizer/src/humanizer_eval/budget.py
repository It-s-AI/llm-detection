"""Local spend ledger with hard per-service caps.

Every client that can spend money (paid detectors, OpenRouter, Lium) must
`reserve()` an upper-bound estimate *before* the request and `commit()` the
actual cost afterwards. A reservation that is never closed (crash) keeps
counting against the cap at its estimate, so a crash can only make us more
conservative, never overspend.

The ledger is a single SQLite file; check-and-reserve runs inside
`BEGIN IMMEDIATE`, so concurrent processes on one machine cannot race past a cap.

In production a second, dynamic limit (``PoolLimit``) covers all paid services together: since the
launch, what we spent and reserved may not exceed what miners paid in (net of refunds) plus a gap for the
owner's own runs plus a weekly allowance for every period begun since the launch (building a period: the
human reference, the king's run). More submissions bring more money and so more room; a bug that loops on
paid calls is stopped after the gap.
"""

from __future__ import annotations

import json
import math
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


@dataclass(frozen=True)
class PoolLimit:
    billing_db: Path  # intake billing.sqlite (payments of the miners)
    since_ts: float
    gap_usd: float
    services: tuple[str, ...] = (
        "pangram", "gptzero", "originality", "winston", "zerogpt", "openrouter", "lium"
    )  # fmt: skip
    weekly_usd: float = 0.0  # our own spend per period (the period's build), added once per period begun
    period_s: float = 7 * 86400.0

    def allowance_usd(self, now: float | None = None) -> float:
        """The room besides the miners' money: the gap plus ``weekly_usd`` per period begun."""
        return pool_allowance(self.since_ts, self.gap_usd, self.weekly_usd, self.period_s, now)

    def paid_in_usd(self) -> float:
        if not Path(self.billing_db).exists():
            return 0.0
        with sqlite3.connect(f"file:{self.billing_db}?mode=ro", uri=True, timeout=30) as c:
            cents = c.execute(
                "SELECT COALESCE(SUM(net_cents - refunded_cents), 0) FROM payments WHERE ts >= ?",
                (self.since_ts,),
            ).fetchone()[0]
        return cents / 100


def pool_allowance(since_ts: float, gap_usd: float, weekly_usd: float, period_s: float = 7 * 86400.0,
                   now: float | None = None) -> float:  # fmt: skip
    """gap + weekly × the periods begun since ``since_ts`` (the first counts from the launch on)."""
    t = time.time() if now is None else now
    begun = 0 if t < since_ts else math.floor((t - since_ts) / period_s) + 1
    return gap_usd + weekly_usd * begun


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
    def __init__(self, path: Path | str, caps: dict[str, float], pool: PoolLimit | None = None):
        self.pool = pool
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

    def open_reservations(self, service: str) -> list[tuple[str, str, float]]:
        """``(id, ref, estimate)`` of the reservations of ``service`` still open."""
        with self._conn() as c:
            return [
                (r[0], r[1] or "", float(r[2]))
                for r in c.execute(
                    "SELECT id, ref, estimate_usd FROM ledger WHERE service = ? AND status = 'open'",
                    (service,),
                )
            ]

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
                if self.pool is not None and service in self.pool.services:
                    p = self.pool
                    marks = ",".join("?" * len(p.services))
                    used = c.execute(
                        "SELECT COALESCE(SUM(COALESCE(actual_usd, estimate_usd)), 0) FROM ledger "  # noqa: S608
                        f"WHERE ts >= ? AND status != 'released' AND service IN ({marks})",
                        (p.since_ts, *p.services),
                    ).fetchone()[0]
                    paid, room = p.paid_in_usd(), p.allowance_usd()
                    if used + estimate_usd > paid + room + 1e-9:
                        raise BudgetExceededError(
                            f"spend limit: ${used:.2f} spent or reserved since launch, miners paid in "
                            f"${paid:.2f} + gap and weekly allowance ${room:.0f}; requested {service} "
                            f"${estimate_usd:.2f}"
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
