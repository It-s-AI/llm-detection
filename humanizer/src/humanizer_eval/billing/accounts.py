"""Miner balances: payments, holds, settlements and refunds in integer cents of USDC (D-80, D-82).

    balance(hotkey)   = sum(net payments) - sum(settled charges) - sum(refunds)
    available(hotkey) = balance - sum(open holds)

* A payment is a TAO deposit converted to USDC (billing/tao.py), credited with the USDC that came out.
* A submission opens a hold for its quote; settling charges the metered statement, capped at the hold
  (the quote is an upper bound; an overrun is the owner's), and frees the rest.
* A refund is reserved against the payments, newest first, never more than a payment's remaining amount;
  it is refused while any hold of the hotkey is open (no withdrawal during a run).

Every mutating call is idempotent on its natural key (payment id, submission id, refund key) and runs in
``BEGIN IMMEDIATE`` so concurrent processes cannot double-spend a balance.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS payments (
    id TEXT PRIMARY KEY, hotkey TEXT NOT NULL, gross_cents INTEGER NOT NULL, fee_cents INTEGER NOT NULL,
    net_cents INTEGER NOT NULL, refunded_cents INTEGER NOT NULL DEFAULT 0, ts REAL NOT NULL, meta TEXT
);
CREATE TABLE IF NOT EXISTS holds (
    submission_id TEXT PRIMARY KEY, hotkey TEXT NOT NULL, amount_cents INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'settled', 'released')),
    charged_cents INTEGER NOT NULL DEFAULT 0, ts REAL NOT NULL, closed_ts REAL, statement TEXT
);
CREATE TABLE IF NOT EXISTS refunds (
    key TEXT NOT NULL, payment_id TEXT NOT NULL, hotkey TEXT NOT NULL, amount_cents INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'done', 'failed')), stripe_id TEXT, ts REAL NOT NULL,
    PRIMARY KEY (key, payment_id)
);
CREATE TABLE IF NOT EXISTS quotes (
    id TEXT PRIMARY KEY, hotkey TEXT NOT NULL, hold_cents INTEGER NOT NULL, expires_ts REAL NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS holds_hotkey ON holds(hotkey, status);
CREATE INDEX IF NOT EXISTS payments_hotkey ON payments(hotkey);
"""


_SELECT_HOLD = (
    "SELECT submission_id, hotkey, amount_cents, status, charged_cents FROM holds WHERE submission_id = ?"
)
_INSERT_PAYMENT = (
    "INSERT INTO payments (id, hotkey, gross_cents, fee_cents, net_cents, ts, meta) VALUES (?,?,?,?,?,?,?)"
)
_SETTLE = (
    "UPDATE holds SET status = 'settled', charged_cents = ?, closed_ts = ?, statement = ? "
    "WHERE submission_id = ?"
)
_INSERT_REFUND = (
    "INSERT INTO refunds (key, payment_id, hotkey, amount_cents, status, ts) VALUES (?,?,?,?, 'pending', ?)"
)


class BillingError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hold:
    submission_id: str
    hotkey: str
    amount_cents: int
    status: str
    charged_cents: int


class Accounts:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def _tx(self):
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise

    # --- reads ------------------------------------------------------------------------------------
    @staticmethod
    def _balance(c: sqlite3.Connection, hotkey: str) -> tuple[int, int]:
        net = c.execute(
            "SELECT COALESCE(SUM(net_cents), 0) FROM payments WHERE hotkey = ?", (hotkey,)
        ).fetchone()[0]
        charged = c.execute(
            "SELECT COALESCE(SUM(charged_cents), 0) FROM holds WHERE hotkey = ? AND status = 'settled'",
            (hotkey,),
        ).fetchone()[0]
        refunded = c.execute(
            "SELECT COALESCE(SUM(amount_cents), 0) FROM refunds WHERE hotkey = ? AND status != 'failed'",
            (hotkey,),
        ).fetchone()[0]
        held = c.execute(
            "SELECT COALESCE(SUM(amount_cents), 0) FROM holds WHERE hotkey = ? AND status = 'open'", (hotkey,)
        ).fetchone()[0]
        balance = net - charged - refunded
        return balance, balance - held

    def balance(self, hotkey: str) -> int:
        with self._conn() as c:
            return self._balance(c, hotkey)[0]

    def available(self, hotkey: str) -> int:
        with self._conn() as c:
            return self._balance(c, hotkey)[1]

    def hold(self, submission_id: str) -> Hold | None:
        with self._conn() as c:
            r = c.execute(
                _SELECT_HOLD,
                (submission_id,),
            ).fetchone()
        return Hold(*r) if r else None

    # --- quotes -----------------------------------------------------------------------------------
    def save_quote(self, quote_id: str, hotkey: str, hold_cents: int, expires_ts: float, body: dict) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO quotes (id, hotkey, hold_cents, expires_ts, body) VALUES (?,?,?,?,?)",
                (quote_id, hotkey, hold_cents, expires_ts, json.dumps(body)),
            )

    def get_quote(self, quote_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute(
                "SELECT hotkey, hold_cents, expires_ts, body FROM quotes WHERE id = ?", (quote_id,)
            ).fetchone()
        if not r:
            return None
        return {
            "id": quote_id,
            "hotkey": r[0],
            "hold_cents": r[1],
            "expires_ts": r[2],
            "body": json.loads(r[3]),
        }

    # --- payments ---------------------------------------------------------------------------------
    def record_payment(
        self, payment_id: str, hotkey: str, gross_cents: int, fee_cents: int, meta: dict | None = None
    ) -> bool:
        """Credit ``gross - fee``. Returns False if this payment was already recorded (idempotent)."""
        if gross_cents <= 0 or not 0 <= fee_cents <= gross_cents:
            raise BillingError("invalid payment amounts")
        with self._tx() as c:
            row = c.execute("SELECT hotkey, gross_cents FROM payments WHERE id = ?", (payment_id,)).fetchone()
            if row:
                if tuple(row) != (hotkey, gross_cents):
                    raise BillingError(f"payment {payment_id} already recorded with different data")
                return False
            c.execute(
                _INSERT_PAYMENT,
                (
                    payment_id,
                    hotkey,
                    gross_cents,
                    fee_cents,
                    gross_cents - fee_cents,
                    time.time(),
                    json.dumps(meta or {}),
                ),
            )
            return True

    # --- holds ------------------------------------------------------------------------------------
    def open_hold(self, submission_id: str, hotkey: str, amount_cents: int) -> Hold:
        if amount_cents <= 0:
            raise BillingError("hold amount must be positive")
        with self._tx() as c:
            row = c.execute(
                _SELECT_HOLD,
                (submission_id,),
            ).fetchone()
            if row:
                h = Hold(*row)
                if (h.hotkey, h.amount_cents) != (hotkey, amount_cents):
                    raise BillingError(f"submission {submission_id} already has a different hold")
                return h
            _, avail = self._balance(c, hotkey)
            if avail < amount_cents:
                raise BillingError(f"insufficient balance: available {avail} < hold {amount_cents} cents")
            c.execute(
                "INSERT INTO holds (submission_id, hotkey, amount_cents, status, ts) "
                "VALUES (?,?,?, 'open', ?)",
                (submission_id, hotkey, amount_cents, time.time()),
            )
            return Hold(submission_id, hotkey, amount_cents, "open", 0)

    def settle(self, submission_id: str, actual_cents: int, statement: dict | None = None) -> Hold:
        """Charge ``min(actual, hold)``; the rest of the hold returns to the balance. Idempotent."""
        if actual_cents < 0:
            raise BillingError("actual cost must be non-negative")
        with self._tx() as c:
            row = c.execute(
                _SELECT_HOLD,
                (submission_id,),
            ).fetchone()
            if not row:
                raise BillingError(f"no hold for submission {submission_id}")
            h = Hold(*row)
            if h.status != "open":
                return h
            charged = min(actual_cents, h.amount_cents)
            c.execute(
                _SETTLE,
                (charged, time.time(), json.dumps(statement or {}), submission_id),
            )
            return Hold(h.submission_id, h.hotkey, h.amount_cents, "settled", charged)

    def release(self, submission_id: str) -> Hold:
        """Close a hold without charge (the run failed on our side). Idempotent."""
        with self._tx() as c:
            row = c.execute(
                _SELECT_HOLD,
                (submission_id,),
            ).fetchone()
            if not row:
                raise BillingError(f"no hold for submission {submission_id}")
            h = Hold(*row)
            if h.status == "open":
                c.execute(
                    "UPDATE holds SET status = 'released', closed_ts = ? WHERE submission_id = ?",
                    (time.time(), submission_id),
                )
                return Hold(h.submission_id, h.hotkey, h.amount_cents, "released", 0)
            return h

    # --- refunds ----------------------------------------------------------------------------------
    def plan_refund(
        self, key: str, hotkey: str, amount_cents: int | None = None, prefix: str | None = None
    ) -> list[tuple[str, int]]:
        """Reserve a refund of ``amount`` (default: everything available) across payments, newest first;
        ``prefix``: only payments whose id starts with it (e.g. the TAO ones, billing/tao.py).

        Returns ``[(payment_id, cents)]`` rows in state 'pending'; re-planning the same key returns the
        rows already planned (idempotent). Refused while a hold is open.
        """
        with self._tx() as c:
            existing = c.execute(
                "SELECT payment_id, amount_cents FROM refunds WHERE key = ? AND hotkey = ?", (key, hotkey)
            ).fetchall()
            if existing:
                return [(p, a) for p, a in existing]
            if c.execute("SELECT 1 FROM holds WHERE hotkey = ? AND status = 'open'", (hotkey,)).fetchone():
                raise BillingError(
                    "a submission of this hotkey is being evaluated; refunds wait until it is settled"
                )
            _, avail = self._balance(c, hotkey)
            if prefix is not None:
                avail = min(avail, self._refundable(c, hotkey, prefix))
            want = avail if amount_cents is None else amount_cents
            if want <= 0:
                raise BillingError("nothing to refund")
            if want > avail:
                raise BillingError(f"refund {want} exceeds available balance {avail} cents")
            plan, left = [], want
            for pid, gross, refunded in c.execute(
                "SELECT id, gross_cents, refunded_cents FROM payments WHERE hotkey = ? AND id LIKE ? "
                "ORDER BY ts DESC, id DESC",
                (hotkey, (prefix or "") + "%"),
            ).fetchall():
                take = min(left, gross - refunded)
                if take > 0:
                    plan.append((pid, take))
                    left -= take
                if left == 0:
                    break
            if left:
                raise BillingError("payments cannot cover the refund")
            now = time.time()
            for pid, amt in plan:
                c.execute(
                    _INSERT_REFUND,
                    (key, pid, hotkey, amt, now),
                )
                c.execute("UPDATE payments SET refunded_cents = refunded_cents + ? WHERE id = ?", (amt, pid))
            return plan

    @staticmethod
    def _refundable(c: sqlite3.Connection, hotkey: str, prefix: str) -> int:
        return c.execute(
            "SELECT COALESCE(SUM(gross_cents - refunded_cents), 0) FROM payments "
            "WHERE hotkey = ? AND id LIKE ?",
            (hotkey, prefix + "%"),
        ).fetchone()[0]

    def refundable(self, hotkey: str, prefix: str) -> int:
        """What payments whose id starts with ``prefix`` can still take back (before the balance cap)."""
        with self._conn() as c:
            return self._refundable(c, hotkey, prefix)

    def finish_refund(self, key: str, payment_id: str, ok: bool, ref: str | None = None) -> None:
        with self._tx() as c:
            row = c.execute(
                "SELECT status, amount_cents FROM refunds WHERE key = ? AND payment_id = ?", (key, payment_id)
            ).fetchone()
            if not row:
                raise BillingError("unknown refund")
            status, amt = row
            if status != "pending":
                return
            c.execute(
                "UPDATE refunds SET status = ?, stripe_id = ? WHERE key = ? AND payment_id = ?",
                ("done" if ok else "failed", ref, key, payment_id),
            )
            if not ok:  # the money stays with the miner's balance
                c.execute(
                    "UPDATE payments SET refunded_cents = refunded_cents - ? WHERE id = ?", (amt, payment_id)
                )

    def pending_refunds(self) -> list[tuple[str, str, str, int]]:
        with self._conn() as c:
            return c.execute(
                "SELECT key, payment_id, hotkey, amount_cents FROM refunds WHERE status = 'pending'"
            ).fetchall()
