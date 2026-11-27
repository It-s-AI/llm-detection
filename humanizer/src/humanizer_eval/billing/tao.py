"""Paying for evaluations: in TAO, kept as USDC (D-80, D-82; owner's decisions 2026-11-25).

    miner                              evaluator
    -----                              ---------
    deposit / submit             ->    the hotkey's own deposit address and, for a submission, the TAO to
                                       send: the USDC missing for its hold at the current rate, + buffer
    sends TAO (any wallet)       ->    each deposit, whole and on its own: bridged to Base (ForeverMoney over
                                       Chainlink CCIP), swapped TAO -> WETH -> USDC on Uniswap V3; the USDC
                                       that came out is credited to the hotkey's balance
    evaluations                  ->    held and charged in USDC
    refund (any time, on request) ->   USDC -> TAO on Base at the rate of that moment, bridged back to
                                       Finney, to the address the miner names; the cost of the way
                                       back (the CCIP fee, paid in ETH) comes out of the amount; the miner
                                       sees the TAO sent

Balances are kept in cents of USDC (1 cent = 0.01 USDC; the sub-cent rest of a swap stays with the
treasury). A deposit address is the Substrate mirror of an EVM account (``blake2_256("evm:" + h160)``): a
plain transfer funds it, from btcli, any wallet or an exchange. Deposit keys derive from one master secret.

Money safety: a transaction is signed, its hash written down, and only then broadcast; every step resumes
from receipts after a restart; a swap more than ``max_slippage`` under the small-trade price waits; a
refund is marked failed (the money back on the balance) only before anything moved.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TAO_PAYMENT_PREFIX = "tao:"  # payment ids in billing/accounts.py
WEI = 10**18
RAO_WEI = 10**9
USDC_UNIT = 10**6
CENT_USDC = USDC_UNIT // 100
SS58_PREFIX = 42
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


class TaoError(RuntimeError):
    pass


class SlippageTooHigh(TaoError):
    """The pools would pay too little right now: try again later."""


def usdc(cents: int) -> str:
    """Cents shown as USDC: 12345 -> "123.45"."""
    return f"{cents / 100:.2f}"


# --- addresses ---------------------------------------------------------------------------------------
def h160_mirror(h160: str) -> bytes:
    """The Substrate account (32 bytes) whose free balance is the EVM balance of ``h160``."""
    raw = bytes.fromhex(h160.removeprefix("0x"))
    if len(raw) != 20:
        raise ValueError("an EVM address has 20 bytes")
    return hashlib.blake2b(b"evm:" + raw, digest_size=32).digest()


def ss58(pub: bytes) -> str:
    from humanizer_eval.intake.auth import ss58_encode

    return ss58_encode(pub, SS58_PREFIX)


def ss58_pubkey(address: str) -> bytes:
    from humanizer_eval.intake.auth import ss58_decode

    return ss58_decode(address, SS58_PREFIX)


def deposit_key(master: bytes, hotkey: str) -> bytes:
    """secp256k1 private key of a hotkey's deposit account (deterministic: a restart finds it again)."""
    for i in range(256):
        k = hmac.new(master, f"sn32-tao-deposit:{hotkey}:{i}".encode(), hashlib.sha256).digest()
        if 0 < int.from_bytes(k, "big") < SECP256K1_N:
            return k
    raise TaoError("no valid key")  # pragma: no cover


_SCHEMA = """
CREATE TABLE IF NOT EXISTS deposits (
    hotkey TEXT PRIMARY KEY, evm TEXT NOT NULL, address TEXT NOT NULL,
    created_ts REAL NOT NULL, active_until REAL NOT NULL DEFAULT 0, checked_ts REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS inbound (
    tx TEXT PRIMARY KEY, hotkey TEXT NOT NULL, tao_wei TEXT NOT NULL, stage TEXT NOT NULL,
    message_id TEXT, from_block INTEGER, swap_tx TEXT, usdc INTEGER, cents INTEGER, ts REAL NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS refunds (
    key TEXT PRIMARY KEY, hotkey TEXT NOT NULL, dest TEXT NOT NULL, cents INTEGER NOT NULL,
    rows TEXT NOT NULL, stage TEXT NOT NULL, fee_usdc INTEGER, usdc_in INTEGER, buy_tx TEXT, tao_wei TEXT,
    bridge_tx TEXT, message_id TEXT, from_block INTEGER, ts REAL NOT NULL, note TEXT
);
"""
_INBOUND_OPEN = ("bridging", "on_base", "swapping")
_REFUND_OPEN = ("new", "buying", "bought", "bridging", "sent")


@dataclass
class TaoConfig:
    #: the TAO asked for a submission is the USDC missing x (1 + buffer) at the current rate, plus the
    #: bridge's cost: room for the rate to move while the miner pays; the rest stays on the balance
    buffer: float = 0.05
    #: a swap paying more than this under the small-trade price waits (fees are in both)
    max_slippage: float = 0.02
    tx_tolerance: float = 0.005  # the swap's own minimum output, under the quote just taken
    min_bridge_wei: int = WEI // 100  # 0.01 TAO net: less stays on the deposit until more arrives
    min_refund_tao_wei: int = WEI // 100  # the bridge's minimum for a liquid exit on Finney
    refund_gas_usdc: int = 5 * CENT_USDC  # Base gas of a refund's swap, approval and bridge call
    poll_s: float = 30.0  # deposit addresses of hotkeys that are paying now
    idle_poll_s: float = 1800.0  # every other deposit address
    active_s: float = 24 * 3600.0


class TaoRail:
    def __init__(
        self,
        path: Path | str,
        route: Any,  # billing.evm.ForeverMoneyBaseRoute or a fake
        master_key: bytes,
        accounts: Any,  # billing.accounts.Accounts
        cfg: TaoConfig | None = None,
        now: Callable[[], float] = time.time,
        alert: Callable[[str], None] | None = None,
    ):
        if len(master_key) < 32:
            raise ValueError("the deposit master key needs 32 bytes")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.route = route
        self.master = master_key
        self.accounts = accounts
        self.cfg = cfg or TaoConfig()
        self._now = now
        self._alert = alert or (lambda m: None)
        self._lock = threading.RLock()
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        try:
            yield c
        finally:
            c.close()

    def _set(self, table: str, keycol: str, key: str, **cols: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in cols)
        with self._conn() as c:
            q = f"UPDATE {table} SET {sets} WHERE {keycol} = ?"  # noqa: S608 (our own names)
            c.execute(q, (*cols.values(), key))

    # --- deposit addresses -----------------------------------------------------------------------------
    def deposit(self, hotkey: str, active: bool = True) -> dict[str, Any]:
        """The hotkey's deposit address (created once; anyone may send to it); ``active``: watched closely
        for a day."""
        now = self._now()
        with self._conn() as c:
            r = c.execute("SELECT * FROM deposits WHERE hotkey = ?", (hotkey,)).fetchone()
            if r is None:
                evm = self.route.address_of(deposit_key(self.master, hotkey))
                c.execute("INSERT OR IGNORE INTO deposits (hotkey, evm, address, created_ts) "
                          "VALUES (?,?,?,?)",
                          (hotkey, evm, ss58(h160_mirror(evm)), now))  # fmt: skip
            if active:
                c.execute(
                    "UPDATE deposits SET active_until = ? WHERE hotkey = ?", (now + self.cfg.active_s, hotkey)
                )
            r = c.execute("SELECT * FROM deposits WHERE hotkey = ?", (hotkey,)).fetchone()
        return {"address": r["address"], "evm_address": r["evm"]}

    def tao_for_cents(self, cents: int, price: float | None = None) -> int:
        """Wei of TAO to send so that ``cents`` arrive: x (1 + buffer) at the current rate, plus the
        bridge's cost, rounded up to 0.0001 TAO."""
        price = self.route.price() if price is None else price
        if not math.isfinite(price) or price <= 0:
            raise TaoError("no TAO price")
        wei = cents / 100.0 * (1.0 + self.cfg.buffer) / price * WEI
        wei += self.route.bridge_out_cost(int(wei))
        step = WEI // 10_000
        return int(math.ceil(wei / step) * step)

    def pending_cents(self, hotkey: str) -> int:
        """A rough value of the hotkey's TAO already on its way (bridging, swapping), for display."""
        with self._conn() as c:
            rows = c.execute(f"SELECT tao_wei FROM inbound WHERE hotkey = ? AND stage IN {_INBOUND_OPEN}",  # noqa: S608
                             (hotkey,)).fetchall()  # fmt: skip
        wei = sum(int(r[0]) for r in rows)
        return 0 if not wei else int(wei / WEI * self.route.price() * 100)

    # --- inbound: deposit -> Base -> USDC ---------------------------------------------------------------
    def collect(self, hotkey: str) -> int:
        """Start bridging whatever TAO sits on the hotkey's deposit address (whole, as one transfer) and move
        every open conversion on. Returns the cents credited during this call."""
        with self._lock:
            with self._conn() as c:
                d = c.execute("SELECT * FROM deposits WHERE hotkey = ?", (hotkey,)).fetchone()
                c.execute("UPDATE deposits SET checked_ts = ? WHERE hotkey = ?", (self._now(), hotkey))
                busy = c.execute("SELECT 1 FROM inbound WHERE hotkey = ? AND stage = 'bridging'",
                                 (hotkey,)).fetchone()  # fmt: skip
            if d is None:
                raise TaoError(f"no deposit address for {hotkey}")
            if busy is None:  # one bridge at a time per deposit: its balance is the next one's
                bal = self.route.deposit_balance(d["evm"])
                if bal > self.cfg.min_bridge_wei:
                    amount = (bal - self.route.bridge_out_cost(bal)) // RAO_WEI * RAO_WEI
                    if amount >= self.cfg.min_bridge_wei:
                        if not self.route.bridge_ready():
                            raise TaoError("the bridge is paused; TAO stays on the deposit address")
                        tx = self.route.bridge_out_tx(deposit_key(self.master, hotkey), amount)
                        with self._conn() as c:  # written down before it is broadcast
                            c.execute("INSERT INTO inbound (tx, hotkey, tao_wei, stage, ts) "
                                      "VALUES (?,?,?, 'bridging', ?)",
                                      (tx.hash, hotkey, str(amount), self._now()))  # fmt: skip
                        self.route.send("bittensor", tx)
                        log.info("TAO deposit of %s: bridging %.4f TAO", hotkey, amount / WEI)
            return self._advance_inbound(hotkey)

    def _advance_inbound(self, hotkey: str | None = None) -> int:
        credited = 0
        with self._conn() as c:
            q = f"SELECT * FROM inbound WHERE stage IN {_INBOUND_OPEN}"  # noqa: S608
            rows = c.execute(
                q + (" AND hotkey = ?" if hotkey else ""), (hotkey,) if hotkey else ()
            ).fetchall()
        for r in rows:
            try:
                credited += self._step_inbound(r)
            except SlippageTooHigh as e:
                self._set("inbound", "tx", r["tx"], note=str(e))
                self._alert(f"TAO conversion waits: {e}")
            except Exception as e:  # RPC trouble: the next pass
                log.warning("TAO conversion %s: %s", r["tx"], e)
        return credited

    def _step_inbound(self, r: sqlite3.Row) -> int:
        tx, st = r["tx"], r["stage"]
        if st == "bridging":
            res = self.route.bridge_out_result(tx)
            if res is None:
                if self._now() - r["ts"] > 3600:  # never mined: the TAO is still on the deposit
                    self._set("inbound", "tx", tx, stage="failed", note="not mined")
                return 0
            if not res["ok"]:
                self._set("inbound", "tx", tx, stage="failed", note="bridge call reverted")
                return 0
            self._set(
                "inbound",
                "tx",
                tx,
                stage="on_base",
                message_id=res["message_id"],
                from_block=res["from_block"],
            )
            return 0
        if st == "on_base":
            got = self.route.arrived_on_base(r["message_id"], r["from_block"])
            if got is None:
                return 0
            if got is False:
                self._set("inbound", "tx", tx, stage="stuck", note="CCIP delivery failed: retry it manually")
                self._alert(f"TAO bridge message {r['message_id']} failed to deliver on Base (deposit {tx})")
                return 0
            wei = int(r["tao_wei"])
            expect = self.route.quote_sell(wei)
            fair = wei / WEI * self.route.price() * USDC_UNIT
            if expect < fair * (1.0 - self.cfg.max_slippage):
                raise SlippageTooHigh(
                    f"{1 - expect / fair:.1%} under the small-trade price for {wei / WEI:.4f} TAO"
                )
            swap = self.route.sell_tx(wei, int(expect * (1.0 - self.cfg.tx_tolerance)))
            self._set("inbound", "tx", tx, stage="swapping", swap_tx=swap.hash, note=None)
            self.route.send("base", swap)
            return 0
        if st == "swapping":
            res = self.route.swap_result(r["swap_tx"])
            if res is None:
                return 0
            if not res["ok"]:  # the TAO is still on Base: swap again
                self._set("inbound", "tx", tx, stage="on_base", swap_tx=None)
                return 0
            amount = int(res["usdc_in"])
            cents = amount // CENT_USDC
            if cents > 0:
                self.accounts.record_payment(
                    TAO_PAYMENT_PREFIX + tx, r["hotkey"], cents, 0,
                    {"tao_wei": r["tao_wei"], "usdc": amount, "message_id": r["message_id"],
                     "swap_tx": r["swap_tx"]},
                )  # fmt: skip
            self._set("inbound", "tx", tx, stage="done", usdc=amount, cents=cents)
            log.info(
                "TAO deposit %s of %s: %.4f TAO -> %s USDC",
                tx,
                r["hotkey"],
                int(r["tao_wei"]) / WEI,
                usdc(cents),
            )
            return cents
        return 0

    def inbound(self, hotkey: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM inbound WHERE hotkey = ? ORDER BY ts DESC LIMIT ?", (hotkey, limit)
            ).fetchall()
        return [{"tx": r["tx"], "tao": int(r["tao_wei"]) / WEI, "stage": r["stage"],
                 "usdc": None if r["cents"] is None else usdc(r["cents"]),
                 "at": r["ts"]} for r in rows]  # fmt: skip

    # --- refunds: USDC -> TAO -> Finney -----------------------------------------------------------------
    def refund_cost_usdc(self, cents: int, dest: str) -> int:
        """USDC units the way back costs now: the CCIP fee (in ETH, at the pools' rate) and Base gas."""
        tao = self.route.quote_buy(cents * CENT_USDC)
        fee_eth = self.route.bridge_in_fee(max(tao, self.cfg.min_refund_tao_wei), ss58_pubkey(dest))
        return math.ceil(fee_eth / WEI * self.route.eth_usd() * USDC_UNIT) + self.cfg.refund_gas_usdc

    def start_refund(self, key: str, hotkey: str, cents: int, dest: str, rows: list[tuple[str, int]]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO refunds (key, hotkey, dest, cents, rows, stage, ts) "
                "VALUES (?,?,?,?,?, 'new', ?)",
                (key, hotkey, dest, int(cents), json.dumps(rows), self._now()),
            )

    def refund_state(self, key: str) -> dict[str, Any] | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM refunds WHERE key = ?", (key,)).fetchone()
        if r is None:
            return None
        return {"key": key, "stage": r["stage"], "amount_usdc": usdc(r["cents"]), "to": r["dest"],
                "cost_usdc": None if r["fee_usdc"] is None else f"{r['fee_usdc'] / USDC_UNIT:.2f}",
                "tao_sent": None if not r["tao_wei"] else int(r["tao_wei"]) / WEI,
                "ccip_message": r["message_id"], "note": r["note"]}  # fmt: skip

    def _finish_rows(self, r: sqlite3.Row, ok: bool, ref: str | None = None) -> None:
        for pid, _ in json.loads(r["rows"]):
            self.accounts.finish_refund(r["key"], pid, ok, ref)

    def drive_refund(self, key: str) -> str:
        """Move a refund as far as it goes now. Returns its stage ("sent": the TAO left for Finney)."""
        with self._lock:
            for _ in range(6):
                with self._conn() as c:
                    r = c.execute("SELECT * FROM refunds WHERE key = ?", (key,)).fetchone()
                st = r["stage"]
                if st == "new":
                    fee = self.refund_cost_usdc(r["cents"], r["dest"])
                    usdc_in = r["cents"] * CENT_USDC - fee
                    expect = self.route.quote_buy(usdc_in) if usdc_in > 0 else 0
                    if expect < self.cfg.min_refund_tao_wei:
                        self._set("refunds", "key", key, stage="failed", fee_usdc=fee,
                                  note="the amount does not cover the way back plus the bridge's "
                                  "0.01 TAO minimum")  # fmt: skip
                        self._finish_rows(r, False)
                        return "failed"
                    fair = usdc_in / USDC_UNIT / self.route.price() * WEI
                    if expect < fair * (1.0 - self.cfg.max_slippage):
                        self._set("refunds", "key", key, note="waiting for the pools")
                        raise SlippageTooHigh(
                            f"refund {key}: {1 - expect / fair:.1%} under the small-trade price"
                        )
                    tx = self.route.buy_tx(usdc_in, int(expect * (1.0 - self.cfg.tx_tolerance)))
                    self._set(
                        "refunds",
                        "key",
                        key,
                        stage="buying",
                        fee_usdc=fee,
                        usdc_in=usdc_in,
                        buy_tx=tx.hash,
                        note=None,
                    )
                    self.route.send("base", tx)
                elif st == "buying":
                    res = self.route.swap_result(r["buy_tx"])
                    if res is None:
                        return st
                    if not res["ok"]:
                        self._set("refunds", "key", key, stage="new", buy_tx=None)
                        continue
                    self._set("refunds", "key", key, stage="bought", tao_wei=str(int(res["tao_in"])))
                elif st == "bought":
                    tx = self.route.bridge_in_tx(int(r["tao_wei"]), ss58_pubkey(r["dest"]))
                    self._set("refunds", "key", key, stage="bridging", bridge_tx=tx.hash)
                    self.route.send("base", tx)
                elif st == "bridging":
                    res = self.route.bridge_in_result(r["bridge_tx"])
                    if res is None:
                        return st
                    if not res["ok"]:
                        self._set("refunds", "key", key, stage="bought", bridge_tx=None)
                        continue
                    self._set(
                        "refunds",
                        "key",
                        key,
                        stage="sent",
                        message_id=res["message_id"],
                        from_block=res["from_block"],
                    )
                    self._finish_rows(r, True, r["bridge_tx"])
                    log.info("refund %s: %.4f TAO to %s", key, int(r["tao_wei"]) / WEI, r["dest"])
                elif st == "sent":
                    got = self.route.delivered_on_finney(r["message_id"], r["from_block"])
                    if got is None:
                        if self._now() - r["ts"] > 86400:  # the delivery cannot be seen: stop looking
                            self._set("refunds", "key", key, stage="sent_unconfirmed")
                        return st
                    self._set("refunds", "key", key, stage="delivered" if got else "stuck")
                    if not got:
                        self._alert(f"refund {key}: CCIP delivery to Finney failed, retry it manually")
                    return "delivered" if got else "stuck"
                else:
                    return st
            return st

    # --- background ----------------------------------------------------------------------------------
    def watch(self) -> dict[str, list[str]]:
        """Look at deposit addresses (paying hotkeys every ``poll_s``, the others every ``idle_poll_s``),
        move conversions and refunds on."""
        now = self._now()
        with self._conn() as c:
            deps = c.execute("SELECT hotkey, active_until, checked_ts FROM deposits").fetchall()
            refunds = [r[0] for r in c.execute(f"SELECT key FROM refunds WHERE stage IN {_REFUND_OPEN}")]  # noqa: S608
        credited = []
        for d in deps:
            every = self.cfg.poll_s if now < d["active_until"] else self.cfg.idle_poll_s
            if now - d["checked_ts"] < every:
                continue
            try:
                if self.collect(d["hotkey"]) > 0:
                    credited.append(d["hotkey"])
            except Exception as e:
                log.warning("deposit of %s not checked: %s", d["hotkey"], e)
        credited += [h for h, n in self._credited_elsewhere().items() if n]
        for key in refunds:
            try:
                self.drive_refund(key)
            except Exception as e:
                log.warning("refund %s waits: %s", key, e)
        return {"credited": credited}

    def _credited_elsewhere(self) -> dict[str, int]:
        """Conversions of hotkeys not checked in this pass (their deposit was idle) still move on."""
        out: dict[str, int] = {}
        with self._conn() as c:
            hks = [
                r[0]
                for r in c.execute(f"SELECT DISTINCT hotkey FROM inbound WHERE stage IN {_INBOUND_OPEN}")  # noqa: S608
            ]
        for hk in hks:
            with self._lock:
                out[hk] = self._advance_inbound(hk)
        return out

    def summary(self) -> dict[str, Any]:
        with self._conn() as c:
            conv = c.execute(
                "SELECT COUNT(*), COALESCE(SUM(usdc), 0) FROM inbound WHERE stage = 'done'"
            ).fetchone()
            ref = c.execute(
                "SELECT COUNT(*), COALESCE(SUM(cents), 0) FROM refunds WHERE stage IN ('sent', 'delivered')"
            ).fetchone()
            open_in = c.execute(f"SELECT COUNT(*) FROM inbound WHERE stage IN {_INBOUND_OPEN}").fetchone()[0]  # noqa: S608
            open_ref = c.execute(f"SELECT COUNT(*) FROM refunds WHERE stage IN {_REFUND_OPEN}").fetchone()[0]  # noqa: S608
            stuck = c.execute("SELECT COUNT(*) FROM inbound WHERE stage = 'stuck'").fetchone()[0] + c.execute(
                "SELECT COUNT(*) FROM refunds WHERE stage = 'stuck'").fetchone()[0]  # fmt: skip
        return {"deposits_converted": conv[0], "converted_usdc": conv[1] / USDC_UNIT, "refunds": ref[0],
                "refunded_usdc": ref[1] / 100, "in_progress": {"deposits": open_in, "refunds": open_ref},
                "stuck": stuck}  # fmt: skip
