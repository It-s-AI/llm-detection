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
    note TEXT, reverts INTEGER NOT NULL DEFAULT 0, raw_tx TEXT, raw_chain TEXT, raw_ts REAL
);
CREATE TABLE IF NOT EXISTS refunds (
    key TEXT PRIMARY KEY, hotkey TEXT NOT NULL, dest TEXT NOT NULL, cents INTEGER NOT NULL,
    rows TEXT NOT NULL, stage TEXT NOT NULL, fee_usdc INTEGER, usdc_in INTEGER, buy_tx TEXT, tao_wei TEXT,
    bridge_tx TEXT, message_id TEXT, from_block INTEGER, ts REAL NOT NULL, note TEXT,
    claim_tx TEXT, forward_tx TEXT, reverts INTEGER NOT NULL DEFAULT 0, raw_tx TEXT, raw_chain TEXT,
    raw_ts REAL
);
"""
#: a transaction that reverts is sent again (the next pass, fresh quote) at most this many times in all for
#: one deposit or refund; then it is "stuck" and the owner is alerted: every revert costs gas, and a revert
#: that repeats (a bad parameter, a paused contract) would otherwise burn it on every pass
MAX_REVERTS = 3
REFUND_WAIT_NOTE = "waiting for funds: the refund is paid within 24 hours"
#: a recorded transaction without a receipt this long is reported (it keeps being rebroadcast)
PENDING_ALERT_S = 1800.0
_INBOUND_OPEN = ("bridging", "on_base", "swapping")
_REFUND_OPEN = (
    "new",
    "buying",
    "bought",
    "bridging",
    "sent",
    "claimable",
    "claiming",
    "claimed",
    "forwarding",
)


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
            have = {row[1] for row in c.execute("PRAGMA table_info(refunds)")}
            for col in ("claim_tx", "forward_tx"):  # tables made before the claim path
                if col not in have:
                    c.execute(f"ALTER TABLE refunds ADD COLUMN {col} TEXT")
            for table in ("inbound", "refunds"):  # tables made before the revert limit / the rebroadcast
                cols = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
                for col, typ in (("reverts", "INTEGER NOT NULL DEFAULT 0"), ("raw_tx", "TEXT"),
                                 ("raw_chain", "TEXT"), ("raw_ts", "REAL")):  # fmt: skip
                    if col not in cols:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

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
    def _raw(self, chain: str, tx: Any) -> dict[str, Any]:
        """The signed transaction, kept with its row before it is broadcast: if the broadcast is lost (RPC
        error, restart in between) it is sent again until it is mined."""
        return {"raw_tx": tx.raw, "raw_chain": chain, "raw_ts": self._now()}

    def _still_pending(
        self, table: str, keycol: str, key: str, r: sqlite3.Row, tx_hash: str, **back: Any
    ) -> None:
        """No receipt yet: send it again. 'nonce too low' means mined (receipt not visible yet) or replaced:
        the step is never signed again on its own; if it lasts past PENDING_ALERT_S the row is stuck for the
        owner (``back`` marks the steps where a replacement is possible)."""
        if not r["raw_tx"]:
            return
        # imported here: the metagraph script imports billing in an environment without the EVM libraries
        from humanizer_eval.billing.evm import SignedTx

        try:
            self.route.send(r["raw_chain"], SignedTx(tx_hash, r["raw_tx"]))  # "already known" is fine
        except Exception as e:
            log.warning("%s %s: rebroadcast failed: %s", table, key, e)
            used = ("nonce too low", "nonce has already been used", "noncetoolow")
            if back and any(k in str(e).lower() for k in used):
                # 'nonce too low' also comes back for a tx already MINED whose receipt this node does not show
                # yet: never sign the step again on that alone (a second swap/bridge/forward would pay twice).
                # Past PENDING_ALERT_S still without a receipt it is stuck for the owner, who checks the chain
                age = self._now() - float(r["raw_ts"] or 0)
                if age > PENDING_ALERT_S and (r["stage"] or "") != "stuck":
                    note = f"no receipt and its nonce is used ({tx_hash}); check on chain by hand"
                    self._set(table, keycol, key, stage="stuck", note=note)
                    self._alert(f"{table} {key}: tx {tx_hash} has no receipt and its nonce is used "
                                "(mined, not indexed, or replaced): stuck; check it on chain")  # fmt: skip
                return
        old = self._now() - float(r["raw_ts"] or 0) > PENDING_ALERT_S
        if old and not (r["note"] or "").startswith("not mined"):
            why = f"not mined after {PENDING_ALERT_S / 60:.0f} min (gas? nonce?)"
            self._set(table, keycol, key, note=why)
            self._alert(f"{table} {key}: a transaction is {why}; it keeps being sent again")

    def _reverted(self, table: str, keycol: str, key: str, r: sqlite3.Row, what: str, **back: Any) -> bool:
        """A transaction of this row reverted: back to the stage before it (``back``) for another try, or
        "stuck" after ``MAX_REVERTS``. Returns True when it is stuck."""
        n = int(r["reverts"] or 0) + 1
        if n >= MAX_REVERTS:
            self._set(table, keycol, key, stage="stuck", reverts=n, note=f"{what} reverted {n} times")
            self._alert(f"{table} {key}: {what} reverted {n} times; stopped, look at it by hand")
            return True
        self._set(table, keycol, key, reverts=n, **back)
        return False

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

    def incoming(self, hotkey: str) -> bool:
        """TAO of this hotkey on its way: a conversion in flight, or TAO on its deposit address not bridged
        yet (just arrived, or the bridge is paused). No price is needed (it must not fail on a quote)."""
        with self._conn() as c:
            if c.execute(f"SELECT 1 FROM inbound WHERE hotkey = ? AND stage IN {_INBOUND_OPEN}",  # noqa: S608
                         (hotkey,)).fetchone():  # fmt: skip
                return True
            d = c.execute("SELECT evm FROM deposits WHERE hotkey = ?", (hotkey,)).fetchone()
        try:  # only TAO that can actually be bridged (dust under the minimum never will be)
            bal = self.route.deposit_balance(d["evm"]) if d is not None else 0
            return bal - self.route.bridge_out_cost(bal) >= self.cfg.min_bridge_wei
        except Exception:  # RPC trouble: assume it may be on its way (the window still ends later)
            return True

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
                reverted = c.execute(
                    "SELECT COUNT(*) FROM inbound WHERE hotkey = ? AND stage = 'failed' AND note = ?",
                    (hotkey, "bridge call reverted"),
                ).fetchone()[0]
            if d is None:
                raise TaoError(f"no deposit address for {hotkey}")
            if busy is None:  # one bridge at a time per deposit: its balance is the next one's
                bal = self.route.deposit_balance(d["evm"])
                if bal > self.cfg.min_bridge_wei:
                    amount = (bal - self.route.bridge_out_cost(bal)) // RAO_WEI * RAO_WEI
                    if amount >= self.cfg.min_bridge_wei and reverted >= MAX_REVERTS:
                        # the gas of each try comes from the deposit: the owner looks first
                        self._alert(
                            f"deposit of {hotkey}: {reverted} bridge calls reverted; stopped, look at it"
                        )
                        raise TaoError(f"{reverted} reverted bridge calls: the deposit waits for the owner")
                    if amount >= self.cfg.min_bridge_wei:
                        if not self.route.bridge_ready():
                            raise TaoError("the bridge is paused; TAO stays on the deposit address")
                        tx = self.route.bridge_out_tx(deposit_key(self.master, hotkey), amount)
                        with self._conn() as c:  # written down before it is broadcast
                            c.execute(
                                "INSERT INTO inbound (tx, hotkey, tao_wei, stage, ts, raw_tx, raw_chain, "
                                "raw_ts) VALUES (?,?,?, 'bridging', ?, ?, 'bittensor', ?)",
                                (tx.hash, hotkey, str(amount), self._now(), tx.raw, self._now()),
                            )
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
                self._still_pending("inbound", "tx", tx, r, tx)
                if self._now() - r["ts"] > 3600:  # never mined: the TAO is still on the deposit
                    self._set("inbound", "tx", tx, stage="failed", note="not mined")
                    # it can still be mined later (then the wTAO reaches the treasury unswapped): look by hand
                    self._alert(f"deposit bridge {tx} of {r['hotkey']}: not mined in 1 h, marked failed")
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
            raw = self._raw("base", swap)
            self._set("inbound", "tx", tx, stage="swapping", swap_tx=swap.hash, note=None, **raw)
            self.route.send("base", swap)
            return 0
        if st == "swapping":
            res = self.route.swap_result(r["swap_tx"])
            if res is None:
                self._still_pending("inbound", "tx", tx, r, r["swap_tx"], stage="on_base", swap_tx=None)
                return 0
            if not res["ok"]:  # the TAO is still on Base: swap again
                self._reverted("inbound", "tx", tx, r, "the swap", stage="on_base", swap_tx=None)
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
        """USDC units the way back costs now: the CCIP fee (in ETH, at the pools' rate), Base gas, and the
        Bittensor EVM gas of claiming the TAO and sending it on (the bridge's liquid exit books it)."""
        tao = self.route.quote_buy(cents * CENT_USDC)
        whole = max(tao // RAO_WEI * RAO_WEI, self.cfg.min_refund_tao_wei)
        fee_eth = self.route.bridge_in_fee(whole, ss58_pubkey(dest))
        claim = self.route.finney_gas_wei() / WEI * self.route.price()
        return (
            math.ceil((fee_eth / WEI * self.route.eth_usd() + claim) * USDC_UNIT) + self.cfg.refund_gas_usdc
        )

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
                        # three steps, each safe to crash after: 'failing' (no pass buys any more), the
                        # balance back, then 'failed' (reconcile_refunds finishes a 'failing' one)
                        self._set("refunds", "key", key, stage="failing", fee_usdc=fee,
                                  note="the amount does not cover the way back plus the bridge's "
                                  "0.01 TAO minimum")  # fmt: skip
                        self._finish_rows(r, False)
                        self._set("refunds", "key", key, stage="failed")
                        return "failed"
                    # the earnings are paid out (scripts/tao_treasury.py --payout): if the treasury is short,
                    # the refund waits for the owner to top it up (promised within a day) and nothing moves
                    if hasattr(self.route, "treasury_usdc") and self.route.treasury_usdc() < usdc_in:
                        self._set("refunds", "key", key, note=REFUND_WAIT_NOTE)
                        need = usdc_in / USDC_UNIT
                        self._alert(f"refund {key}: the treasury has under {need:.2f} USDC; top it up (24 h)")
                        return st
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
                        **self._raw("base", tx),
                    )
                    self.route.send("base", tx)
                elif st == "buying":
                    res = self.route.swap_result(r["buy_tx"])
                    if res is None:
                        self._still_pending("refunds", "key", key, r, r["buy_tx"], stage="new", buy_tx=None)
                        return st
                    if not res["ok"]:
                        if self._reverted("refunds", "key", key, r, "the buy", stage="new", buy_tx=None):
                            return "stuck"
                        continue
                    # the bridge moves whole RAO: the dust under one stays on Base with the treasury
                    tao = int(res["tao_in"]) // RAO_WEI * RAO_WEI
                    self._set("refunds", "key", key, stage="bought", tao_wei=str(tao))
                elif st == "bought":
                    tx = self.route.bridge_in_tx(int(r["tao_wei"]), ss58_pubkey(r["dest"]))
                    raw = self._raw("base", tx)
                    self._set("refunds", "key", key, stage="bridging", bridge_tx=tx.hash, **raw)
                    self.route.send("base", tx)
                elif st == "bridging":
                    res = self.route.bridge_in_result(r["bridge_tx"])
                    if res is None:
                        self._still_pending("refunds", "key", key, r, r["bridge_tx"], stage="bought",
                                            bridge_tx=None)  # fmt: skip
                        return st
                    if not res["ok"]:
                        back = {"stage": "bought", "bridge_tx": None}
                        if self._reverted("refunds", "key", key, r, "the bridge", **back):
                            return "stuck"
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
                    got = self.route.finney_outcome(r["message_id"], r["from_block"])
                    if got is None:
                        # counted from the bridge transaction (raw_ts), not from the request: a refund
                        # that waited for the pools must still be watched for a day after it is sent
                        sent_at = float(r["raw_ts"] or r["ts"])
                        if self._now() - sent_at > 86400:  # the delivery cannot be seen: stop looking
                            self._set("refunds", "key", key, stage="sent_unconfirmed")
                            mid = r["message_id"]
                            self._alert(f"refund {key}: CCIP delivery not seen in 24 h (message {mid}): look")
                        return st
                    if got == "failed":
                        self._set("refunds", "key", key, stage="stuck")
                        self._alert(f"refund {key}: CCIP delivery to Finney failed, retry it manually")
                        return "stuck"
                    if got == "liquid":
                        self._set("refunds", "key", key, stage="delivered")
                        return "delivered"
                    self._set("refunds", "key", key, stage="claimable")
                elif st == "claimable":
                    booked = self.route.claimable_tao()
                    if booked <= 0:  # an earlier claim took it with another refund's: it is on the account
                        self._set("refunds", "key", key, stage="claimed")
                        continue
                    if not self._gas_for(key, r, 0):
                        return st
                    tx = self.route.claim_tx(booked)
                    self._set("refunds", "key", key, stage="claiming", claim_tx=tx.hash, note=None,
                              **self._raw("bittensor", tx))  # fmt: skip
                    self.route.send("bittensor", tx)
                elif st == "claiming":
                    res = self.route.bt_tx_ok(r["claim_tx"])
                    if res is None:
                        self._still_pending("refunds", "key", key, r, r["claim_tx"], stage="claimable",
                                            claim_tx=None)  # fmt: skip
                        return st
                    if not res:
                        if self._reverted("refunds", "key", key, r, "the claim", stage="claimable"):
                            return "stuck"
                        return "claimable"  # the next pass, not now: the same block would revert again
                    self._set("refunds", "key", key, stage="claimed")
                elif st == "claimed":
                    wei = int(r["tao_wei"]) // RAO_WEI * RAO_WEI
                    if not self._gas_for(key, r, wei):
                        return st
                    tx = self.route.forward_tx(wei, ss58_pubkey(r["dest"]))
                    self._set("refunds", "key", key, stage="forwarding", forward_tx=tx.hash, note=None,
                              **self._raw("bittensor", tx))  # fmt: skip
                    self.route.send("bittensor", tx)
                elif st == "forwarding":
                    res = self.route.bt_tx_ok(r["forward_tx"])
                    if res is None:
                        self._still_pending("refunds", "key", key, r, r["forward_tx"], stage="claimed",
                                            forward_tx=None)  # fmt: skip
                        return st
                    if not res:
                        back = {"stage": "claimed", "forward_tx": None}
                        if self._reverted("refunds", "key", key, r, "the send", **back):
                            return "stuck"
                        continue
                    self._set("refunds", "key", key, stage="delivered")
                    log.info("refund %s: claimed from the bridge and sent to %s", key, r["dest"])
                    return "delivered"
                else:
                    return st
            return st

    def _gas_for(self, key: str, r: sqlite3.Row, sending: int) -> bool:
        """The treasury's Bittensor EVM account pays the claim's gas: it must hold that (and ``sending``)."""
        if self.route.treasury_bt_balance() >= sending + self.route.finney_gas_wei():
            return True
        note = "waiting: the treasury needs TAO for gas on Bittensor EVM"
        if r["note"] != note:
            self._set("refunds", "key", key, note=note)
            self._alert(
                f"refund {key}: send ~0.02 TAO to the treasury on Bittensor EVM ({self.route.treasury})"
            )
        return False

    # --- background ----------------------------------------------------------------------------------
    def reconcile_refunds(self, older_than_s: float = 600.0) -> list[str]:
        """Accounts refund rows left 'pending' without a live rail refund (a crash between the two databases):
        a rail refund that failed gets its rows finished as failed; a reservation never handed to the rail
        gives the balance back. Returns the keys fixed."""
        fixed = []
        with self._conn() as c:
            rail = {r["key"]: r for r in c.execute("SELECT key, stage, rows, ts FROM refunds")}
        for key, pid, _hk, _cents in self.accounts.pending_refunds():
            r = rail.get(key)
            if (r is not None and r["stage"] in ("failed", "failing")) or (
                r is None and self._reserved_age(key, pid) > older_than_s
            ):
                self.accounts.finish_refund(key, pid, False, None)
                if r is not None and r["stage"] == "failing":
                    self._set("refunds", "key", key, stage="failed")
                fixed.append(key)
        return sorted(set(fixed))

    def _reserved_age(self, key: str, pid: str) -> float:
        ts = self.accounts.refund_ts(key, pid)
        return self._now() - ts if ts is not None else 0.0

    def watch(self) -> dict[str, list[str]]:
        """Look at deposit addresses (paying hotkeys every ``poll_s``, the others every ``idle_poll_s``),
        move conversions and refunds on."""
        try:
            for key in self.reconcile_refunds():
                log.warning(
                    "refund %s: reservation given back (left by a crash between the two ledgers)", key
                )
        except Exception as e:
            log.warning("refund reconcile failed: %s", e)
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
            except SlippageTooHigh as e:  # waits for the pools, without a bound: the owner should know
                log.warning("refund %s waits: %s", key, e)
                self._alert(f"refund {key} waits for the pools: {e}")
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
                "SELECT COUNT(*), COALESCE(SUM(cents), 0) FROM refunds WHERE stage NOT IN "
                "('new', 'buying', 'bought', 'bridging', 'failed')"
            ).fetchone()
            open_in = c.execute(f"SELECT COUNT(*) FROM inbound WHERE stage IN {_INBOUND_OPEN}").fetchone()[0]  # noqa: S608
            open_ref = c.execute(f"SELECT COUNT(*) FROM refunds WHERE stage IN {_REFUND_OPEN}").fetchone()[0]  # noqa: S608
            stuck = c.execute("SELECT COUNT(*) FROM inbound WHERE stage = 'stuck'").fetchone()[0] + c.execute(
                "SELECT COUNT(*) FROM refunds WHERE stage = 'stuck'").fetchone()[0]  # fmt: skip
        return {"deposits_converted": conv[0], "converted_usdc": conv[1] / USDC_UNIT, "refunds": ref[0],
                "refunded_usdc": ref[1] / 100, "in_progress": {"deposits": open_in, "refunds": open_ref},
                "stuck": stuck}  # fmt: skip
