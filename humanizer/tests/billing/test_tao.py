"""Payments in TAO kept as USDC (billing/tao.py), on an in-memory Bittensor EVM + bridge + Base."""

import contextlib
import hashlib

import pytest

from humanizer_eval.billing import Accounts, BillingService
from humanizer_eval.billing.evm import SignedTx
from humanizer_eval.billing.tao import (
    CENT_USDC,
    WEI,
    TaoConfig,
    TaoRail,
    h160_mirror,
    ss58,
    ss58_pubkey,
)
from humanizer_eval.intake.client import SeedSigner

from .test_billing import CARD

COLD = SeedSigner(b"\x11" * 32).ss58_address  # where the miner wants its refund (any address)
HK = SeedSigner(b"\x12" * 32).ss58_address
PRICE = 200.0  # USDC per TAO (small trade)
ETH = 2000.0
BRIDGE_OUT_COST = 5 * 10**15  # 0.005 TAO kept back by a deposit for the bridge
BRIDGE_IN_FEE = 2 * 10**15  # 0.002 ETH for a refund's CCIP message


class FakeRoute:
    """Deposits on Bittensor EVM, a bridge that delivers after ``lag`` looks, Base pools paying ``1 - impact``
    of the small-trade price. Transactions take effect when sent and are mined after ``mine_after`` looks."""

    def __init__(self):
        self.bt = {}  # deposit h160 -> wei
        self.usdc = 0  # treasury on Base
        self.tao = 0
        self.impact = 0.003
        self.txs, self.pending = {}, {}
        self.mine_after = 0
        self.lag = 0
        self.revert_next = False
        self.paused = False
        self.messages = {}  # message id -> [looks left, wei, pubkey or None, delivered]
        self.delivered = []  # (pubkey, wei) delivered on Finney
        self.n = 0

    def address_of(self, key):
        return "0x" + hashlib.sha256(key).hexdigest()[:40]

    def deposit_balance(self, a):
        return self.bt.get(a, 0)

    def bridge_ready(self):
        return not self.paused

    def bridge_out_cost(self, amount):
        return BRIDGE_OUT_COST

    def price(self):
        return PRICE

    def eth_usd(self):
        return ETH

    def quote_sell(self, wei):
        return int(wei / WEI * PRICE * (1 - self.impact) * 10**6)

    def quote_buy(self, usdc):
        return int(usdc / 10**6 / PRICE * (1 - self.impact) * WEI)

    def bridge_in_fee(self, wei, pub):
        return BRIDGE_IN_FEE

    def _tx(self, chain, **kw):
        self.n += 1
        h = f"0x{self.n:064x}"
        self.txs[h] = {"chain": chain, **kw}
        return SignedTx(h, "raw")

    def bridge_out_tx(self, key, amount):
        return self._tx("bittensor", kind="out", frm=self.address_of(key), wei=amount)

    def sell_tx(self, wei, min_usdc):
        return self._tx("base", kind="sell", wei=wei, min=min_usdc)

    def buy_tx(self, usdc, min_wei):
        return self._tx("base", kind="buy", usdc=usdc, min=min_wei)

    def bridge_in_tx(self, wei, pub):
        return self._tx("base", kind="in", wei=wei, pub=pub)

    def send(self, chain, tx):
        t = self.txs[tx.hash]
        assert t["chain"] == chain
        if "result" in t:
            return
        ok, out = not self.revert_next, {"usdc_in": 0, "tao_in": 0}
        self.revert_next = False
        if ok and t["kind"] == "out":
            ok = self.bt.get(t["frm"], 0) >= t["wei"] + BRIDGE_OUT_COST
            if ok:
                self.bt[t["frm"]] -= t["wei"] + BRIDGE_OUT_COST
                out["message_id"] = f"0xm{self.n}"
                self.messages[out["message_id"]] = [self.lag, t["wei"], None, False]
        elif ok and t["kind"] == "sell":
            got = self.quote_sell(t["wei"])
            ok = got >= t["min"] and self.tao >= t["wei"]
            if ok:
                self.tao -= t["wei"]
                self.usdc += got
                out["usdc_in"] = got
        elif ok and t["kind"] == "buy":
            got = self.quote_buy(t["usdc"])
            ok = got >= t["min"] and self.usdc >= t["usdc"]
            if ok:
                self.usdc -= t["usdc"]
                self.tao += got
                out["tao_in"] = got
        elif ok and t["kind"] == "in":
            ok = self.tao >= t["wei"]
            if ok:
                self.tao -= t["wei"]
                out["message_id"] = f"0xr{self.n}"
                self.messages[out["message_id"]] = [self.lag, t["wei"], t["pub"], False]
        t["result"] = {"ok": ok, **out}
        self.pending[tx.hash] = self.mine_after

    def _mined(self, h):
        if h not in self.pending:
            return None
        if self.pending[h] > 0:
            self.pending[h] -= 1
            return None
        return self.txs[h]["result"]

    def bridge_out_result(self, h):
        r = self._mined(h)
        return None if r is None else {"ok": r["ok"], "message_id": r.get("message_id"), "from_block": 1}

    def swap_result(self, h):
        return self._mined(h)

    def bridge_in_result(self, h):
        r = self._mined(h)
        return None if r is None else {"ok": r["ok"], "message_id": r.get("message_id"), "from_block": 1}

    def _deliver(self, mid):
        m = self.messages[mid]
        if m[0] > 0:
            m[0] -= 1
            return None
        if not m[3]:
            m[3] = True
            if m[2] is None:
                self.tao += m[1]  # arrived on Base
            else:
                self.delivered.append((m[2], m[1]))  # arrived on Finney
        return True

    def arrived_on_base(self, mid, from_block):
        return self._deliver(mid)

    def delivered_on_finney(self, mid, from_block):
        return self._deliver(mid)


def rail(tmp_path, route=None, **cfg):
    acc = Accounts(tmp_path / "b.sqlite")
    route = route or FakeRoute()
    t = [1_800_000_000.0]
    r = TaoRail(tmp_path / "tao.sqlite", route, b"m" * 32, acc, TaoConfig(**cfg), now=lambda: t[0])
    return r, route, acc, t


def send(r, route, hotkey, tao):
    d = r.deposit(hotkey)
    route.bt[d["evm_address"]] = route.bt.get(d["evm_address"], 0) + int(tao * WEI)


def settle_all(r, hotkey, n=10):
    got = 0
    for _ in range(n):
        with contextlib.suppress(Exception):
            got += r.collect(hotkey)
    return got


def test_one_deposit_address_per_hotkey_mirroring_its_evm_account(tmp_path):
    r, _, _, _ = rail(tmp_path)
    d = r.deposit(HK)
    assert d["address"] == ss58(h160_mirror(d["evm_address"]))
    assert r.deposit(HK) == d  # the same address again
    assert r.deposit(SeedSigner(b"\x13" * 32).ss58_address)["address"] != d["address"]


def test_h160_mirror_is_bittensors_mapping():
    assert h160_mirror("0x" + "00" * 20) == hashlib.blake2b(b"evm:" + bytes(20), digest_size=32).digest()


def test_tao_asked_covers_the_amount_the_buffer_and_the_bridge(tmp_path):
    r, _, _, _ = rail(tmp_path)
    wei = r.tao_for_cents(16_500)  # 165 USDC
    assert wei / WEI == pytest.approx(165 * 1.05 / PRICE + BRIDGE_OUT_COST / WEI, abs=1e-4)


def test_each_deposit_is_bridged_whole_swapped_and_credited_in_usdc(tmp_path):
    r, route, acc, _ = rail(tmp_path)
    route.lag = 2
    send(r, route, HK, 1.0)
    assert r.collect(HK) == 0  # bridging
    assert r.pending_cents(HK) > 0
    cents = settle_all(r, HK)
    net = WEI - BRIDGE_OUT_COST
    assert cents == int(net / WEI * PRICE * (1 - route.impact) * 100)
    assert acc.balance(HK) == cents and route.usdc // CENT_USDC == cents and r.pending_cents(HK) == 0
    assert r.inbound(HK)[0]["stage"] == "done" and r.inbound(HK)[0]["usdc"] == f"{cents / 100:.2f}"
    send(r, route, HK, 0.5)  # a second deposit is its own conversion
    assert settle_all(r, HK) > 0 and len(r.inbound(HK)) == 2


def test_dust_and_a_paused_bridge_leave_the_tao_on_the_deposit(tmp_path):
    r, route, _acc, _ = rail(tmp_path)
    send(r, route, HK, 0.012)  # minus the bridge's cost: under 0.01 TAO
    assert settle_all(r, HK) == 0 and route.bt[r.deposit(HK)["evm_address"]] == int(0.012 * WEI)
    send(r, route, HK, 1.0)
    route.paused = True
    with pytest.raises(Exception, match="paused"):
        r.collect(HK)
    route.paused = False
    assert settle_all(r, HK) > 0


def test_a_bad_price_makes_the_swap_wait(tmp_path):
    r, route, acc, _ = rail(tmp_path)
    send(r, route, HK, 1.0)
    route.impact = 0.05
    assert settle_all(r, HK) == 0 and acc.balance(HK) == 0
    assert route.tao > 0  # arrived on Base, waits there
    route.impact = 0.003
    assert settle_all(r, HK) > 0


def test_restart_mid_way_credits_once(tmp_path):
    r, route, acc, t = rail(tmp_path)
    route.mine_after = 1
    send(r, route, HK, 1.0)
    r.collect(HK)
    r2 = TaoRail(tmp_path / "tao.sqlite", route, b"m" * 32, acc, TaoConfig(), now=lambda: t[0])
    got = settle_all(r2, HK)
    assert got > 0 and acc.balance(HK) == got and settle_all(r2, HK) == 0


def test_reverted_swap_is_retried(tmp_path):
    r, route, _acc, _ = rail(tmp_path)
    send(r, route, HK, 1.0)
    r.collect(HK)  # bridged; the arrival is seen on the next step, then the swap
    route.revert_next = True
    assert settle_all(r, HK) > 0


def _billing(tmp_path, **cfg):
    r, route, acc, t = rail(tmp_path, **cfg)
    return BillingService(CARD, acc, r, now=lambda: t[0]), r, route, acc


def test_refund_at_the_current_rate_to_the_coldkey_minus_the_way_back(tmp_path):
    b, r, route, acc = _billing(tmp_path)
    send(r, route, HK, 1.0)
    credited = settle_all(r, HK)
    cost = BRIDGE_IN_FEE / WEI * ETH + 0.05  # the CCIP fee in ETH at its rate + Base gas
    assert float(b.refund_quote(HK, COLD, 5_000)["cost_usdc"]) == pytest.approx(cost, abs=0.01)
    b.refund(HK, "k-1", COLD, 5_000)  # 50 USDC
    for _ in range(5):
        r.drive_refund("k-1")
    st = r.refund_state("k-1")
    assert st["stage"] == "delivered" and st["to"] == COLD
    ((pub, wei),) = route.delivered
    assert pub == ss58_pubkey(COLD)
    assert wei / WEI == pytest.approx((50 - cost) / PRICE * (1 - route.impact), rel=1e-4)
    assert st["tao_sent"] == pytest.approx(wei / WEI)
    assert acc.available(HK) == credited - 5_000
    assert b.refund(HK, "k-1", COLD, 5_000)["stage"] == "delivered"  # the same key: nothing new
    assert len(route.delivered) == 1


def test_refund_too_small_for_the_way_back_fails_and_keeps_the_money(tmp_path):
    b, r, route, acc = _billing(tmp_path)
    send(r, route, HK, 0.5)
    credited = settle_all(r, HK)
    st = b.refund(HK, "k-small", COLD, 300)  # 3 USDC < the 4 USDC bridge fee
    assert st["stage"] == "failed" and acc.available(HK) == credited and route.delivered == []


def test_refund_waits_while_a_hold_is_open(tmp_path):
    b, r, route, acc = _billing(tmp_path)
    send(r, route, HK, 1.0)
    settle_all(r, HK)
    acc.open_hold("s1", HK, 1_000)
    with pytest.raises(Exception, match="being evaluated"):
        b.refund(HK, "k-2", COLD)


def test_refund_survives_interruptions(tmp_path):
    b, r, route, acc = _billing(tmp_path)
    send(r, route, HK, 1.0)
    credited = settle_all(r, HK)
    route.mine_after, route.lag = 1, 1
    st = b.refund(HK, "k-3", COLD)  # everything available
    assert st["stage"] in ("buying", "bridging")
    for _ in range(10):
        r.watch()
    assert r.refund_state("k-3")["stage"] == "delivered" and len(route.delivered) == 1
    assert acc.available(HK) == 0 and credited > 0


def test_intake_submission_paid_in_tao(tmp_path):
    from tests.intake.test_intake import URL, World, _sha

    w = World(tmp_path)
    route = FakeRoute()
    w.billing.tao = TaoRail(tmp_path / "tao.sqlite", route, b"m" * 32, w.billing.accounts, TaoConfig(),
                            now=lambda: w.t[0])  # fmt: skip
    assert w.svc.info()["payment_methods"] == ["tao"]
    out = w.svc.submit(HK, bundle_url=URL, bundle_sha256=_sha(HK), bundle_bytes=10)
    assert out["state"] == "awaiting_payment"
    pay = out["payment"]
    assert pay["method"] == "tao" and pay["due_usdc"] == out["quote"]["hold_usdc"]
    route.bt[pay["evm_address"]] = int((float(pay["amount_tao"]) + 0.5) * WEI)  # more than due
    w.svc.paid(HK, out["submission_id"])  # bridge sent
    for _ in range(4):
        w.t[0] += 60
        w.svc.enqueue_ready()  # the background pass: arrival, swap, credit, queue
    assert w.queue.get(out["submission_id"]).state == "queued"
    assert w.worker.run_once()["state"] == "done"
    assert float(w.svc.balance(HK)["available_usdc"]) > 0  # the buffer and what was not charged stay
    with pytest.raises(ValueError, match="ss58"):
        w.svc.refund(HK, "all-of-it", "not-an-address")
    res = w.svc.refund(HK, "all-of-it", COLD)
    for _ in range(4):
        w.svc.enqueue_ready()
    assert w.svc.refund_status(HK, "all-of-it")["stage"] == "delivered"
    assert route.delivered[0][0] == ss58_pubkey(COLD) and res["refund"]["to"] == COLD
    assert w.svc.balance(HK)["available_usdc"] == "0.00"
