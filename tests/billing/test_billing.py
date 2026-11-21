import hashlib
import hmac
import json
import math
import urllib.parse
from pathlib import Path

import httpx
import pytest

from humanizer_eval.billing import (
    Accounts,
    BillingError,
    BillingService,
    PriceCard,
    StripeAPI,
    StripeError,
    quote,
    statement,
    verify_webhook,
)

CARD = PriceCard.load(Path(__file__).parents[2] / "configs" / "price_card.json")
HK = "5Fminer"


class FakeStripe:
    """In-memory Stripe: checkout sessions, test payments (2.9% + 30c fee), refunds, idempotency."""

    def __init__(self):
        self.sessions, self.pis, self.refunds, self.idem = {}, {}, {}, {}
        self.paid = True
        self.down = False  # every call answers 503
        self.calls = []

    def _pi(self, amount, meta):
        pid = f"pi_{len(self.pis) + 1}"
        fee = math.ceil(amount * 0.029) + 30
        self.pis[pid] = {
            "id": pid,
            "amount": amount,
            "status": "succeeded",
            "metadata": meta,
            "latest_charge": {"balance_transaction": {"fee": fee}},
            "refunded": 0,
        }
        return self.pis[pid]

    def handler(self, req: httpx.Request) -> httpx.Response:
        assert req.headers["authorization"].startswith("Basic ")
        path = req.url.path
        form = dict(urllib.parse.parse_qsl(req.content.decode())) if req.method == "POST" else {}
        self.calls.append((req.method, path, form, req.headers.get("idempotency-key")))
        if self.down:
            return httpx.Response(503, json={"error": {"type": "api_error", "message": "down"}})
        key = req.headers.get("idempotency-key")
        if key and key in self.idem:
            return httpx.Response(200, json=self.idem[key])
        if req.method == "POST" and path == "/v1/checkout/sessions":
            sid = f"cs_test_{len(self.sessions) + 1}"
            amount = int(form["line_items[0][price_data][unit_amount]"])
            meta = {"hotkey": form["metadata[hotkey]"], "ref": form["metadata[ref]"]}
            self.sessions[sid] = {
                "amount": amount,
                "meta": meta,
                "ref": form["client_reference_id"],
                "pi": None,
                "expires_at": int(form.get("expires_at", 0)),
                "expired": False,  # tests flip it: Stripe closed the unpaid link
            }
            out = {
                "id": sid,
                "url": f"https://checkout.stripe.test/{sid}",
                "expires_at": int(form.get("expires_at", 0)),
            }
        elif req.method == "GET" and path.startswith("/v1/checkout/sessions/"):
            s = self.sessions[path.rsplit("/", 1)[1]]
            if self.paid and s["pi"] is None:
                s["pi"] = self._pi(s["amount"], s["meta"])
            out = {
                "id": path.rsplit("/", 1)[1],
                "payment_status": "paid" if self.paid else "unpaid",
                "status": "complete" if self.paid else ("expired" if s["expired"] else "open"),
                "url": None
                if self.paid or s["expired"]
                else f"https://checkout.stripe.test/{path.rsplit('/', 1)[1]}",
                "expires_at": s["expires_at"],
                "amount_total": s["amount"],
                "metadata": s["meta"],
                "client_reference_id": s["ref"],
                "payment_intent": s["pi"],
            }
        elif req.method == "GET" and path.startswith("/v1/payment_intents/"):
            out = self.pis[path.rsplit("/", 1)[1]]
        elif req.method == "POST" and path == "/v1/payment_intents":
            out = self._pi(int(form["amount"]), {"hotkey": form["metadata[hotkey]"]})
        elif req.method == "POST" and path == "/v1/refunds":
            pi = self.pis[form["payment_intent"]]
            amt = int(form["amount"])
            if pi["refunded"] + amt > pi["amount"]:
                return httpx.Response(
                    400, json={"error": {"type": "invalid_request_error", "message": "too much"}}
                )
            pi["refunded"] += amt
            rid = f"re_{len(self.refunds) + 1}"
            self.refunds[rid] = (pi["id"], amt)
            out = {"id": rid, "status": "succeeded"}
        else:
            return httpx.Response(404, json={"error": {"type": "invalid_request_error", "message": path}})
        if key:
            self.idem[key] = out
        return httpx.Response(200, json=out)


def make(tmp_path, now=lambda: 1000.0):
    fake = FakeStripe()
    api = StripeAPI("sk_test_x", transport=httpx.MockTransport(fake.handler))
    return BillingService(CARD, Accounts(tmp_path / "b.sqlite"), api, now=now), fake


# --- prices -----------------------------------------------------------------------------------------
def test_quote_is_an_upper_bound_with_fee():
    q = quote(CARD, [260] * 750)
    items = {i.item: i for i in q.items}
    # cap = floor(1.3 * 260) = 338 words -> Pangram ceil(3.38) = 4 units x $0.05 per text
    assert items["detector:pangram"].amount_cents == 750 * 20
    assert items["detector:gptzero"].amount_cents == math.ceil(750 * 338 * 0.15 / 1000 * 100)
    net = q.total_cents - items["payment_fee_estimate"].amount_cents
    gross, fee = CARD.with_payment_fee(net)
    assert q.total_cents == gross and fee == items["payment_fee_estimate"].amount_cents
    assert gross - (0.06 * gross + 35) >= net - 1  # the estimate leaves at least the net amount
    assert 15_000 < q.total_cents < 30_000  # ~$150-300 for 750 texts of ~260 words


def test_statement_from_metered_quantities():
    st = statement(
        CARD,
        detector_words={"pangram": [120, 250], "gptzero": [120, 250]},
        judge_tokens={"openai/gpt-6-luna": (1_000_000, 100_000)},
        pod_hours=0.5,
    )
    items = {i.item: i.amount_cents for i in st.items}
    assert items["detector:pangram"] == 10 + 15  # ceil(1.2)=2, ceil(2.5)=3 units x 5c
    assert items["judge:openai/gpt-6-luna"] == 15  # $0.10 + $0.05
    assert items["lium:h100"] == 125
    assert st.card_sha256 == CARD.sha256 and len(CARD.sha256) == 64
    with pytest.raises(KeyError):
        statement(CARD, detector_words={}, judge_tokens={"x/unknown": (1, 1)}, pod_hours=0)


# --- accounts ---------------------------------------------------------------------------------------
def test_payment_idempotent_and_net_of_fee(tmp_path):
    a = Accounts(tmp_path / "a.sqlite")
    assert a.record_payment("pi_1", HK, 10_000, 320) is True
    assert a.record_payment("pi_1", HK, 10_000, 320) is False
    assert a.balance(HK) == 9_680
    with pytest.raises(BillingError):
        a.record_payment("pi_1", "other", 10_000, 320)


def test_hold_settle_release(tmp_path):
    a = Accounts(tmp_path / "a.sqlite")
    a.record_payment("pi_1", HK, 10_000, 0)
    with pytest.raises(BillingError, match="insufficient"):
        a.open_hold("s1", HK, 20_000)
    a.open_hold("s1", HK, 8_000)
    assert a.available(HK) == 2_000 and a.balance(HK) == 10_000
    assert a.open_hold("s1", HK, 8_000).status == "open"  # idempotent
    h = a.settle("s1", 9_999)  # overrun is capped at the hold
    assert h.charged_cents == 8_000 and a.balance(HK) == 2_000 == a.available(HK)
    assert a.settle("s1", 1).charged_cents == 8_000  # idempotent
    a.open_hold("s2", HK, 1_000)
    a.release("s2")
    assert a.available(HK) == 2_000


def test_refund_newest_first_and_blocked_during_run(tmp_path):
    a = Accounts(tmp_path / "a.sqlite")
    a.record_payment("pi_old", HK, 5_000, 0)
    a.record_payment("pi_new", HK, 3_000, 0)
    a.open_hold("s1", HK, 1_000)
    with pytest.raises(BillingError, match="being evaluated"):
        a.plan_refund("r1", HK)
    a.settle("s1", 1_000)
    plan = a.plan_refund("r1", HK)
    assert plan == [("pi_new", 3_000), ("pi_old", 4_000)]
    assert a.plan_refund("r1", HK) == plan  # idempotent on the key
    a.finish_refund("r1", "pi_new", ok=True)
    a.finish_refund("r1", "pi_old", ok=False)  # failed refund: money stays on the balance
    assert a.balance(HK) == 4_000
    with pytest.raises(BillingError, match="exceeds"):
        a.plan_refund("r2", HK, 5_000)


# --- service with a fake Stripe ---------------------------------------------------------------------
def test_full_flow_checkout_hold_settle_refund(tmp_path):
    svc, fake = make(tmp_path)
    q = svc.make_quote(HK, [260] * 750)
    due = svc.amount_due(q["id"])
    assert due == q["total_cents"]  # empty balance: pay the whole quote incl. the fee estimate
    co = svc.checkout(q["id"], "https://x/ok", "https://x/cancel")
    assert co["url"].startswith("https://checkout.stripe.test/") and co["amount_cents"] == due
    assert svc.confirm_checkout(co["session_id"]) is True
    assert svc.confirm_checkout(co["session_id"]) is False  # webhook + poll never double-credit
    assert svc.accounts.available(HK) >= q["hold_cents"]  # the actual fee was below the estimate
    assert svc.amount_due(q["id"]) == 0
    svc.start(q["id"], "sub-1")
    st = statement(
        CARD,
        detector_words={d: [270] * 750 for d in ("pangram", "gptzero", "zerogpt")},
        judge_tokens={
            "openai/gpt-6-luna": (9_000_000, 2_000_000),
            "google/gemini-3.8-flash": (3_000_000, 900_000),
        },
        pod_hours=0.9,
    )
    res = svc.finish("sub-1", st)
    assert res["charged_cents"] == st.total_cents < res["hold_cents"]
    left = svc.accounts.available(HK)
    assert left == svc.accounts.balance(HK) > 0
    out = svc.refund(HK, "refund-1")
    assert out[0]["status"] == "succeeded" and out[0]["amount_cents"] == left
    assert svc.accounts.balance(HK) == 0
    again = svc.refund(HK, "refund-1")  # same key: the same Stripe refund comes back, none is created
    assert [r["refund"] for r in again] == [r["refund"] for r in out]
    assert len(fake.refunds) == 1 and svc.accounts.balance(HK) == 0


def test_unpaid_session_not_credited_and_expired_quote(tmp_path):
    t = [1000.0]
    svc, fake = make(tmp_path, now=lambda: t[0])
    q = svc.make_quote(HK, [200] * 10)
    fake.paid = False
    co = svc.checkout(q["id"], "https://x/ok", "https://x/c")
    with pytest.raises(BillingError, match="not paid"):
        svc.confirm_checkout(co["session_id"])
    t[0] += 3601
    with pytest.raises(BillingError, match="expired"):
        svc.start(q["id"], "s")


def test_statement_with_other_card_rejected(tmp_path):
    svc, _ = make(tmp_path)
    other = PriceCard({**CARD.raw, "version": "other"})
    st = statement(other, detector_words={}, judge_tokens={}, pod_hours=0)
    with pytest.raises(BillingError, match="price card"):
        svc.finish("s", st)


def test_webhook_signature(tmp_path):
    svc, _ = make(tmp_path)
    q = svc.make_quote(HK, [200] * 10)
    co = svc.checkout(q["id"], "https://x/ok", "https://x/c")
    payload = json.dumps(
        {
            "type": "checkout.session.completed",
            "data": {"object": {"id": co["session_id"], "payment_status": "paid"}},
        }
    ).encode()
    secret, ts = "whsec_test", 1_700_000_000
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    header = f"t={ts},v1={sig}"
    assert verify_webhook(payload, header, secret, now=ts + 10)["type"] == "checkout.session.completed"
    with pytest.raises(StripeError, match="mismatch"):
        verify_webhook(payload + b" ", header, secret, now=ts)
    with pytest.raises(StripeError, match="tolerance"):
        verify_webhook(payload, header, secret, now=ts + 10_000)


def test_live_key_refused_and_errors_do_not_echo_secrets():
    with pytest.raises(StripeError, match="live"):
        StripeAPI("sk_live_abc")
    api = StripeAPI(
        "sk_test_secret123", transport=httpx.MockTransport(lambda r: httpx.Response(401, json={}))
    )
    with pytest.raises(StripeError) as e:
        api.retrieve_payment_intent("pi_1")
    assert "secret123" not in str(e.value)


def test_payment_with_unknown_fee_is_not_credited(tmp_path):
    fake = FakeStripe()
    api = StripeAPI("sk_test_x", transport=httpx.MockTransport(fake.handler))
    svc = BillingService(CARD, Accounts(tmp_path / "b.sqlite"), api, sleep=lambda s: None)
    orig = fake._pi

    def no_fee(amount, meta):
        pi = orig(amount, meta)
        pi["latest_charge"] = {"balance_transaction": None}
        return pi

    fake._pi = no_fee
    with pytest.raises(BillingError, match="fee not available"):
        svc.test_topup(HK, 1_000, ref="r")
    assert svc.accounts.balance(HK) == 0


def test_fee_converted_from_settlement_currency():
    from humanizer_eval.billing.stripe_api import payment_fee_cents

    pi = {
        "currency": "usd",
        "latest_charge": {"balance_transaction": {"fee": 234, "currency": "aed", "exchange_rate": 3.67255}},
    }
    assert payment_fee_cents(pi) == 64  # 234 fils / 3.67255 = 63.7 -> 64 US cents (rounded up)
    pi["latest_charge"]["balance_transaction"] = {"fee": 50, "currency": "usd"}
    assert payment_fee_cents(pi) == 50


def test_quote_with_no_paid_detectors_has_no_detector_lines():
    q = quote(CARD, [200] * 10, [])
    assert not [i for i in q.items if i.item.startswith("detector:")]
    assert {i.item for i in quote(CARD, [200] * 10).items} >= {f"detector:{d}" for d in CARD.detectors}
