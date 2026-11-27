import math
from pathlib import Path

import pytest

from humanizer_eval.billing import (
    Accounts,
    BillingError,
    BillingService,
    PriceCard,
    quote,
    statement,
)

CARD = PriceCard.load(Path(__file__).parents[2] / "configs" / "price_card.json")
HK = "5Fminer"


def make(tmp_path, now=lambda: 1000.0):
    return BillingService(CARD, Accounts(tmp_path / "b.sqlite"), None, now=now)


# --- prices -----------------------------------------------------------------------------------------
def test_quote_is_an_upper_bound():
    q = quote(CARD, [260] * 750)
    items = {i.item: i for i in q.items}
    # cap = floor(1.3 * 260) = 338 words -> Pangram ceil(3.38) = 4 units x $0.05 per text
    assert items["detector:pangram"].amount_cents == 750 * 20
    assert items["detector:gptzero"].amount_cents == math.ceil(750 * 338 * 0.15 / 1000 * 100)
    assert "payment_fee_estimate" not in items  # paid in TAO: no card processor
    assert q.total_cents == sum(i.amount_cents for i in q.items)
    assert 15_000 < q.total_cents < 30_000  # ~$150-300 for 750 texts of ~260 words
    assert CARD.raw["currency"] == "usdc"


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
def test_payment_idempotent(tmp_path):
    a = Accounts(tmp_path / "a.sqlite")
    assert a.record_payment("tao:0x1", HK, 10_000, 0) is True
    assert a.record_payment("tao:0x1", HK, 10_000, 0) is False
    assert a.balance(HK) == 10_000
    with pytest.raises(BillingError):
        a.record_payment("tao:0x1", "other", 10_000, 0)


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


# --- service --------------------------------------------------------------------------------------------
def test_quote_hold_settle(tmp_path):
    svc = make(tmp_path)
    q = svc.make_quote(HK, [200] * 50)
    assert q["hold_cents"] == q["total_cents"] and svc.amount_due(q["id"]) == q["hold_cents"]
    svc.accounts.record_payment("tao:0xa", HK, q["hold_cents"] + 500, 0)
    assert svc.amount_due(q["id"]) == 0
    svc.start(q["id"], "s1")
    st = statement(CARD, detector_words={"pangram": [100]}, judge_tokens={}, pod_hours=0.1)
    r = svc.finish("s1", st)
    assert r["charged_cents"] == st.total_cents
    assert svc.accounts.available(HK) == q["hold_cents"] + 500 - st.total_cents


def test_expired_quote_cannot_be_held(tmp_path):
    t = [1000.0]
    svc = make(tmp_path, now=lambda: t[0])
    q = svc.make_quote(HK, [200] * 5)
    svc.accounts.record_payment("tao:0xa", HK, 100_000, 0)
    t[0] += 3601
    with pytest.raises(BillingError, match="expired"):
        svc.start(q["id"], "s1")
    assert svc.start(q["id"], "s1", paid_in_time=True)["hold_cents"] == q["hold_cents"]


def test_refunds_need_the_tao_rail(tmp_path):
    with pytest.raises(BillingError, match="not configured"):
        make(tmp_path).refund(HK, "k", "5Fcold")


def test_statement_with_other_card_rejected(tmp_path):
    svc = make(tmp_path)
    other = PriceCard({**CARD.raw, "version": "other"})
    st = statement(other, detector_words={}, judge_tokens={}, pod_hours=0)
    with pytest.raises(BillingError, match="price card"):
        svc.finish("s", st)


def test_quote_with_no_paid_detectors_has_no_detector_lines():
    q = quote(CARD, [200] * 10, [])
    assert not [i for i in q.items if i.item.startswith("detector:")]
    assert {i.item for i in quote(CARD, [200] * 10).items} >= {f"detector:{d}" for d in CARD.detectors}
