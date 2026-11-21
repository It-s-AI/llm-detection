"""End-to-end billing check against Stripe TEST mode (no real money): quote -> checkout link -> test-card
payment -> hold -> settle -> refund of the rest. Refuses live keys.
Usage: python scripts/billing_live_test.py
"""

from __future__ import annotations

import json
import time
import uuid

from humanizer_eval.billing import Accounts, BillingService, PriceCard, StripeAPI, statement
from humanizer_eval.config import get_settings
from humanizer_eval.paths import var_dir


def main() -> int:
    s = get_settings()
    api = StripeAPI(s.stripe_secret_key)  # raises on a live key
    if not api.test_mode:
        raise SystemExit("needs a Stripe test key")
    card = PriceCard.load("configs/price_card.json")
    svc = BillingService(card, Accounts(var_dir() / "billing" / "live_test.sqlite"), api)
    hotkey = f"5Ftest{uuid.uuid4().hex[:8]}"
    q = svc.make_quote(hotkey, [200] * 10)
    co = svc.checkout(q["id"], "https://example.com/paid", "https://example.com/cancel")
    due = svc.amount_due(q["id"])
    pay = svc.test_topup(hotkey, due, ref=f"{q['id']}:testcard")
    after_pay = svc.accounts.available(hotkey)
    sub = f"sub-{uuid.uuid4().hex[:8]}"
    svc.start(q["id"], sub)
    st = statement(
        card,
        detector_words={d: [210] * 10 for d in ("pangram", "gptzero", "zerogpt")},
        judge_tokens={"openai/gpt-6-luna": (60_000, 12_000), "google/gemini-3.8-flash": (30_000, 8_000)},
        pod_hours=0.4,
        meta={"submission_id": sub},
    )
    fin = svc.finish(sub, st)
    before_refund = svc.accounts.balance(hotkey)
    ref = svc.refund(hotkey, f"refund-{sub}")
    time.sleep(1)
    pi = api.retrieve_payment_intent(pay["payment_intent"])
    charge = pi.get("latest_charge") or {}
    print(
        json.dumps(
            {
                "quote_total_cents": q["total_cents"],
                "quote_hold_cents": q["hold_cents"],
                "checkout_url_created": bool(co.get("url")),
                "paid_gross_cents": pay["gross_cents"],
                "stripe_fee_cents": pay["fee_cents"],
                "available_after_payment": after_pay,
                "settle": fin,
                "statement_items": {i.item: i.amount_cents for i in st.items},
                "balance_before_refund": before_refund,
                "refunds": ref,
                "balance_after_refund": svc.accounts.balance(hotkey),
                "stripe_charge_amount_refunded": charge.get("amount_refunded"),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
