"""Billing flow for one submission (D-80): quote -> top-up -> hold -> settle -> refund on request.

q = svc.make_quote(hotkey, input_words)          # upper bound, valid for card["quote_ttl_seconds"]
pay = svc.checkout(q["id"], success_url, cancel_url)   # Stripe Checkout for what the balance lacks
svc.confirm_checkout(session_id)                 # webhook or poll: credit net of the actual fee
svc.start(q["id"], submission_id)                # hold the quote; refunds wait until settled
svc.finish(submission_id, statement)             # charge metered cost (<= hold), free the rest
svc.refund(hotkey, key)                          # unused balance back to the original payments
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any

from humanizer_eval.billing.accounts import Accounts, BillingError
from humanizer_eval.billing.prices import PriceCard, Statement, quote
from humanizer_eval.billing.stripe_api import StripeAPI, StripeError, payment_fee_cents, verify_webhook

STRIPE_MIN_CENTS = 50
STRIPE_MIN_EXPIRY_S = 31 * 60  # Checkout: expires_at at least 30 min ahead
STRIPE_MAX_EXPIRY_S = 24 * 3600 - 60  # ... and at most 24 h


class BillingService:
    def __init__(
        self,
        card: PriceCard,
        accounts: Accounts,
        stripe: StripeAPI | None = None,
        now=time.time,
        sleep=time.sleep,
    ):
        self.card = card
        self.accounts = accounts
        self.stripe = stripe
        self._now = now
        self._sleep = sleep

    def _stripe(self) -> StripeAPI:
        if self.stripe is None:
            raise BillingError("Stripe is not configured")
        return self.stripe

    # --- quote and top-up -------------------------------------------------------------------------
    def make_quote(
        self, hotkey: str, input_words: Sequence[int], detectors: Sequence[str] | None = None
    ) -> dict:
        st = quote(self.card, input_words, detectors)
        fee = next(i.amount_cents for i in st.items if i.item == "payment_fee_estimate")
        hold = st.total_cents - fee  # the fee is paid at top-up, the hold covers the evaluation itself
        qid = f"q_{uuid.uuid4().hex[:20]}"
        expires = self._now() + float(self.card.raw.get("quote_ttl_seconds", 3600))
        body = {**st.to_dict(), "hotkey": hotkey, "hold_cents": hold, "expires_ts": expires}
        self.accounts.save_quote(qid, hotkey, hold, expires, body)
        return {"id": qid, **body}

    def amount_due(self, quote_id: str) -> int:
        """Gross amount to pay now so the balance covers the quote's hold (0 if it already does)."""
        q = self._quote(quote_id, check_expiry=False)
        need = q["hold_cents"] - self.accounts.available(q["hotkey"])
        if need <= 0:
            return 0
        gross, _ = self.card.with_payment_fee(need)
        return max(gross, STRIPE_MIN_CENTS)

    def checkout(self, quote_id: str, success_url: str, cancel_url: str) -> dict[str, Any]:
        q = self._quote(quote_id)
        due = self.amount_due(quote_id)
        if due == 0:
            return {"amount_cents": 0, "url": None, "session_id": None}
        # the link works as long as the quote (Stripe allows 30 min .. 24 h from now)
        now = self._now()
        expires_at = int(min(max(q["expires_ts"], now + STRIPE_MIN_EXPIRY_S), now + STRIPE_MAX_EXPIRY_S))
        s = self._stripe().create_checkout_session(
            amount_cents=due,
            hotkey=q["hotkey"],
            ref=quote_id,
            success_url=success_url,
            cancel_url=cancel_url,
            description=f"Evaluation credit for hotkey {q['hotkey'][:12]}..., quote {quote_id}",
            expires_at=expires_at,
        )
        return {
            "amount_cents": due,
            "url": s.get("url"),
            "session_id": s["id"],
            "expires_at": s.get("expires_at") or expires_at,
        }

    def check_checkout(self, session_id: str) -> dict[str, Any]:
        """Ask Stripe about a Checkout Session and credit it if it is paid (idempotent).

        ``{"status": open|complete|expired, "paid": bool, "credited": bool, "url": link while open,
        "expires_at": ts}``. Raises BillingError when it is paid but the fee is not known yet (retry)."""
        s = self._stripe().retrieve_checkout_session(session_id)
        paid = s.get("payment_status") == "paid"
        out = {
            "status": s.get("status"),
            "paid": paid,
            "credited": False,
            "url": s.get("url") if s.get("status") == "open" else None,
            "expires_at": s.get("expires_at"),
        }
        if paid:
            self._credit_session(session_id, s)
            out["credited"] = True
        return out

    def confirm_checkout(self, session_id: str) -> bool:
        """Credit a paid Checkout Session (idempotent). False if already credited."""
        s = self._stripe().retrieve_checkout_session(session_id)
        if s.get("payment_status") != "paid":
            raise BillingError(f"session {session_id} is not paid ({s.get('payment_status')})")
        return self._credit_session(session_id, s)

    def _credit_session(self, session_id: str, s: dict[str, Any]) -> bool:
        pi = s.get("payment_intent")
        if not isinstance(pi, dict):
            raise BillingError("payment intent not expanded")
        fee = self._wait_fee(pi)
        hotkey = (s.get("metadata") or {}).get("hotkey")
        if not hotkey:
            raise BillingError("session has no hotkey")
        return self.accounts.record_payment(
            pi["id"],
            hotkey,
            int(s["amount_total"]),
            fee,
            {"session": session_id, "ref": s.get("client_reference_id")},
        )

    def _wait_fee(self, pi: dict, tries: int = 10, delay: float = 1.0) -> int:
        """The processor's actual fee; Stripe creates the balance transaction shortly after the charge.
        Never credits a payment with an unknown fee (crediting gross would refund money we never got)."""
        for attempt in range(tries):
            fee = payment_fee_cents(pi)
            if fee is not None:
                return fee
            if attempt < tries - 1:
                self._sleep(delay)
                pi = self._stripe().retrieve_payment_intent(pi["id"])
        raise BillingError("processor fee not available yet; retry shortly")

    def handle_webhook(self, payload: bytes, sig_header: str, secret: str) -> str:
        event = verify_webhook(payload, sig_header, secret)
        if event.get("type") in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
            obj = event["data"]["object"]
            if obj.get("payment_status") == "paid":
                return "credited" if self.confirm_checkout(obj["id"]) else "duplicate"
        return "ignored"

    def test_topup(self, hotkey: str, amount_cents: int, ref: str) -> dict[str, Any]:
        """Test mode only: pay with Stripe's test card and credit the balance."""
        pi = self._stripe().create_test_payment(amount_cents=amount_cents, hotkey=hotkey, ref=ref)
        if pi.get("status") != "succeeded":
            raise BillingError(f"test payment not succeeded: {pi.get('status')}")
        fee = self._wait_fee(pi)
        self.accounts.record_payment(pi["id"], hotkey, int(pi["amount"]), fee, {"ref": ref, "test": True})
        return {"payment_intent": pi["id"], "gross_cents": int(pi["amount"]), "fee_cents": fee}

    # --- evaluation ---------------------------------------------------------------------------------
    def _quote(self, quote_id: str, check_expiry: bool = True) -> dict:
        q = self.accounts.get_quote(quote_id)
        if q is None:
            raise BillingError(f"unknown quote {quote_id}")
        if check_expiry and self._now() > q["expires_ts"]:
            raise BillingError("quote expired; request a new one")
        return q

    def start(self, quote_id: str, submission_id: str, *, paid_in_time: bool = False) -> dict:
        """Hold the quote. ``paid_in_time``: its Checkout was paid before the link expired, so the quote
        holds even if the payment is confirmed after the quote's own expiry."""
        q = self._quote(quote_id, check_expiry=not paid_in_time)
        h = self.accounts.open_hold(submission_id, q["hotkey"], q["hold_cents"])
        return {"submission_id": h.submission_id, "hold_cents": h.amount_cents}

    def finish(self, submission_id: str, statement: Statement) -> dict:
        if statement.card_sha256 != self.card.sha256:
            raise BillingError("statement priced with another price card")
        h = self.accounts.settle(submission_id, statement.total_cents, statement.to_dict())
        return {
            "submission_id": submission_id,
            "hold_cents": h.amount_cents,
            "charged_cents": h.charged_cents,
            "returned_cents": h.amount_cents - h.charged_cents,
            "statement_total_cents": statement.total_cents,
        }

    def fail(self, submission_id: str) -> dict:
        h = self.accounts.release(submission_id)
        return {"submission_id": submission_id, "status": h.status, "charged_cents": 0}

    # --- refunds ------------------------------------------------------------------------------------
    def refund(self, hotkey: str, key: str, amount_cents: int | None = None) -> list[dict]:
        """Refund (default: all available) to the original payments; safe to call again with the same key."""
        plan = self.accounts.plan_refund(key, hotkey, amount_cents)
        out = []
        for pid, amt in plan:
            try:
                r = self._stripe().create_refund(
                    payment_intent=pid, amount_cents=amt, idem=f"refund:{key}:{pid}", hotkey=hotkey
                )
                ok = r.get("status") in ("succeeded", "pending")
                self.accounts.finish_refund(key, pid, ok, r.get("id"))
                out.append(
                    {
                        "payment_intent": pid,
                        "amount_cents": amt,
                        "status": r.get("status"),
                        "refund": r.get("id"),
                    }
                )
            except StripeError as e:
                self.accounts.finish_refund(key, pid, False)
                out.append({"payment_intent": pid, "amount_cents": amt, "status": "failed", "error": str(e)})
        return out
