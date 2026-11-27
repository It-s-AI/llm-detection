"""Billing flow for one submission (D-80, D-82): quote -> pay in TAO -> hold -> settle -> refund on request.

q = svc.make_quote(hotkey, input_words)        # upper bound in USDC, valid for card["quote_ttl_seconds"]
svc.deposit(hotkey)                            # the hotkey's TAO deposit address (billing/tao.py)
svc.payment(q["id"])                           # what is missing for the hold, in USDC and in TAO now
svc.collect(hotkey)                            # TAO that arrived -> bridged, swapped, credited in USDC
svc.start(q["id"], submission_id)              # hold the quote; refunds wait until it is settled
svc.finish(submission_id, statement)           # charge the metered cost (<= hold), free the rest
svc.refund(hotkey, key, to)                    # USDC -> TAO at the current rate, to any ss58 address

Balances are cents of USDC (accounts.py); every payment is a TAO deposit converted to USDC.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any

from humanizer_eval.billing.accounts import Accounts, BillingError
from humanizer_eval.billing.prices import PriceCard, Statement, card_archive, quote
from humanizer_eval.billing.tao import WEI, SlippageTooHigh, TaoRail, usdc


class BillingService:
    def __init__(self, card: PriceCard, accounts: Accounts, tao: TaoRail | None = None, now=time.time):
        self.card = card
        self.accounts = accounts
        self.tao = tao
        self._now = now

    def _rail(self) -> TaoRail:
        if self.tao is None:
            raise BillingError("payments in TAO are not configured here")
        return self.tao

    # --- quote and payment --------------------------------------------------------------------------
    def make_quote(
        self, hotkey: str, input_words: Sequence[int], detectors: Sequence[str] | None = None
    ) -> dict:
        st = quote(self.card, input_words, detectors)
        qid = f"q_{uuid.uuid4().hex[:20]}"
        expires = self._now() + float(self.card.raw.get("quote_ttl_seconds", 3600))
        body = {**st.to_dict(), "hotkey": hotkey, "hold_cents": st.total_cents, "expires_ts": expires}
        self.accounts.save_quote(qid, hotkey, st.total_cents, expires, body)
        return {"id": qid, **body}

    def amount_due(self, quote_id: str) -> int:
        """Cents of USDC still missing on the balance for the quote's hold (0 if it is covered)."""
        q = self._quote(quote_id, check_expiry=False)
        return max(0, q["hold_cents"] - self.accounts.available(q["hotkey"]))

    def deposit(self, hotkey: str) -> dict[str, Any]:
        return self._rail().deposit(hotkey)

    def payment(self, quote_id: str) -> dict[str, Any]:
        """How to cover the quote: the deposit address and the TAO to send at the current rate."""
        q = self._quote(quote_id, check_expiry=False)
        due = self.amount_due(quote_id)
        out: dict[str, Any] = {"method": "tao", **self.deposit(q["hotkey"]), "due_usdc": usdc(due),
                               "expires_ts": q["expires_ts"]}  # fmt: skip
        if due > 0:
            out["amount_tao"] = f"{self._rail().tao_for_cents(due) / WEI:.4f}"
            out["buffer"] = self._rail().cfg.buffer
        return out

    def collect(self, hotkey: str) -> int:
        """Convert TAO that reached the hotkey's deposit address; cents credited by this call."""
        return self._rail().collect(hotkey)

    # --- evaluation ---------------------------------------------------------------------------------
    def _quote(self, quote_id: str, check_expiry: bool = True) -> dict:
        q = self.accounts.get_quote(quote_id)
        if q is None:
            raise BillingError(f"unknown quote {quote_id}")
        if check_expiry and self._now() > q["expires_ts"]:
            raise BillingError("quote expired; request a new one")
        return q

    def start(self, quote_id: str, submission_id: str, *, paid_in_time: bool = False) -> dict:
        """Hold the quote. ``paid_in_time``: the TAO arrived within the window even if its conversion
        finished after the quote's own expiry."""
        q = self._quote(quote_id, check_expiry=not paid_in_time)
        h = self.accounts.open_hold(submission_id, q["hotkey"], q["hold_cents"])
        sha = q.get("card_sha256") or (q.get("body") or {}).get("card_sha256")
        if sha:  # the run is settled with the card it was quoted with
            self.accounts.set_hold_card(submission_id, sha)
        return {"submission_id": h.submission_id, "hold_cents": h.amount_cents}

    def card_for(self, submission_id: str) -> PriceCard:
        """The card the submission was quoted with (settled with it, whatever card is current); the current
        card when the quote's card is not known (a hold from before cards were pinned)."""
        sha = self.accounts.hold_card(submission_id)
        if sha is None or sha == self.card.sha256:
            return self.card
        card = card_archive().get(sha)
        if card is None:
            raise BillingError(
                f"the price card {sha[:12]} of {submission_id}'s quote is not in configs/price_cards"
            )
        return card

    def finish(self, submission_id: str, statement: Statement) -> dict:
        if statement.card_sha256 != self.card_for(submission_id).sha256:
            raise BillingError("statement priced with another price card than the quote's")
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

    # --- refunds --------------------------------------------------------------------------------------
    def refund_quote(self, hotkey: str, to: str, amount_cents: int | None = None) -> dict[str, Any]:
        """What a refund would send now (no money moves)."""
        rail = self._rail()
        cents = self.accounts.available(hotkey) if amount_cents is None else amount_cents
        if cents <= 0:
            raise BillingError("nothing to refund")
        cost = rail.refund_cost_usdc(cents, to)
        net = cents * 10_000 - cost
        tao = rail.route.quote_buy(net) / WEI if net > 0 else 0.0
        return {"amount_usdc": usdc(cents), "cost_usdc": f"{cost / 1e6:.2f}", "tao_estimate": round(tao, 4),
                "to": to}  # fmt: skip

    def refund(self, hotkey: str, key: str, to: str, amount_cents: int | None = None) -> dict[str, Any]:
        """USDC of the balance (default: all available) back in TAO to ``to`` (any ss58 address),
        at the rate of now; the cost of the way back comes out of it. Safe to call again with the key."""
        rail = self._rail()
        plan = self.accounts.plan_refund(key, hotkey, amount_cents)
        cents = sum(a for _, a in plan)
        rail.start_refund(key, hotkey, cents, to, plan)
        try:
            rail.drive_refund(key)
        except SlippageTooHigh:
            pass  # waits; the background pass continues it
        except Exception as e:  # RPC trouble: the background pass continues it
            import logging

            logging.getLogger(__name__).warning("refund %s waits: %s", key, e)
        return rail.refund_state(key) or {}

    def housekeeping(self) -> dict[str, list[str]]:
        """Background pass: deposits converted, conversions and refunds moved on."""
        return {} if self.tao is None else self.tao.watch()
