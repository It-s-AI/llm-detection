"""Minimal Stripe REST client (Checkout, PaymentIntents, Refunds, webhook signatures) over httpx.

No SDK dependency: Stripe's API is form-encoded POST/GET with a bearer secret key. Live keys are refused
unless ``allow_live=True`` (the project runs on test keys until launch). Idempotency keys are sent on
every money-moving call, so a retried request never charges or refunds twice.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import time
from typing import Any
from urllib.parse import urlencode

import httpx

API = "https://api.stripe.com"


class StripeError(RuntimeError):
    pass


def _flatten(data: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for k, v in data.items():
        key = f"{prefix}[{k}]" if prefix else str(k)
        if isinstance(v, dict):
            out += _flatten(v, key)
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if isinstance(x, dict):
                    out += _flatten(x, f"{key}[{i}]")
                else:
                    out.append((f"{key}[{i}]", str(x)))
        elif isinstance(v, bool):
            out.append((key, "true" if v else "false"))
        elif v is not None:
            out.append((key, str(v)))
    return out


class StripeAPI:
    def __init__(
        self,
        secret_key: str,
        *,
        allow_live: bool = False,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ):
        if not secret_key:
            raise StripeError("STRIPE_SECRET_KEY is not set")
        if secret_key.startswith(("sk_live_", "rk_live_")) and not allow_live:
            raise StripeError("live Stripe key refused (allow_live=False)")
        self.test_mode = secret_key.startswith(("sk_test_", "rk_test_"))
        self._http = httpx.Client(
            base_url=API,
            auth=(secret_key, ""),
            timeout=timeout,
            transport=transport,
            headers={"Stripe-Version": "2024-06-20"},
        )

    def _req(
        self, method: str, path: str, data: dict | None = None, idem: str | None = None
    ) -> dict[str, Any]:
        headers = {"Idempotency-Key": idem} if idem else {}
        if method == "GET":
            r = self._http.get(path, params=_flatten(data or {}), headers=headers)
        else:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            r = self._http.post(path, content=urlencode(_flatten(data or {})), headers=headers)
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code >= 400:
            err = body.get("error", {}) if isinstance(body, dict) else {}
            # never echo request data or headers: only Stripe's own error type/code/message
            raise StripeError(
                f"stripe {r.status_code} {err.get('type')}/{err.get('code')}: {err.get('message')}"
            )
        return body

    # --- checkout ---------------------------------------------------------------------------------
    def create_checkout_session(
        self,
        *,
        amount_cents: int,
        hotkey: str,
        ref: str,
        success_url: str,
        cancel_url: str,
        description: str,
        expires_at: int | None = None,
    ) -> dict[str, Any]:
        """Hosted Checkout, cards only (a card payment is final at once: no "processing" state)."""
        extra = {} if expires_at is None else {"expires_at": int(expires_at)}
        return self._req(
            "POST",
            "/v1/checkout/sessions",
            {
                "mode": "payment",
                "payment_method_types": ["card"],
                **extra,
                "success_url": success_url,
                "cancel_url": cancel_url,
                "client_reference_id": ref,
                "line_items": [
                    {
                        "quantity": 1,
                        "price_data": {
                            "currency": "usd",
                            "unit_amount": amount_cents,
                            "product_data": {
                                "name": "SN32 humanizer evaluation credit",
                                "description": description,
                            },
                        },
                    }
                ],
                "metadata": {"hotkey": hotkey, "ref": ref},
                "payment_intent_data": {"metadata": {"hotkey": hotkey, "ref": ref}},
            },
            idem=f"checkout:{ref}",
        )

    def retrieve_checkout_session(self, session_id: str) -> dict[str, Any]:
        return self._req(
            "GET",
            f"/v1/checkout/sessions/{session_id}",
            {"expand": ["payment_intent.latest_charge.balance_transaction"]},
        )

    # --- payment intents (test-mode payments without a browser) -----------------------------------
    def create_test_payment(self, *, amount_cents: int, hotkey: str, ref: str) -> dict[str, Any]:
        """Pay with Stripe's test card token; refused on live keys."""
        if not self.test_mode:
            raise StripeError("test payments need a test key")
        return self._req(
            "POST",
            "/v1/payment_intents",
            {
                "amount": amount_cents,
                "currency": "usd",
                "payment_method": "pm_card_visa",
                "payment_method_types": ["card"],
                "confirm": True,
                "metadata": {"hotkey": hotkey, "ref": ref},
                "expand": ["latest_charge.balance_transaction"],
            },
            idem=f"testpay:{ref}",
        )

    def retrieve_payment_intent(self, pi_id: str) -> dict[str, Any]:
        return self._req(
            "GET", f"/v1/payment_intents/{pi_id}", {"expand": ["latest_charge.balance_transaction"]}
        )

    # --- refunds ----------------------------------------------------------------------------------
    def create_refund(
        self, *, payment_intent: str, amount_cents: int, idem: str, hotkey: str
    ) -> dict[str, Any]:
        return self._req(
            "POST",
            "/v1/refunds",
            {"payment_intent": payment_intent, "amount": amount_cents, "metadata": {"hotkey": hotkey}},
            idem=idem,
        )


def payment_fee_cents(payment_intent: dict[str, Any]) -> int | None:
    """The processor's actual fee in the charge currency (USD cents), or None if not available yet.

    The balance transaction is in the account's settlement currency (e.g. AED for a UAE account) and
    includes conversion fees and VAT; ``exchange_rate`` converts charge -> settlement currency, so the fee
    in the charge currency is ``fee / exchange_rate`` (rounded up: never credit more than we received).
    """
    charge = payment_intent.get("latest_charge")
    if not isinstance(charge, dict):
        return None
    bt = charge.get("balance_transaction")
    if not isinstance(bt, dict):
        return None
    fee = int(bt.get("fee", 0))
    charge_cur = str(payment_intent.get("currency") or charge.get("currency") or "usd").lower()
    if str(bt.get("currency", charge_cur)).lower() == charge_cur:
        return fee
    rate = bt.get("exchange_rate")
    if not rate:
        raise StripeError("fee in another currency without an exchange rate")
    return math.ceil(fee / float(rate))


def verify_webhook(
    payload: bytes, sig_header: str, secret: str, *, tolerance: int = 300, now: float | None = None
) -> dict:
    """Check a ``Stripe-Signature`` header (``t=...,v1=...``) and return the parsed event."""
    parts: dict[str, list[str]] = {}
    for item in sig_header.split(","):
        k, _, v = item.strip().partition("=")
        parts.setdefault(k, []).append(v)
    try:
        ts = int(parts["t"][0])
    except (KeyError, ValueError, IndexError) as e:
        raise StripeError("malformed Stripe-Signature header") from e
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, s) for s in parts.get("v1", [])):
        raise StripeError("webhook signature mismatch")
    if abs((now if now is not None else time.time()) - ts) > tolerance:
        raise StripeError("webhook timestamp outside tolerance")
    return json.loads(payload)
