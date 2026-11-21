# 08. Miner billing (D-80) — report

## What was done
- **Public price card** (`configs/price_card.json`, `billing/prices.py`): detectors per word, judges per token
  (OpenRouter prices), Lium per hour, a payment fee estimate. The price card's version and sha256 are in every quote and statement.
- **Upper-bound estimate** per submission: detector words are bounded by G0 (output ≤ 1.3 × input), judges — $0.04 per output,
  pod — 1.5 h. For 750 texts of ~260 words: ~$200–250.
- **Balances** (`billing/accounts.py`, SQLite, integer cents): a payment is credited net of the actual fee;
  a hold per submission; settlement at actual cost, no more than the hold, with the remainder going to the balance; refunds go to the original payments
  (newest first); while a hold is open, refunds are blocked. All operations are idempotent, within `BEGIN IMMEDIATE`.
- **Stripe** (`billing/stripe_api.py`, no SDK, httpx): Checkout Session, test PaymentIntent payment
  (`pm_card_visa`, test key only), refunds with an idempotency key, webhook signature verification. A live
  key is rejected until explicitly allowed.
- **Service** (`billing/service.py`): quote → payment link → crediting (webhook or polling) → hold →
  statement → refund. Public policy: `docs/billing-policy.md`.

## Check on test Stripe (`scripts/billing_live_test.py`, test money)
| Step | Amount |
|---|---|
| Quote (10 texts) | $6.99 (hold $6.22 + fee estimate) |
| Payment with a test card | $6.99, Stripe fee **$0.65** → $6.34 to the balance |
| Settlement at actual cost | $3.05 charged (detectors $1.97, judges $0.08, pod 0.4 h $1.00), $3.17 returned to the balance |
| Refund on request | $3.29 → Stripe confirmed `amount_refunded = 329` |

## Found during the check
- **The Stripe account settles in AED.** The fee in the balance transaction is in fils and includes: processing
  3.9% + AED 1, conversion 1%, 5% VAT on fees. The code converts the fee to cents at Stripe's rate (rounding
  up); the estimate in the price card was raised to 6% + $0.35.
- **The fee does not appear immediately.** The code waits up to 10 s for it and does not credit a payment with an unknown fee (otherwise
  a refund would pay out money we never received).
- Currency risk: a refund of a USD payment is debited from the AED balance at the current rate — a small risk borne by the owner.

## Not done
- HTTP endpoints (top-up page, webhook, refund request) — together with the validator service.
- The webhook secret (`STRIPE_WEBHOOK_SECRET`) will appear after the endpoint is registered in Stripe.
- Per-submission cost accounting (detector words, judge tokens, pod hours) is wired in during the end-to-end run.

## Tests
`tests/billing/test_billing.py`: 12 tests on a fake Stripe — quote, statement, idempotency of payments and
refunds, hold/settle/release, refund order, blocking during a run, unpaid session,
expired quote, a different price card, webhook signature, live key rejection, unknown fee, currency conversion.
