# 01. Detector gateway
Status: draft, implemented autonomously (the owner is away; changes after review)
Implements decisions: D-24, D-30, D-31, D-32, D-33, D-91

## Goal
A single way to get a **continuous** AI score (0..1) from each detector for a list of texts —
with rate limiting, retries, caching, cost accounting, model version control and free
tests without real APIs.

## Inputs and outputs
- Input: a list of texts (+ optional ids).
- Output: a `DetectorScore` per text: `score` (the primary continuous score), `candidates`
  (several score variants from the response — the final choice of the primary one is made by the admission test),
  `version` (the vendor's model version, if it returns one), `raw` (the raw response), `cached`, `cost_usd`.

## Interfaces
- `Detector` (protocol): `name`, `score(texts) -> list[DetectorScore]`.
- Adapters: `GPTZero` (`/v2/predict/text`), `ZeroGPT` (`/api/detect/detectText`, score =
  `fakePercentage/100`, **not** `isHuman`), `Pangram` (asynchronous `text.external-api.pangram.com/task`
  + polling, the model is specified explicitly, the v2 endpoints were shut down on 2026-04-01), `ItsAI` (batch, mean of
  per-word probabilities; for simulations only — D-33).
- Default primary score (provisional, until the admission test):
  GPTZero = `P(ai) + P(mixed)`; ZeroGPT = `fakePercentage/100`;
  Pangram = `max(window-length-weighted P(AI-generated)+P(AI-assisted), weighted humanizer_score)`;
  It's AI = mean of per-word probabilities.
- Shared HTTP layer: retries with exponential backoff on 429/5xx/timeouts (honors
  `Retry-After`), a token bucket per detector, a pluggable transport.
- Cache: SQLite keyed by (detector, config, text sha256); stores the model version.
- Cost accounting: before a request — a reservation in the spending ledger at the upper-bound price estimate, after it —
  a charge (vendors do not return the actual price, so we charge the estimate).
- Fakes: `FakeDetector` (no HTTP, deterministic, programmable) for downstream modules;
  fake transports emulating each vendor's formats (including 429/5xx/malformed responses);
  recording and replay of real responses (keys are stripped).
- Canaries: a fixed set of texts → comparison of version and scores against a saved baseline.

## What we do not do
The admission test (needs phase 3 data); a burst load test (waiting for the vendors' written rate limits).

## Tests and acceptance criteria
- Offline: parsing of all vendors' responses, retries, limiter, cache, budget (refusal when exceeded), canaries.
- Live (manual, `-m live`): 20 texts per detector — response formats; determinism 20×2 —
  matching scores; cost ≤ $10 in total.

## Risks and open questions
- GPTZero/ZeroGPT prices are taken from third-party sources → estimate with margin (upper bound).
- The choice of the primary score — after the admission test (phase 3).
