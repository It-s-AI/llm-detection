# 01. Detector gateway — report

## What was done
- A single layer for detectors: **GPTZero, ZeroGPT, Pangram** (take part in the crown) and **It's AI** (proxy for simulations only).
  Each text → a continuous AI score 0..1 + fallback score variants from the same response + the vendor's model version.
- Rate limiting (token bucket), retries on 429/5xx/timeouts honoring `Retry-After`, response cache (SQLite).
- **Every paid request goes through the spending ledger** with hard caps: without the ledger a paid detector
  refuses to work at all; when the cap is exceeded, the request is not sent.
- Fakes for free tests: `FakeDetector`, emulation of each vendor's protocol (including 429/5xx/malformed responses),
  recording of real responses (keys are stripped) and their replay in tests.
- Canaries: a fixed set of texts → comparison of version and scores against a baseline (a vendor model change gets caught).

## Mapping to decisions
| Decision | Where in code | Verified by | Status |
|---|---|---|---|
| D-30 three crown detectors | `detectors/registry.py` (`CROWN_DETECTORS`), `detectors/vendors.py` | `test_vendors.py`, live check | ✅ |
| D-32 model version and canaries | `vendors.py` (field `version`), `detectors/canary.py` | `test_canaries_detect_version_change` | ✅ |
| D-33 It's AI proxy only | `ItsAI` is not in `CROWN_DETECTORS` | review | ✅ |
| D-24 continuous score | `Extracted.score` + `candidates` | `test_*_primary_*` | ✅ (the primary score is provisional, see below) |
| D-91 spending ledger | `budget.py`, `VendorDetector._one` | `test_budget.py`, `test_budget_cap_blocks_requests`, `test_paid_detector_without_ledger_refuses` | ✅ |

## Deviations and important findings
- **ZeroGPT:** score = `fakePercentage/100`. The `isHuman` field takes 0 or 100 on real responses — it is a flag,
  not a probability (your old client in `its-ai-leaderboard` used it as a probability).
- **Pangram:** the v2 endpoints (`*.pangramlabs.com`, `text-batch`) were shut down on 2026-04-01 — the old client is broken.
  The asynchronous `text.external-api.pangram.com/task` is used, with the model set explicitly to **`pangram-4.1`**
  (available: `default`, `pangram-4`, `pangram-4.1`). There is no batch endpoint — requests go one at a time, in parallel.
- **The primary score is provisional** (the final choice follows the admission test in phase 3):
  GPTZero `P(ai)+P(mixed)`; ZeroGPT `fakePercentage/100`; Pangram `max(weighted P(AI), weighted humanizer_score)`;
  It's AI — the mean of per-word probabilities. The other variants are stored in `candidates`, so there is no need to re-request.
- Vendors do not return the price per request — we charge an **upper-bound estimate** (Pangram officially $0.05/100 words,
  rounded up; GPTZero and ZeroGPT from third-party data, with margin). Actual spending is most likely lower.

## Live example (real responses)
10 human texts (Pile paragraphs, before 2021) + 10 AI texts (written by an LLM), each twice without cache:

| Detector | Successful | Version | Determinism max\|Δ\| | Human score (mean / max) | AI score (mean / min) | s/text |
|---|---|---|---|---|---|---|
| GPTZero | 40/40 | `2026-11-04-base` | 7.6e-7 (float noise) | 0.0000 / 0.0001 | 1.000 / 1.000 | 0.26 |
| ZeroGPT | 40/40 | — (not returned) | 0 | 0.015 / 0.151 | 1.000 / 1.000 | 0.24 |
| Pangram | 40/40 | `4.1` | ~1e-17 | ~0 | 1.000 | 0.85 (task polling) |
| It's AI | 40/40 | — | 0 | 0.031 | 0.99999 | 0.06 |

All four separate these texts perfectly — as expected: the AI texts are "textbook" ones, not humanized.
Raw per-text numbers: `docs/reports/data/01-live-check.json` (for Pangram the per-text data was not saved
due to a bug in the first run — the totals above are from the log; I did not repeat the $4 run just for that).

## Test results
- Offline: 31 tests, all green (parsing of all vendors' responses on fakes and on **recorded real responses**,
  retries, limiter, cache, budget, canaries, absence of secrets in recordings and errors).
- Live: determinism confirmed for all; responses are stable → caching and paired comparison are valid.

## Phase spending
Per the ledger (upper-bound estimates): GPTZero **$2.51**, Pangram **$4.00**, ZeroGPT **$1.15** — total **~$7.7**
(of which ~$2.4 went on repeats caused by my bug in response unpacking; plus ~$0.15 on the initial key check
before the ledger existed).

## How to check it yourself
```bash
.venv/bin/pytest tests/detectors          # free, on fakes and recorded responses
.venv/bin/python -m humanizer_eval.cli budget   # current spending against caps
.venv/bin/python scripts/live_detector_check.py --detectors its_ai   # live run (It's AI is free)
```

## What you should look at
- The table above and `docs/reports/data/01-live-check.json`.
- Whether the provisional definitions of the primary score (above) are acceptable until the admission test.

## Risks, open questions, next steps
- Pangram and ZeroGPT rate limits are still not confirmed in writing; I did not run a burst test.
- The admission test (AUROC ≥ 0.9 on raw AI vs humans) is in phase 3, once data of the right distribution is available.
