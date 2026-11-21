# 10. Submission intake: miner API, queue, worker — report

## What was done
- **Miner API** (`src/humanizer_eval/intake/api.py`, FastAPI):
  - `POST /v1/submissions` — bundle link, sha256 and size. The response contains a quote and, if the balance is insufficient, a Stripe Checkout payment link.
  - `GET /v1/submissions/{id}` — stage, queue position, result (the same as what is visible under D-58).
  - `GET .../statement` — itemized statement.
  - `DELETE /v1/submissions/{id}` — cancellation before the run starts; the hold is released.
  - `GET /v1/balance`, `POST /v1/refunds` — balance and refund.
  - `GET /v1/info` (no signature) — period, detectors, quote estimate, queue length.
  - `POST /v1/submissions/{id}/paid` — "I have paid": the server queries Stripe, credits the payment and enqueues the submission (2026-11-17).
  - `POST /v1/stripe/webhook` — payment crediting, optional.
- **Hotkey signature.**
  - Every miner request is signed with the miner's Bittensor hotkey (sr25519).
  - The signature covers the method, path, time, a one-time nonce and the body hash.
  - The window is ±5 min and a nonce is accepted only once, so an intercepted request cannot be replayed.
  - Compatibility with `bittensor_wallet` is verified on a signature produced by the wallet itself.
- **Registration.** Only hotkeys registered in SN32 can submit.
  - The metagraph snapshot is written by `scripts/refresh_metagraph.py`. Tested on finney: block 9 496 894, 256 hotkeys.
  - If the snapshot is older than an hour, intake is temporarily closed (503). Otherwise an unregistered hotkey would pay for a crown it cannot receive.
- **Queue** (SQLite).
  - Order is by payment time.
  - One evaluation at a time: one pod, D-65.
  - One active submission per hotkey.
  - No more than 50 paid submissions waiting.
- **Worker** (`intake/worker.py`).
  - Takes a submission from the queue and hands it to the orchestrator from report 09.
  - Before the first submission of a period, it runs the king on the new inputs at the owner's expense: the baseline by replay without a pod, a miner's bundle on a fresh pod (D-11, D-63).
  - Pause (judges or detectors unavailable): retry after 15 min, then ×2, but no less often than once every 6 h. After 8 pauses (≈ a day) the evaluation is withdrawn: the hold is released and nothing is charged.
  - A bug in our code: up to 3 attempts, then also release.
  - Process restart: an unfinished submission returns to the queue. Saved outputs are picked up again; the pod is not rented a second time.
- **Miner client.** `intake/client.py` and `scripts/miner_submit.py`: wallet signing, submission, waiting, balance, refund.
- **Server.** `scripts/intake_server.py` runs the API and the worker in one process.

## Mapping to decisions
| Decision | Where in code | Verified by | Status |
|---|---|---|---|
| D-61 bundle ≤ 100 GB | `IntakeLimits` (the same limit as on the pod) | test | ✅ |
| D-65 one pod at a time | `SubmissionQueue.claim` | test: does not hand out a second one while the first is running | ✅ |
| D-80 quote → payment → hold → statement → refund | `IntakeService` + `billing` | e2e over HTTP with Stripe in test mode | ✅ |
| D-11, D-63 king on a new period | `QueueWorker.ensure_king` | test: the king's bundle is run on the new period, the miner does not pay | ✅ |
| D-58 only aggregates before reveal | `status` returns the public part of the record | test: only allowed fields | ✅ |
| D-90 pause when services are unavailable | worker: retry with waiting | test: continuation without a second pod | ✅ |
| D-92 weights by registered hotkeys | `MetagraphSnapshot` | live SN32 snapshot | ✅ (intake), validator — next |

## Deviations and new rules
- **The hold is opened at payment, not at launch.**
  - The money is locked while the submission waits, so a refund during the wait is impossible.
  - The submission can be canceled before launch; then the hold is released.
- **An unpaid quote expires after an hour.** Money that arrives later stays on the balance.
- **Bundle link.**
  - https only, port 443, no login or password in the link.
  - Host name only: not an IP, not localhost, not `.internal`/`.local`.
  - Redirects are also https only (`curl --proto-redir =https`).
  - Why: the pod must not reach internal addresses of the Lium network, and a redirect must not lead to `file://`.
- **The queue limit counts only paid submissions.** Otherwise a stream of unpaid quotes could close the queue to honest miners.
- **Quote with an empty list of paid detectors.** Previously an empty list meant "all detectors in the price list", and the quote on the pilot with It's AI included GPTZero, ZeroGPT and Pangram. Now an empty list means "none". In production this changed nothing: the crown detectors are in the price list.
- **Sybils.** Several hotkeys can be in the queue at the same time, but each pays its own hold. Resubmitting the same bundle does not change the outputs: the harness seed is the same for the whole period. A sybil gains no advantage, only cost.

## Live example (pilot 1, It's AI detector, Stripe in test mode)
Server on localhost, the bundle replaced with saved outputs of reference v2 (no pod), judges from cache.

| Step | Result |
|---|---|
| Baseline king on the period | S 0.0086 (as in report 09) |
| Request without a signature | 401 |
| Submission | `awaiting_payment`, a real Stripe Checkout link for $29.99 (the quote also counted paid detectors; fixed after the run, see below) |
| Second submission from the same hotkey | 409 |
| Submission status with another hotkey | 404 |
| Payment with a test card → queue → run | `queued → running → done` in 9 s |
| Result | S 0.0424, **crown**: ΔS +0.0338, lower bound +0.0246 |
| Statement | $0: everything from cache, the detector is not paid, no pod; the $27.84 hold returned to the balance |
| Refund | $28.16 to the original payment; the Stripe fee of $1.83 stays with the processor (per policy) |

Data: `docs/reports/data/10-intake-e2e.json`.

## Test results
- `tests/intake/` — 20 tests:
  - signature: wallet compatibility, forgeries, replay, Polkadot.js wrapper;
  - full cycle over HTTP;
  - invalid requests;
  - queue limit and order;
  - cancellation and quote expiry;
  - crediting without a webhook;
  - pause and continuation;
  - giving up after pauses;
  - restart;
  - king on a new period;
  - worker crash.
- The whole suite is green: 418 tests.

## How to check it yourself
```
python scripts/intake_server.py --period pilot-1 --detectors its_ai --allow-missing-human \
    --dev-any-hotkey --replay-map var/intake-dev/replay.json --state-dir var/intake-dev
python scripts/miner_submit.py --api http://127.0.0.1:8032 --dev-seed <hex32> info
```

## What we need from you
- **Where to deploy the API.** We need a domain with TLS and a machine where the server and the worker live. The Lium and detector keys exist only there; the pod does not have them.
- **The Stripe webhook is not required** (decision 2026-11-17: payment without a website). A submission returns a Stripe link valid for 60 minutes; the miner pays and confirms with the `paid` command. The server also polls Stripe about unpaid links on its own every 15 s. A webhook can be added later; it only speeds up crediting.
- **Checking the download by link on the pod.** We need permission to publish the identity bundle (our trivial code, 2 KB) publicly, for example in an HF repository. Then I will test the "link → download on the pod → sha256 check" path on Lium for ~$0.04.

## Payment without a website (2026-11-17)
- **Link.** `submit` / `publish` returns a Stripe Checkout link whose expiry equals the quote's expiry (60 minutes; Stripe allows from 30 minutes to 24 hours). Card payment only, no deferred methods: the card is charged immediately. The CLI prints the amount, the expiry, the link and the next command.
- **`paid <id>`.** The server queries Stripe (no more often than once every 3 s) and, if paid, credits the balance minus the actual fee and enqueues the submission. If not, the response repeats the link and the expiry. Retrying is safe: the payment is credited only once.
- **Last-minute payment.** A link paid before its expiry counts, even if the confirmation arrives after the quote's expiry. After the expiry the server queries Stripe once more and closes the submission only on a definitive "not paid" answer. While Stripe does not respond, the decision is postponed, but for no longer than 24 hours.
- **Post-payment page.** Stripe requires a return URL (verified: without `success_url` the session is not created). This is a plain-text response from our API with the `paid <id>` command, not a website.
- **Verified:** tests (link, `paid` before and after payment, retry, last-minute payment, Stripe failure, closed unpaid link, background crediting) and live Stripe in test mode (link expiry 60 minutes, card only, statuses open → expired).

## Risks and next steps
- The server and the worker run in one process. This is enough for the v1 load, but a server restart interrupts the current run (it resumes from saved outputs). Later they can be split: the SQLite queue already allows this.
- Next: integration into the `llm-detection` validator (mechanism 1 weights from the king state, D-92), commit-reveal on localnet, publication.
