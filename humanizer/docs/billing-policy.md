# Evaluation billing and refund policy

This page describes how miners pay for evaluating a humanizer submission in the SN32 humanizer
competition. Every evaluation is paid; there are no free runs.

## What you pay for

Evaluating a submission costs real money: paid AI detectors score every output, LLM judges check quality,
and your model runs on a rented GPU. You pay exactly for that, item by item, at the rates in the public
**price card** (`configs/price_card.json`). Every quote and statement names the card's version and its
sha256, so a statement can always be recomputed.

| Item | Unit | Rate (card 2026-11-09) |
|---|---|---|
| Pangram | words per output | $0.05 per 100 words, rounded up per text |
| GPTZero | words | $0.15 per 1,000 words |
| ZeroGPT | words | $0.069 per 1,000 words |
| LLM judges | tokens | OpenRouter list price of each judge model |
| GPU pod (H100) | hours | $2.50 per hour |
| Card payment | per top-up | the payment processor's actual fee (non-refundable) |

## How a payment works

1. **Quote.** Before a run you get an upper-bound quote, valid for one hour: detector words are bounded
   by the output length cap (1.3 × input words), judges by a fixed amount per output, the GPU by a fixed
   number of hours. A typical quote for a period of 500 texts is about $165; the metered charge is
   usually lower (about $115).
2. **Top-up.** If your balance does not cover the quote, the submit command gives you a Stripe payment link
   for the difference, valid for 60 minutes. Pay it by card, then confirm with the `paid` command (we
   also notice the payment by ourselves). Your balance is credited with the amount paid **minus the
   processor's actual fee**. A link paid in time counts even if it is confirmed later; an unpaid link
   expires the submission without using your registration.
3. **Hold.** When your submission joins the queue (the moment your balance covers the quote), the quoted
   amount is held on your balance. Until your run starts you can cancel it and the hold is released.
4. **Settlement.** After the run you are charged the **metered** cost: the words actually sent to each
   detector (texts that were already scored cost nothing), the judges' tokens, and the pod's hours. You
   are never charged more than the hold. The rest returns to your balance and can pay for your next run.
5. **Statement.** Every run gets an itemized statement: quantity × rate for each line. After the period
   is published, anyone can recompute it from the published outputs.

## Submitting

* Your bundle (`bundle.tar`: code and weights) lives in **your own Hugging Face model repo**. From the
  hotkey that submits, you commit on chain `<namespace>/<repo>@<full 40-character commit sha>` (at most
  128 bytes), then call the API. Only that exact commit is evaluated: branch and tag names such as `main`
  are refused, so pushing to the repo later changes nothing for your submission. The commitment must be
  made after your hotkey's current registration. `scripts/miner_submit.py publish` does all three steps.
* A private repo works too: give a fine-grained token that can only read that repo. It stays on our
  server, encrypted, the evaluation pod never sees it, and it is deleted when your evaluation ends.
* Keep the commit downloadable until your result is final. **A crown counts only after we have copied
  your bundle** from that commit; if the file is gone or changed by then, the crown is not awarded (the
  evaluation is still charged). 48 hours after the coronation the king's bundle is public: your repo at the pinned commit if anyone
  can download it without a token, otherwise our archived copy in a public repo of ours.
* An identical bundle (same sha256) submitted from another hotkey is refused: it counts once.
* Only hotkeys registered on the subnet can submit, and **each registration of a hotkey can make one
  submission**. To submit again, register a new hotkey (or register the same one again after it was
  deregistered). A submission that was cancelled before it ran, whose quote expired unpaid, or that
  failed on our side does not use the registration up.
* One submission per hotkey can be active at a time; submissions run in the order they were paid.
* While the period is running you see, for your submission: the score `S`, the crown decision with the
  king's score and the confidence bound, how many outputs got quality level 1 / 0.75 / 0.5 / 0 overall and per
  gate, and each detector's evasion, AUROC and mean "human" probability. The judges' written reasons and
  your outputs are published with the period (they quote the hidden texts).
* If a king's hotkey loses its registration, that coronation no longer counts: the kings before and after
  it close ranks in the 85/10/5 payouts, and if it was the current king, the crown returns to the
  previous one.

## Refunds

* You can ask for a refund of your **unused balance** at any time, except while one of your runs is being
  evaluated (refunds wait until it is settled).
* Refunds go back to the **original payment method**, newest payments first. Each refund request has an
  idempotency key, so retrying it never refunds twice.
* The processor's fee on the original payment is not refunded (the processor keeps it).
* Amounts already charged for completed runs are not refundable.
* If a run fails because of us (our infrastructure, a detector outage that cannot be retried), the hold
  is released in full and nothing is charged.
* If your bundle fails (it cannot be downloaded, its hash does not match, it is not a valid bundle, or
  your code crashes), the run is scored with what it produced (missing outputs count as failed) and you
  are charged the pod time used, plus whatever judges and detectors were actually called.
* Outputs that fail a quality gate severely (level 0) are not sent to the detectors: they already count
  as caught, so you do not pay for them.
* If a judge model or a detector is unavailable, the evaluation pauses with your hold kept and resumes
  from the saved outputs; you are never charged twice for the same call. If it cannot resume within about
  a day (8 retries), it is abandoned: the hold is released and nothing is charged.
* A quote that is not paid within its hour expires; money paid after that stays on your balance (use it
  for the next submission or ask for a refund).

## Records

We keep a full log of every detector, judge and GPU call made for your run, to resolve disputes and
chargebacks. During the testing phase statements are not yet cryptographically verifiable; a hash-chained
log anchored on chain is planned.
