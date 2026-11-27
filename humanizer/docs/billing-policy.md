# Evaluation billing and refund policy

This page describes how miners pay for evaluating a humanizer submission in the SN32 humanizer
competition. Every evaluation is paid; there are no free runs.

## What you pay for

Evaluating a submission costs real money: paid AI detectors score every output, LLM judges check quality,
and your model runs on a rented GPU. You pay exactly for that, item by item, at the rates in the public
**price card** (`configs/price_card.json`). Every quote and statement names the card's version and its
sha256, so a statement can always be recomputed.

All money here is **USDC** (1 USDC = 1 US dollar on the card): your balance, quotes, holds, charges and
refunds. You pay in **TAO**; it is converted to USDC when it arrives.

| Item | Unit | Rate (card 2026-11-26) |
|---|---|---|
| Pangram | words per output | 0.05 USDC per 100 words, rounded up per text |
| GPTZero | words | 0.15 USDC per 1,000 words |
| ZeroGPT | words | 0.069 USDC per 1,000 words |
| LLM judges | tokens | OpenRouter list price of each judge model |
| GPU pod (H100 80GB HBM3, else H200) | hours | 4.50 USDC per hour |

## How a payment works

1. **Quote.** Before a run you get an upper-bound quote, valid for one hour: detector words are bounded
   by the output length cap (1.3 × input words), judges by a fixed amount per output, the GPU by a fixed
   number of hours. A typical quote for a period of 500 texts is about 167 USDC; the metered charge is
   usually lower (about 117 USDC).
2. **Top-up in TAO.** If your balance does not cover the quote, the submit command gives you your
   hotkey's **deposit address** and the TAO to send (see "Paying in TAO" below). Send it within the
   quote's hour, then run `paid` (we also notice it by ourselves). TAO sent in time still counts if it
   lands up to one hour after the window closes; otherwise the submission expires without using your
   registration, and whatever arrived stays on your balance.
3. **Hold.** When your submission joins the queue (the moment your balance covers the quote), the quoted
   amount is held on your balance. Until your run starts you can cancel it and the hold is released.
4. **Settlement.** After the run you are charged the **metered** cost: the words actually sent to each
   detector (texts that were already scored cost nothing), the judges' tokens, and the pod's hours. You
   are never charged more than the hold. The rest returns to your balance and can pay for your next run.
5. **Statement.** Every run gets an itemized statement: quantity × rate for each line. After the period
   is published, anyone can recompute it from the published outputs.

## Paying in TAO

1. **Your deposit address.** Each hotkey has **one deposit address of its own** (`deposit` shows it, with
   the current rate and your balance). It is the Substrate mirror of an EVM account, so a plain TAO
   transfer funds it: from btcli, any wallet or an
   exchange (`--send` sends it from your coldkey). TAO sent there at any time is converted and credited.
2. **How much.** When your balance is short, submit shows the TAO to send: the USDC missing × 1.05 (a 5%
   buffer, since the rate moves) at the current rate, plus the bridge's cost. What is left over stays on
   your balance.
3. **Converted on arrival.** Each deposit is converted whole and on its own: the ForeverMoney bridge
   carries it from Bittensor EVM to Base over Chainlink CCIP (about 2 minutes, about 0.005 TAO ≈ 1 USDC
   including gas), then it is swapped on Base TAO → WETH (Uniswap V3, 0.3% pool) → USDC (0.05% pool),
   about 0.4% in all for 1 TAO. Your balance is credited with **the USDC that actually came out**; price
   moves after that do not touch your money or ours. If the swap would pay more than 2% under the
   small-trade price, it waits and is retried. Not enough after the swap? Send the rest to the same
   address.

### Risks

* ForeverMoney is young: live since 2026-08-20. Its admin is a 2-of-3 Safe; it was audited by Hackerbane
  in September 2026 (no critical, high or medium findings). Bridged TAO on Base is backed by root-staked
  TAO in their vault on Finney.
* If the bridge is paused, your TAO waits on your deposit address and is converted when it resumes.
* Your balance is USDC held by the evaluator's treasury (native Circle USDC on Base).

## Submitting

* Your bundle (`bundle.tar`: code and weights) lives in **your own Hugging Face model repo**. From the
  hotkey that submits, you commit on chain `<namespace>/<repo>@<full 40-character commit sha>` (at most
  128 bytes), then call the API. Only that exact commit is evaluated: branch and tag names such as `main`
  are refused, so pushing to the repo later changes nothing for your submission. The commitment must be
  made after your hotkey's current registration. `scripts/miner_submit.py publish` does all three steps.
* The repo may be private while you submit and pay (nobody can copy it meanwhile), but it must be
  **public** (no token, not gated) when your evaluation starts: otherwise the submission is refused at
  that moment, nothing is charged and the registration counts as used.
* Keep the commit downloadable until your result is final. **A crown counts only after we have copied
  your bundle** from that commit; if the file is gone or changed by then, the crown is not awarded (the
  evaluation is still charged). 48 hours after the coronation the king's bundle is public: your repo at the pinned commit if anyone
  can download it without a token, otherwise our archived copy in a public repo of ours.
* An identical bundle (same sha256) submitted from another hotkey counts once, for the hotkey that
  **committed it on chain first**; a later copy is refused, or dropped before it runs (nothing charged).
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

* Refunds are **on request only**, at any time, except while one of your runs holds money (ask again
  once it is settled). You can refund all of your available balance or a part of it
  (`refund --key K [--amount-usdc X]`; `--quote` shows the estimate and moves nothing).
* A refund is paid in **TAO** to **any address you name** in the request (`--to`, default: your wallet's
  coldkey). Deposits can likewise come from any wallet. Refunds keep working after your hotkey is
  deregistered: they are signed with the hotkey, not checked against the metagraph. Whoever holds your
  hotkey can therefore withdraw your balance: keep the balance small. The USDC is swapped to TAO on Base at the rate of that moment
  and bridged back to Finney (about 30 minutes: the bridge unstakes). You see the TAO sent and the CCIP
  message id (`refund-status --key K`).
* The cost of the way back comes out of the refunded amount: the bridge's CCIP fee (about 0.0023 ETH ≈
  3.8 USDC) plus about 0.05 USDC of gas on Base and about 0.0015 TAO of gas on Bittensor EVM (used when
  the bridge books the TAO for us instead of delivering it: we then claim it and send it on, and the
  refund shows the stages `claimable` … `forwarding` before `delivered`). What remains must buy at least 0.01 TAO; otherwise the refund
  fails and the money stays on your balance.
* Refunds are paid from the treasury's USDC. If it is short at that moment, the refund shows "waiting for
  funds" and is paid within 24 hours; nothing is lost meanwhile.
* Each refund request has an idempotency key, so retrying it never refunds twice.
* Amounts already charged for completed runs are not refundable.
* If a run fails because of us (no GPU pod, a lost pod, our infrastructure), **the same submission is run
  again by itself**, from the same pinned commit, with your hold kept: you never resubmit, and your
  on-chain commitment keeps its place. If our side stays down past the retries (about 4 hours), the
  submission is not closed: it waits with its hold until we have looked at it, and we run it again,
  or close it giving the full hold back, or close it charging only what was already used (at most the hold).
* If your bundle fails (it cannot be downloaded, its hash does not match, it is not a valid bundle, or
  your code crashes), the run is scored with what it produced (missing outputs count as failed) and you
  are charged the pod time used, plus whatever judges and detectors were actually called.
* Outputs that fail a quality gate severely (level 0) are not sent to the detectors: they already count
  as caught, so you do not pay for them.
* If a judge model or a detector is unavailable, the evaluation pauses with your hold kept and resumes
  from the saved outputs; you are never charged twice for the same call. If it cannot resume within about
  4 hours (4 retries), it is not closed: it waits with its hold until we have looked at it, and we either run it
  again or close it and give the full hold back.
* A quote that is not paid within its hour expires; money that arrives after that stays on your balance
  (use it for the next submission or ask for a refund).

## Records

We keep a full log of every detector, judge and GPU call made for your run, to resolve disputes.
During the testing phase statements are not yet cryptographically verifiable; a hash-chained
log anchored on chain is planned.
