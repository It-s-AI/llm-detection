# 13. Live check of the miner path: testnet, HF, Lium — report (2026-11-17)

## What we checked
The entire submission path on real services, as in production, except for the network: instead of SN32, testnet subnet 325 (free slots, registration ~τ0, nobody displaced).

1. **HF.** The miner uploads the bundle to their own repository: `sergak0/sn32-test-public` (public) and `sergak0/sn32-test-private` (private).
2. **Chain.** Registers a hotkey and commits `<repo>@<40-hex commit>` to the chain.
3. **Server.** `refresh_metagraph.py` writes a snapshot with the commitments; the server (`intake_server.py --netuid 325 --lium --cheap-gpu`, period p1, It's AI detector) accepts the signed submission.
4. **Payment.** A Stripe link (test mode) valid for 60 minutes, then `paid`. The money was credited by a test payment via the API: no browser was involved in this run.
5. **Evaluation.** Queue, a pod on Lium, bundle download strictly by commit with sha256 verification, gates, It's AI, crown decision, statement, refund of the remainder.

## Results
| | t1: public repository | t2: private repository + read token |
|---|---|---|
| UID in testnet 325 | 3 | 4 |
| Commitment | `sergak0/sn32-test-public@ea554ba2…` | `sergak0/sn32-test-private@5f8504c8…` |
| Bundle visible without a token | yes | no (401) |
| Response to the submission | Stripe link, expiry, `paid` command | same; the token is not returned in the response |
| `paid` | `queued`, position 1 | `queued`, position 1 |
| Pod downloaded the bundle by commit, sha256 matched | yes | yes, via a signed URL; the token never reached the pod |
| S / king's S | 0.0109 / 0.0109, crown not awarded ("no changes" bundle against an identical king) | same |
| Levels 1 / 0.5 / 0 | 479 / 5 / 0 | 479 / 5 / 0 |
| Charged | $0.26 (0.1 h of pod time) | $0.15 |
| Token after the run | — | deleted from the database |

- **King on the period.** The server itself evaluated the "no changes" king on p1 (It's AI, S = 0.0109) before the first submission.
- **Lium.** Each run: one pod of our own (3090-class card, $0.25/h), deleted after the run. Other projects' pods untouched. Our total on Lium is $0.95 of $50.
- **Testnet.** Two registrations and two commitments cost ~τ0.001 in test TAO.

## What we found and fixed along the way
- **ZeroGPT** sometimes responds "code 500, try again" for individual texts. Because of this, loading of period p1 stopped (3 of 484 human texts). Now such texts are retried up to three times with a pause, and scores that were already paid for are taken from the cache (`load_period`, tests `tests/evaluation/test_load_period.py`).
- **Period facts for G1** cost ~$11 on OpenRouter for 484 texts: they are checked by three judges. They are computed once and cached, but two processes loading the period at the same time pay twice (~$8 extra today). The period must be loaded by a single process.
- **The statement line for the pod** is named `lium:h100` even for a cheap card: the price list has a single Lium rate. This does not matter for tests; in production the pod is always an H100.

## What was not verified live
- Payment via the link in a browser (test card `4242…`): Checkout, link expiry and statuses were verified via the Stripe API, but the browser payment itself was not. A spare hotkey t3 is available for this.
- Crown with a copy to the archive and publication after 48 h on real HF: verified by tests; live, the crown was not awarded (the bundle is no better than the king).
