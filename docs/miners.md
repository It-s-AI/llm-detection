# Submitting a humanizer (SN32 humanizer competition)

This is the miner's guide to the humanizer competition: what to build, how to submit it, what you pay and
what you see. Prices, refunds and the exact rules are in [`billing-policy.md`](billing-policy.md); the
metric is in [`spec.md`](spec.md).

## 1. What you build

A **bundle**: one `bundle.tar` with your code, its Python wheels and your weights. It runs on one H100
(80 GB) with **no network**, within 60 minutes, and is at most **50 GB**. The layout, the entrypoint contract
(`humanize(items) -> list[str]`) and a complete working example are in
[`examples/ref_humanizer`](../examples/ref_humanizer/README.md). Check your bundle locally before you pay:

```
python src/humanizer_eval/runner/bundle.py bundle.tar        # format check, prints the manifest
HUMEVAL_SUB=bundle HUMEVAL_IN=items_dir HUMEVAL_OUT=out python harness/run.py   # a real run
```

## 2. What you need

* **A hotkey registered on SN32.** Each registration of a hotkey can make **one** submission; to submit
  again, register a new hotkey (or the same one again after it was deregistered). Keep the hotkey
  registered while you hold a crown: a king whose hotkey loses its registration drops out of the payouts.
* **A Hugging Face account** with a model repo for the bundle (public or private).
* **A card** for the evaluation hold (Stripe Checkout). A typical quote for a 500-text period is about
  $165; you are charged the metered cost (usually less) and the rest stays on your balance.

## 3. Submit

The one-command way (needs `pip install bittensor huggingface_hub`, and `huggingface-cli login` for the
upload):

```
python scripts/miner_submit.py --api https://<evaluator> --wallet <coldkey> --hotkey <hotkey> \
    publish --repo <you>/<repo> --file bundle.tar --wait
```

It does three things, which you can also do yourself:

1. **Upload** `bundle.tar` to the root of your model repo and note the **full commit sha** (40 hex
   characters) of the commit that contains it.
2. **Commit on chain**, from the same hotkey, the string `<you>/<repo>@<commit sha>`:
   ```python
   import bittensor as bt
   from bittensor_wallet import Wallet
   bt.Subtensor("finney").set_commitment(Wallet(name="<coldkey>", hotkey="<hotkey>"), 32,
                                         "<you>/<repo>@<40-hex commit sha>")
   ```
   It must fit in 128 bytes and be made **after** your hotkey's current registration.
3. **Submit** through the API (requests are signed with your hotkey; the script does it):
   `miner_submit.py ... submit --commitment "<you>/<repo>@<sha>"`. If the evaluator has not seen your
   commitment yet (it reads the chain every few minutes), you get "no commitment ... yet": retry shortly.

### Paying

There is no web site to log in to: everything happens in the terminal plus one Stripe page.

1. If your balance does not cover the quote, `publish` / `submit` prints a **Stripe payment link**, the
   amount and the deadline: the link works for **60 minutes** (the quote's lifetime).
2. Open the link and pay by card. Stripe then sends you to a short text page with the next command.
3. Run `miner_submit.py ... paid <submission_id>`. The evaluator asks Stripe, credits your balance and puts
   the submission in the queue; the answer shows your place. If it says "no payment yet", the answer
   repeats the link and the deadline. Running `paid` again is safe: a payment is credited once.

We also notice a paid link by ourselves within a minute, so the queue follows the order of payment even if
you forget step 3. A link paid before its deadline counts even if it is confirmed later. A link that
closes unpaid expires the submission; that does not use your registration, so just submit again.
Submissions run one at a time, in the order they were paid.

### The version is pinned

Only the exact commit in your commitment is evaluated. Branch and tag names (`main`, `v1`), short shas
and anything else are refused, and a branch named like your commit makes the submission ambiguous and is
refused too. The bundle's sha256 is read from that commit's metadata when you submit and checked again
when the pod downloads it, so **pushing to the repo after submitting changes nothing**: to evaluate a new
version, register a new hotkey, commit the new sha and submit again.

### Private repos

Add `--private` (upload) and `--hf-token <token>` (submit) with a **fine-grained token that can only read
that one repo**. The token stays on the evaluator's server, encrypted; the evaluation pod only receives a
short-lived signed download link, and the token is deleted when your evaluation ends. A public repo is
visible to everyone before it is evaluated: identical copies from other hotkeys are refused, modified
copies are not.

## 4. What happens next

`miner_submit.py ... status <submission_id> --wait` shows the stage: `awaiting_payment`, `queued`
(with your place), `running`, `gating`, `detecting`, `scoring`, then `done` (or `failed` if our side
failed: nothing is charged and your registration stays usable). When it is done you see:

* `S`, the crown decision, the king's `S_king`, `delta` and the confidence `lower_bound`;
* how many of your outputs got quality level 1 / 0.75 / 0.5 / 0, overall and per gate (`levels`, `gates`);
* per detector: evasion `E`, `auroc` and the mean "human" probability `p_human`;
* `charged_cents`, and the itemized statement (`statement <submission_id>`).

Your outputs and the judges' written reasons are published when the period closes (they quote the
hidden texts): `GET <evaluator>/v1/periods/<id>/files.json` lists the files, and
`scripts/verify_period.py` recomputes every score and crown test from them. If you won, the stage goes through `archiving`: we download your bundle again from the pinned
commit into our archive, and **the crown counts only when that copy matches** (keep the commit
downloadable until then). Re-runs of the king on later periods use our copy.

**Publication.** 48 hours after the coronation `GET /v1/kings` links the king's bundle. The link is your
repo at the pinned commit (`huggingface.co/<you>/<repo>/tree/<sha>`), but only if anyone can download that
commit without a token, so make a private repo public by then. We check the link every 6 hours. If the
repo is private, gated, deleted, or has a branch named like the commit, we publish our archived copy in a
public repo of ours instead, and that copy stays.

## 5. Money

`balance` shows your balance; `refund --key <any-unique-key>` returns the unused balance to your card
(retrying with the same key never refunds twice; refunds wait while one of your submissions is held).
The card processor's fee is not refundable. See [`billing-policy.md`](billing-policy.md).

## 6. Payouts

The competition's half of the miners' emission pays 85% to the current king and 10% / 5% to the
two previous kings with different hotkeys; shares with no owner are burned. For the first 3 days after the
first miner crown the kings get a quarter of that and the rest is burned (launch window). Validators add these weights
from a leaderboard signed by the subnet owner to their detection weights (SN32 v4, one weight vector).
