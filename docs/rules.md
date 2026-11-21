# SN32 humanizer competition: rules

The humanizer competition of Bittensor subnet 32 (It's AI) pays for a **humanizer**: a model that rewrites AI-generated
English text so that commercial AI detectors take it for human writing, while the meaning and the
quality of the text are kept. This page is the participant's summary of the rules; the numbered design
decisions behind them are in [`spec.md`](spec.md) (`D-xx`). How to build and submit:
[`miners.md`](miners.md). Money: [`billing-policy.md`](billing-policy.md).

## 1. One king, paid every block

* There is **one king**. A submission is compared **directly with the current king** on the same hidden
  texts; if it wins with confidence (section 6) it becomes the new king at once.
* The competition pays **85%** to the current king and **10% / 5%** to the two previous kings with
  **different hotkeys**. If there is no previous king (or it is the owner's baseline), its share goes to
  the current king. While the owner's baseline is king, its share is burned.
* **Launch window:** for the first **3 days** after the first miner is crowned, the kings get only
  **25%** of the competition's share (in the same 85/10/5 proportions) and the rest is burned. This gives
  time to find and stop an exploit that slipped through at launch before it earns the full share; after
  the 3 days payouts are full.
* The competition gets **half** of the miners' emission: from v4 every validator gives it 50% of its
  weights (detection keeps the other 50%), taken from a leaderboard signed by the subnet owner. If the
  owner's server is down, validators keep paying the kings of the last valid leaderboard.
* A king whose hotkey loses its registration drops out: the kings after it move up one place, and if it
  was the current king, the crown goes back to the previous one.

## 2. Periods

* A period lasts **4 days**. It has about **500 hidden English texts** written by strong LLMs (12
  generators, closed and open) across genres (academic writing, essays, business and marketing,
  homework answers, forum Q&A, blog articles, reviews, news, creative), **150–350 words** each.
* Each period also has a **human reference set H** of the same size, from large pre-2022 human corpora,
  matched to the AI texts by genre and length band. Detectors are compared on your outputs vs H.
* The period is fixed in advance. When period t starts we publish the next period's commitment
  `c = H(secret)` together with the number of a future finney block B (about half a day before t ends).
  The next period's seed is `H(secret ‖ hash of B)`: we cannot steer it, because B was fixed before
  its hash existed, and nobody else knows the secret. The Merkle root of its texts is published once
  it is built, before it is used. All of this is at `GET /v1/periods`, and both the commitment and the
  Merkle root are also written to the finney chain as remarks from our notary address
  ([`docs/periods`](periods/README.md)), so the time they were published does not depend on our word.
* When a period closes, it is **revealed** at `GET /v1/periods/<id>/...`: the secret, inputs, facts,
  human text ids and their detector scores, every submission's outputs, the judges' verdicts and
  reasons, the detectors' scores, and the decisions. `scripts/verify_period.py --url <evaluator>/v1/periods/<id>`
  recomputes the commitment, the seed block, the Merkle root, every `S` and every crown test from
  those files alone, and checks on chain that the commitment came before the seed block and the
  Merkle root before the first evaluation.
* The king is re-run on every new period, so it always competes on fresh texts.

## 3. Submission

* A **bundle** (`bundle.tar`: code, Python wheels, weights) of at most **50 GB**, in your own Hugging Face
  model repo. You commit `<repo>@<40-hex commit sha>` on chain from the submitting hotkey; only that
  commit is ever evaluated (branch and tag names are refused), and the file's sha256 is pinned when you
  submit and checked again on the pod.
* Interface: `humanize(items: list[{"text", "lang", "genre"}]) -> list[str]`, one output per input, in
  order. Layout, licenses and a working example: [`examples/ref_humanizer`](../examples/ref_humanizer/README.md).
* **One submission per registration of a hotkey.** A cancelled, expired-unpaid or failed-on-our-side
  submission does not use it. The same bundle (same sha256) from another hotkey is refused.
* Submissions run **one at a time, in the order they were paid**.

## 4. Execution

* One **H100 (80 GB)**, **60 minutes** for the whole period, **no network**, as an unprivileged user on a
  read-only filesystem, in our base image (`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`). We set the
  random seeds.
* Outputs are **normalized** (Unicode NFKC, zero-width and control characters removed, homoglyphs mapped
  back) and **cut to 1.3× the input's word count**; everything is scored on the normalized text.

## 5. Quality: every output gets a level

Two LLM judges from different vendors check each output; a third model decides when they disagree.

* **Injection screen, before everything else.** A dedicated judge looks only for text aimed at an AI, a
  reviewer or a grader instead of a human reader: instructions, notes about the rewrite's quality or
  authorship, fake markup (roles, chat tokens, closing tags, JSON verdicts), hidden text. If it quotes such a
  passage from your output (a quote that is really in your output and not in the input), that output is
  level **0** and the quality judges never see it. Text that is also in the input never counts. A flagged
  output is not sent to the detectors, so it costs you only the screen's tokens.

* **G1 meaning.** The input's facts (extracted and verified when the period is built) must be kept:
  no distortions, no additions, at most max(1, 10%) facts missing, limited filler.
* **G2 style.** Same language and same genre. Register (formality), format (paragraphs, lists) and
  person or tense may change: that is part of humanizing (a change of who did what, or when, is a meaning
  change and G1 catches it).
* **G3 naturalness.** Only defects of form that the input did not have, judged by the conventions of the
  text's genre (what is normal in ordinary human writing of that kind is not a defect). Each judge lists
  the defects with a quote and a severity on one scale, the effect on a reader: *minor* (only a careful
  reader notices; meaning and tone intact), *major* (disrupts reading or the tone; meaning still clear),
  *severe* (meaning lost or the text cannot be followed, an unfinished sentence, junk). A defect counts
  when **both** judges point at the same spot, with the milder of their two severities; the text counts as
  severe when both judges find something severe (the arbiter breaks a tie). Coherence is judged for the
  text as a whole: when both judges find that it cannot be followed (sentences that do not connect, come
  in an order that makes no sense or contradict each other), the level is 0.

| Level | Means |
|---|---|
| **1** | every gate passed |
| **0.75** | only a few minor slips of form: at most max(3, words/75), e.g. 3 in 200 words, 5 in 350 |
| **0.5** | one distortion or addition, one missing fact over the allowance or too much filler (G1); a major defect of form or too many minor slips (G3) |
| **0** | two or more distortions or additions, many missing facts, another language or genre, broken text, empty output |

Text addressed to the judges inside an output (instructions, claims that the checks passed, notes about
the evaluation) is itself a defect: junk for G3 (level 0) and an addition for G1.

An output's level is the lowest over the gates. In G1 and G2 a judgement counts when both judges make
it, and the arbiter decides when they disagree. The level multiplies the output's contribution to the
score: 0 counts as "caught by the detector", 0.5 as half, 0.75 as three quarters.

## 6. Score and crown

For each crown detector `d`:

    E_d = min(0.5, 1 − AUROC_d) + ½ · P_d

* `AUROC_d`: how well detector `d` separates your outputs from the human set H, computed within
  genre × length groups and averaged by group size. 0.5 means it cannot tell them apart.
* `P_d`: the mean probability of "human" that detector `d` gives your outputs.
* Both parts use the quality levels; `E_d = 1` means the detector takes every output for human.

The score is a soft minimum over detectors, so you must beat **all** of them:

    S = ( mean_d E_d^(−4) )^(−1/4)        (S = 0 if any E_d = 0)

**Crown test:** a paired bootstrap over the period's texts (your outputs and the king's resampled
together, H resampled too). You are crowned if the one-sided 5% lower bound of `ΔS = S_you − S_king`
is above `δ_min` (0 at launch; it may be raised after the first periods, with notice).

**Crown detectors: GPTZero, ZeroGPT and Pangram.** It's AI's own detector is used for tests only and
never decides a crown.

**Identical conditions.** You and the king are always judged and scored the same way:
* **Every output is judged and scored completely.** If a judge or a detector fails on even one output, it is
  asked again; if it keeps failing, your evaluation pauses and resumes later. An output never gets level
  0 or a "caught" score because our judge or detector failed.
* **The same judges on the same infrastructure.** Each judge model is pinned to one provider, with no
  fallback to another; if it is down, evaluations wait.
* **The same scoring code and settings.** The king's run carries a fingerprint of the scoring code, the
  gate settings and the detector set. If any of them changes, the king is rescored from his saved outputs
  before anyone is compared with him.
* **The same detector models.** If a detector vendor changes its model during a period (a new version
  string, or two control texts from H scoring differently), evaluations pause until H and the king are
  rescored by the new model.
* **The same GPU model:** every run, the king's included, uses an NVIDIA H100 80GB HBM3 (SXM) pod, never the
  slower PCIe variant.

LLM judges are not deterministic across different texts, but the same text always gets the same verdict
(verdicts are cached), and the king's outputs and yours go through exactly the same steps.

## 7. After a crown

* The crown counts only after we have copied your bundle from the pinned commit into our archive (keep
  it downloadable until your result is final).
* **48 hours after the coronation** the king's bundle is public: your repo at the pinned commit if
  anyone can download it, otherwise our archived copy in a public repo of ours.

## 8. What you see

Before a period is revealed you see, for your own submission: its stage, `S`, the crown decision with
`ΔS` and its lower bound, how many outputs got level 1 / 0.75 / 0.5 / 0 (overall and per gate), and per detector
`E`, AUROC and the mean "human" probability. Texts, per-text levels and the judges' reasons quote the
hidden inputs, so they are published with the period. The public leaderboard shows every finished
submission's scores, the payouts and the kings.

## 9. Money

Every evaluation is paid by the miner: the detectors' words, the judges' tokens and the pod's hours at
the published price card. You pay a hold for the quote (about **$165** for a 500-text period), are
charged the **metered** cost (usually much less), and can refund the rest to your card. Details:
[`billing-policy.md`](billing-policy.md).

## Test launch (before production)

The test launch publishes the code, the rules and the reference humanizer so that you can prepare;
**it takes no submissions.** An evaluation costs the same as in production (judges and a GPU pod) and
would give nothing in return. The evaluator's public API is up in read-only mode (`/v1/info`,
`/v1/leaderboard`, `/v1/results`, `/v1/kings`). Build and check your bundle locally (`miners.md`,
section 1). Submissions open on **Monday, 2026-11-23, 12:00 UTC** on mainnet subnet 32; at the same
moment the validators on v4 start giving the competition its half of the weights.
