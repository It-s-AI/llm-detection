# Specification: SN32 humanizer competition (variant A, v1)

The single source of truth for decisions. Every decision has a number `D-xx`; design notes and
reports refer to these numbers. Any change is a new entry in the "Changelog" at the bottom, with the reason.

Statuses: **✅ accepted**, **🟡 default (can be changed)**, **⏳ decided by the pilot**.

## Principles

- **D-01 ✅ Kerckhoffs's principle.** All code is open. Only seeds, keys and the data of the current
  period are secret, until they are revealed. Protection is structural only: economics, statistics, zero sum,
  time priority, input transformations. No trap heuristics that can be studied and bypassed.
- **D-02 ✅** The crown is decided only by components the miner does not control: paid API detectors,
  validator data, LLM judges (only as quality thresholds).
- **D-03 ✅** There is no human in the evaluation loop (trust in a decentralized system). People take part
  only once — when the tools are chosen and calibrated (labeling for the judges) — but they do not evaluate submissions.
- **D-04 ✅** There are no free evaluations: every submission is paid for. The number of keys does not matter.

## Mechanisms and payouts

- **D-10 ✅** Two competitions in one subnet: detection (as now) and humanizer. Since v4 (owner's decision
  2026-11-21) this is **one weight vector**, without a second mechanism: the validator sets
  `0.5 × detection + 0.5 × humanizer` (branch `v4` in It-s-AI/llm-detection; the evaluator code is the `humanizer/`
  folder in the same place). A duel with the SN32 detectors in v1
  **is not done** (postponed).
- **D-11 ✅** One king. A challenger is compared **directly with the current king** on the same
  inputs. No candidate pools. The queue is ordered by the time of the on-chain commit. If the king changes in the middle
  of a period, subsequent challengers are compared with the new one.
- **D-12 ✅** Mechanism 1 payouts: 85/10/5 — the current king and the two previous **distinct hotkeys** (not "top 3 by
  score": otherwise the top would be taken by copies of the open king). If there is no previous king (or it is the owner's baseline),
  its share goes entirely to the current king (as in SN34). The share of a current king that is the baseline is burned (D-13).
- **D-13 🟡** The starting king is the owner's baseline; its share is burned. There is no growing burn during a long
  reign.
  Launch window (owner's decision 2026-11-21): for 3 days after the first miner coronation the kings get
  25% of the competition's share (85/10/5 proportions kept) and 75% is burned, so that a launch-time
  exploit can be debugged before it earns the full share (`publish/leaderboard.py`, `LAUNCH_RAMP_*`).
- **D-14 ✅** The humanizer share is **50%** of the validator's weights from launch (owner's decision 2026-11-21; previously: 10% →
  25% → 50%). The share is a constant in the validator code: all validators on v4 set the same split; it can be
  changed only by different code.

## Metric

- **D-20 ✅** For each detector d: `E_d = ½·2·min(0.5, 1 − AUROC_d) + ½·P_d`. AUROC is computed between the
  submission's outputs and the human reference set H **within "genre × length range" groups** and averaged
  with weights by the number of texts. `P_d` is the mean predicted probability of the "human" class (1 − detector
  score) over the submission's outputs; an output that did not pass a gate gives 0. The first term rewards
  genuine indistinguishability, the second rewards any decrease in the probabilities, even while the outputs are ranked above all
  humans. `E_d = 1` means the detector does not distinguish the outputs from humans and considers all outputs human. Detector
  scores are probabilities in [0, 1] (the primary score of each client).
- **D-21 ✅** Final score `S = (mean_d E_d^p)^(1/p)`, `p = −4` (soft minimum). If any `E_d = 0`, then `S = 0`.
- **D-22 ✅** Each output has a quality level (the minimum across gates): **1** — all gates passed; **0.75** —
  only a few minor form flaws (G3); **0.5** — a noticeable violation (one distortion or
  addition; 1 omission more than the norm; "filler" above the norm; a noticeable form defect or more minor ones than the
  norm); **0** — gross (≥ 2 distortions/additions or many omissions; change of language or genre; cut-off
  phrases, incoherence, garbage; empty output). A flag counts if both judges set it, or the arbiter in a dispute. An output's contribution to
  both parts of `E_d` is multiplied by the level: 0 — as "caught", 0.5 — half, 0.75 — three quarters. The penalty for gross outputs
  is `λ = 0` (broken text is useless but not harmful; the parameter exists). The requirement "≥ 95% passed the gates"
  for the crown no longer exists.
- **D-23 ✅** Crown test: paired bootstrap (inputs are resampled jointly for the king and the challenger,
  H is resampled too), one-sided **α = 5%**: crown if the 5% quantile of `ΔS = S_ch − S_king` > δ_min.
- **D-24 ✅** Every text goes through **all** detectors (no surrogates/PPI). **n = 500 texts**, H = 500
  (owner's decision 2026-11-10; upper estimate per submission ~$166, actual ~$115). At 500, the crown detects with 80% probability
  a gain in S of 17–47% relative to the king (report 05, "power under D-20/D-22"); as the king's S grows,
  we can move to 750. False crowns are controlled by the test itself (A/A 5.5% ± 0.4%); σ and φ are refined after launch.
- **D-25 ⏳** δ_min (the minimum margin for the crown, against a lottery) — after the first periods.

## Detectors

- **D-30 ✅** Detectors in the crown: **GPTZero, ZeroGPT, Pangram**. Originality — removed for now; Copyleaks and
  Turnitin — excluded.
- **D-31 ✅** Admission test: a detector is included in the track if its AUROC on raw AI texts versus
  human texts is ≥ 0.9.
- **D-32 ✅** Canaries (a fixed set of texts) every period; the detector version is recorded
  (GPTZero and Pangram return the version in the response). A version change in the middle of a period → coronations are frozen
  until the human reference set and the king are recomputed.
- **D-33 ✅** The proxy for simulations and tests is the production It's AI detector. **It never enters crown scoring.**

## Quality gates (all absolute; the king does not take part in them)

- **D-40 ✅ G0** normalization (NFKC, removal of zero-width/control characters, reverse mapping of
  homoglyphs; for Arabic — tatweel, presentation forms, letter look-alikes) as a **transformation**:
  everything is evaluated on the normalized text, and the product returns that same text. Length: the output is truncated to 1.3×
  the input; one that is too short will fail on meaning.
- **D-41 ✅ G1 meaning (v2, report 04b):** input facts tied to sentence numbers, verified by both
  model families on the input itself when the period is assembled; unchanged sentences are credited without an LLM; recall —
  by facts over the whole output ("present / missing / distorted"), precision — by output sentences ("present / filler /
  added / distorted"); two vendors judge, a third (the arbiter) resolves disputes. Thresholds in counts: omissions
  ≤ max(1, 10% of facts), distortions 0, additions 0, "filler" ≤ max(2, 20% of input sentences).
- **D-42 ✅ G2 style:** only language and genre are checked; changing them is level 0. **Register (formality),
  format (paragraphs, comma-separated lists, etc.) and person/tense are not checked** — changing them is exactly the
  humanizer's job (owner's decisions 2026-11-09 and 2026-11-20: addressing the reader as "you" in a blog is a
  normal technique; changing who did what or when is a change of meaning, and G1 catches it).
- **D-43 ✅ G3 naturalness (v3, owner's decisions 2026-11-20, calibration 2026-11-20):** the judges assess **only
  form** (meaning is G1's job) and only defects that are not in the input, by the norms of the text's genre. Each judge
  lists defects with a quote and a severity on a single scale — the impact on the reader: **minor** (only an
  attentive reader will notice; meaning and tone are intact), **noticeable** (hinders reading or breaks the tone; meaning is clear),
  **gross** (meaning is lost or the text does not read coherently, a sentence is cut off, garbage). A defect counts
  if **both** judges pointed to it in the same place, with the milder of the two ratings (the judges' nitpicks
  differ; real defects are seen by both); "gross" — if both found a gross defect; in a dispute the arbiter decides.
  Level: gross — 0; a noticeable defect or more minor ones than max(3, ⌈words/75⌉) — 0.5; only minor ones (or both
  judges say "reads worse") — 0.75; nothing — 1. Calibration on 24 texts × 15 variants
  (`scripts/calibrate_g3.py`, 2026-11-20, overall 89%): clean edits and paraphrases 62–100% → 1 (was 17–88%),
  2 typos 100% → 0.75, many typos and "thesaurus" words 96–100% → 0.5, cut-offs, shuffled
  order and garbage 100% → 0. **Coherence is a verdict on the text as a whole** ("can the reader follow the text: are the
  sentences connected, does the order make sense"), as a flag in G2: "no" from both judges — 0; in a dispute the arbiter decides
  (owner's decision 2026-11-20: not a structural crutch but the right question for the judges; in the list with quotes,
  incoherence was assessed locally and got softened).
- **D-45 ✅ Protecting the judges from injections (2026-11-20).** The miner's output is passed to the judges inside a
  `<document>` tag (opening and closing tags inside the text are neutralized in any letter case), and the response is
  strictly per a JSON schema. The prompts that evaluate the output say: text addressed to the reviewer
  (instructions, "checks passed", comments about the evaluation) is part of the output; evaluate it as content and do not
  change the verdict on the rest because of it. Such an insertion is itself garbage for G3 (both judges → 0) and an addition
  for G1; the result is the minimum across gates. `scripts/injection_check.py`: 5 kinds of attacks (a note to the reviewer,
  closing the tag, an insertion in brackets, fake JSON, hidden characters) on clean and corrupted outputs —
  the final level did not go up even once (90 out of 90).
- **D-44 ✅** Judges for G1–G3: GPT-6 Luna + Gemini 3.8 Flash, arbiter — Claude Sonnet 5.5 (`configs/gates.json`).
  Calibrations 2026-11-09: G1 (report 04b) — 0 false rejections out of 72 correct edits, distortions and insertions 24/24;
  G2/G3 by level (report 04c) — gross violations 24/24, minor ones 92–100%, natural paraphrase: G2 24/24 clean,
  G3 21/24 clean and 3/24 minor violation.
- **D-50 ✅** The human pool is large corpora **from before 2022** (millions of documents, Pile scale),
  guaranteed human. Narrow genres (essays, letters) are a known weak spot.
- **D-51 ✅** AI inputs depend on the pool **only through public metadata** (genre, topical
  cluster of ≥ 300 texts, length, register). No specific human text takes part in generation
  (otherwise a found source text would give a hint that does not exist in production).
- **D-52 ✅** Topics for AI inputs are generated by an LLM from the genre and the label of the pool's topical cluster (as
  implemented); the topic generator does not see the pool's texts. User texts and logs are not used
  (owner's decision 2026-11-09). Topics are needed so that AI inputs and the human reference set H match in topic within groups:
  otherwise a detector would distinguish them by topic rather than by "AI-ness".
- **D-53 ✅** AI inputs are generated by strong models (closed + open) via OpenRouter; the cost of
  generation is part of the owner's expenses.
- **D-54 ✅** Human reference set H per period: a sample from the pool with the same per-group counts as the slots;
  ~500 fresh texts + ~500 from the cache of already scored ones. Detector responses are cached.
- **D-55 🟡** Text length **150–350 words** (owner's decision 2026-11-09), ~230 on average in the pool; 4 ranges
  of 50 words each. Genres and proportions — Q-2.
- **D-56 ✅** A period is 4 days. D_t is hidden until D_{t+1} is assembled; then it is published (inputs, facts,
  IDs of the human texts, seed, outputs of all submissions, raw detector and judge responses, verdicts,
  receipts).
- **D-57 ✅** Seed: at the start of period t−1, `c_t = H(secret_t)` and the number of block B_t, roughly half a day
  before the end of t−1, are published; `seed_t = H(secret_t ‖ blockhash(B_t))`. B_t is fixed before its hash exists, so
  the seed cannot be picked. The dataset's Merkle root is published at assembly, before the first submission is evaluated: in
  `/v1/periods`, in the copy on HF and on chain. Both hashes are written to the chain: `System.remark_with_event` from a separate
  notary wallet that holds only fees (~τ0.00006 per record); the owner's key is not needed
  (`period/notary.py`). A commitment that did not land on chain before B is cancelled openly and published again.
  Rotation is automatic (`period/lifecycle.py`): commitment → assembly after B is finalized → the server prepares the
  period (H and the king's run) → switchover → reveal.
- **D-58 ✅** Before the reveal, the miner sees the decision, the stage and aggregates: S, the distribution of levels 1 / 0.75 / 0.5 / 0 (overall
  and for each gate), the contribution of each detector (E, AUROC, P_human). No outputs and no per-text levels, and
  no texts of the judges' reasons (they quote hidden inputs) — all of that is published with the period.
- **D-59 ✅** Languages: English only in v1. Arabic is a separate track later, after ≥ 2 detectors are admitted.

## Submission and execution

- **D-60 ✅** Submission = code + weights + dependencies (wheels), `bundle.tar` in **the miner's own HF repository**; on
  chain, the miner commits `<repo>@<40-hex commit>` from the submission's hotkey (owner's decision 2026-11-13): only
  that commit is taken (branches and tags, including `main`, are not accepted), the file's sha256 is read from the
  LFS metadata and checked on the pod; the commitment is made after the current registration; a private repo — via the
  miner's read-only token (we store it encrypted; it never reaches the pod). Contract:
  `humanize(items: list[{"text","lang","genre"}]) -> list[str]`. The base image is ours. We set the seed.
- **D-61 ✅** Resources (confirmed 2026-11-09): 1× H100 (≥ 80 GB), ~60 min per run, bundle ≤ **50 GB** (was 100 GB; owner's decision 2026-11-17: the king archive and the download time are halved; models of ~22B in bf16 and ~70–90B in 4 bits fit), license MIT/Apache-2.0
  or an explicit license to the owner.
- **D-62 ✅** Execution on **Lium, regular nodes** (owner's decision 2026-11-10; Intel TDX CVM nodes — when they
  become available with attestation). A new pod for each submission, deleted after
  the run. The miner's code runs in an inner container: `--network none`, non-root, no capabilities,
  `no-new-privileges`, strict seccomp, read-only rootfs; gVisor if it works with the GPU; a kernel
  version check. There are no keys or secrets on the pod.
- **D-63 ✅** The crown is instant, one run on D_t (owner's decision 2026-11-09). A two-stage crown
  (confirmation on D_{t+1}) is a fallback mode in the config, switched off. Protection against sandbox escape is the
  king's open code (D-70) and manual review under D-64. A free signal: the king's S on D_{t+1} (it is run
  anyway) is published next to the S at coronation; a sharp drop is a reason to check the code manually.
- **D-64 ✅** Incident (an attempt to escape the sandbox is found): void the period's results from this
  submission on, rebuild D_t early, re-score the king, continue the period.
- **D-65 ✅** Lium budget in the test phase — **$50**; the account is shared with other projects: we never touch other
  people's pods, we keep a minimum balance, we set a scheduled deletion on every pod,
  no more than one pod at a time, we do not trigger top-ups.
- **D-66 ✅** Submission intake (report 10): the miner's requests are signed with the hotkey (sr25519; ±5 min window,
  single-use nonce); only hotkeys registered in SN32 can submit (a metagraph snapshot no
  older than an hour); **one submission per hotkey registration** (a submission cancelled before launch, expired, or failed
  through our fault does not use up the registration); a king that lost its registration drops out of the history: the others
  shift up in the 85/10/5 payouts, and if it is the current king, the crown returns to the previous one (its run
  on the period is repeated at the owner's expense); one active submission per hotkey; queue by payment time, one evaluation at a time; the hold
  is opened at payment, cancellation before launch releases it; the bundle link — https, a host name (not an IP,
  not internal names), redirects only to https. Before the first submission of a period, the king is run on the
  new inputs at the owner's expense.

## Code openness

- **D-70 ✅** All evaluator code is open. The king's code and weights are published **48 h** (🟡) after
  coronation. The crown counts only after the bundle has been downloaded from the pinned commit into our
  archive and matched by sha256 (otherwise the crown is not awarded); king re-runs also come from the archive.
  The public link (`GET /v1/kings`) is the miner's repository at the pinned commit, if anyone can
  download it without a token (not private, not gated, no branch named after the commit) and the sha256 is unchanged;
  checked every 6 h. Otherwise — our copy from the archive in our public HF repository (`kings/<id>/`), forever
  (owner's decision 2026-11-17). We do not store uncrowned submissions. Initially, the owner checks the candidate's code for abuse against
  published criteria.
- **D-71 ✅** Leakage of D_t by the owner remains a matter of trust in v1 (stated honestly in the documentation); the assembler and the
  scorer are separate services without manual access to the period's data, so that they can later be moved into a TEE.

## Payment

- **D-80 ✅** A reserve via Stripe of **~$220** per submission (itemized: detectors, judges, Lium, fees) →
  settlement at actual cost under the public rates → the remainder goes to the hotkey's balance → refund on request to the original
  payment method (idempotent; withdrawal is blocked during a run). The scheme is the same as at SN34/Yugos.
  The report for a submission is itemized as "quantity × rate" (words per detector, judge tokens, pod hours);
  after the period is revealed, anyone can recompute the total. **No cryptographic proofs at launch**
  (a log with hashes on chain, spot re-checking of detectors — later, D-81). The full call log
  is kept by the owner (disputes/chargebacks). A public refund policy. The result is held back until
  the payment is confirmed. At launch, the period's shared costs are paid by the owner.
- **D-81 ⏳** Verifiability of reports (later): a call log with hash chaining, a hash on chain before the
  result is announced, a verifier script with spot re-checking of detectors.

## External services

- **D-90 ✅** All LLM calls (generation, judges, embeddings) go **only through OpenRouter**. For calls
  with hidden period data: a provider allowlist (`provider.only`), `allow_fallbacks: false`,
  `data_collection: "deny"`, `zdr` where supported; the provider of each response is logged. Judges
  are pinned to a model; if the model is unavailable, evaluation pauses.
- **D-91 ✅** All paid clients go through a local expense log with hard limits
  (`humeval budget`).
- **D-92 ✅** Validator integration: the humanizer share comes from a signed leaderboard. If there is no fresh one
  (server unavailable, invalid signature), the kings of the last valid leaderboard are paid, even after it
  expires (owner's decision 2026-11-21). Burn happens only at a validator that has not yet received any valid leaderboard.
  The leaderboard is signed with a separate ed25519 key (the public key is pinned in the
  validator code), validity 3 h, the version number never decreases; burn and the shares of unregistered hotkeys go to the
  subnet owner's UID (report 11).

## Open questions

| # | Question | Default |
|---|---|---|
| Q-1 | Source of topics (D-52) | ✅ LLM from the pool cluster label, as now; we do not take user data |
| Q-2 | Genres and proportions (D-55) | ✅ per production (2000 English texts, owner's labeling 2026-11-09): academic 39% (of which paper sections 85%, abstracts 12%), essays 15% (expository 49%, argumentative 24%, personal 17%, literary analysis 10%), application 7%, business 7%, forums 6%, homework answers 6%, articles 4.5%, fiction 2.2%, reviews 0.9%, news 0.7%; letters and "other" (reports, résumés, lecture notes) — outside the mix. The pool is rebuilt for this mix Pool v2 is assembled (report 02, "Pool v2"): essays are still ~78% argumentative — no more non-argumentative school essays from before 2022 are available. The owner accepted the skew of essays toward argumentative as is (2026-11-10). **application removed from the mix** (2026-11-10): the only source is essayforum, the date is known only at corpus level (2022-09) and there are traces of crude cleaning; it remains in the pool |
| Q-3 | Judges (D-44) | two closed models from different vendors via OpenRouter |
| Q-4 | δ_min (D-25) | 0 until the first data |
| Q-5 | Deposit size (D-80) | ✅ ~$220 (decided 2026-11-09) |
| Q-6 | Delay before publishing the king's code (D-70) | 48 h |
| Q-7 | Two-stage crown (D-63): gVisor does not start on Lium | ✅ instant, one run; we catch exploits through the open code (D-63, D-64) |
| Q-8 | Lium currently has no CVM nodes that passed TDX attestation (D-62) | ✅ we run on regular Lium nodes (owner's decision 2026-11-10): the risk — the node owner sees the current period's inputs; protection — recomputing the king on D_{t+1}, open code (D-63, D-70). CVM — when they appear |
| Q-9 | Text length: the pool is built for 150–350 words, the specification says 150–300 (D-55) | ✅ 150–350 |
| Q-10 | "Two previous kings" in 85/10/5 — distinct hotkeys | ✅ distinct hotkeys; the share of a missing previous king goes to the current king (D-12) |
| Q-11 | Narrow genres: essays = school PERSUADE/ELLIPSE, letters = Enron 2000–2001 | ✅ essays expanded (r/changemyview, C4 opinion columns, args.me; school essays — 35%). **Letters removed from the mix** (no personal correspondence from before 2022 was found; they remain in the pool) |
| Q-12 | G1 threshold and choice of judges | G1 v2 is calibrated (report 04b). A paraphrase by a strong LLM passes 24/24, Phrasly — 18/78 (the judges find real distortions and fabrications). With the D-22 levels, one added specification ("social media" → "Facebook") is level 0.5 (half the contribution), not zero. **Waiting for the owner to review** 12 examples. A change of person/tense in G2 stays at level 0.5 (decision 2026-11-10). The owner will look at the examples after G3 is tuned |
| Q-13 | Second mechanism in SN32 with 256 UIDs (D-14) | withdrawn 2026-11-21: there is no second mechanism, v4 is one weight vector (D-10); previously: 128 UIDs × 2 mechanisms |
| Q-14 | Where to store miners' bundles | ✅ in the miner's own HF repository + a commitment `<repo>@<commit>` on chain; the king's bundle is copied into our archive before the crown is credited (owner's decision 2026-11-13; the option with our HF organization was rejected) |

## Changelog

| Date | Decision | What changed | Why |
|---|---|---|---|
| 2026-11-08 | all | first version, based on the discussion | — |
| 2026-11-08 | D-62 | gVisor (runsc 20261102) does not start in Lium pods (Sysbox): neither via docker, nor `runsc do`, nor with the GPU. A hardened runc container remains; verified: `--network none` has no network, the GPU works | probe on 4 rentals, $0.035 |
| 2026-11-08 | D-62 | In Lium's public list, all 158 nodes have `tdx_attestation_passed=false` (47 have TDX hardware) | data from the `/executors` API |
| 2026-11-08 | D-90 | The provider allowlist is fixed in code (`TRUSTED_PROVIDERS`); Chutes/Targon are always excluded. Open models without a trusted provider (e.g. Llama 4 Maverick) are not used | live routing check |
| 2026-11-08 | D-44 | Judges default to `reasoning: low` with a token margin (reasoning ate the limit and truncated the JSON) | live runs |
| 2026-11-08 | D-53 | Generators: 12 models, some with their own reasoning settings (`configs/period.json`) | live probe: without this, DeepSeek/Qwen/Mistral returned empty text |
| 2026-11-09 | D-12 | Distinct hotkeys; the share of a missing previous king goes to the current one | owner's decision, as in SN34 |
| 2026-11-09 | D-80, D-81 | Billing on the SN34/Yugos scheme (reserve → actual → hotkey balance → refund on request), itemized report without proofs; proofs postponed to D-81 | owner's decision; analysis of Yugos |
| 2026-11-09 | D-20 | "Genre × length" groups do not need to be merged: noise is set by the total n, the per-group bootstrap is calibrated (0.98–0.99); H = 1:1, doubling H has no effect | simulation on real group sizes |
| 2026-11-10 | D-43, Q-2, Q-12 | G3 v2: form only and new defects only; comma splices — a separate flag `run_ons`, level 0.5. Person/tense in G2 stays 0.5; the essay skew is accepted. Calibration: splices 23/24 → 0.5, inserted detail 24/24 clean, the other variants as in v1 (report 04c) | owner's decisions; $2.43 |
| 2026-11-10 | D-24, D-70 | n = 500 instead of 750 (report 05: sensitivity ~25% worse, a submission a third cheaper); we train a simple reference humanizer — a sample of the submission format for miners | owner's decisions |
| 2026-11-10 | Q-2, D-55, Q-8, D-62 | application removed from the genre mix (the shares of the rest are renormalized); launch on regular Lium nodes without CVM, the ban on real data is lifted | owner's decisions |
| 2026-11-11 | D-22, D-58, D-62, D-80, D-90 | Full-cycle orchestrator: level-0 outputs are not sent to the detectors (their contribution is 0 anyway, the miner does not pay); one harness seed per period (resubmitting the same bundle does not re-roll the dice); before the reveal the miner sees only the stage, the decision and rounded aggregates, never text from their own code (a traceback could carry inputs out); bundle failure — the pod time is paid, our failure — the hold is released, judges/detectors unavailable — pause with the hold kept; results are copied from the pod only as regular files with a size limit | report 09 |
| 2026-11-13 | D-66, D-80, D-90 | Submission intake: hotkey signature, registered hotkeys only, one active submission per hotkey, queue by payment; hold at payment, cancellation before launch; pause — retry with a wait of 15 min ×2 (up to 6 h), after 8 pauses the evaluation is withdrawn without a charge; bundle link only https to a host name | report 10 |
| 2026-11-13 | D-92 | Signed leaderboard (ed25519, validity 3 h, the version never decreases); the validator sets mechanism 1 weights only when the subnet has two mechanisms; burn and the share of a deregistered king → owner's UID | report 11 |
| 2026-11-13 | D-14, D-92 | Localnet: two mechanisms can be enabled only when `max_allowed_uids × 2 ≤ 256` (SN32 currently has 256 UIDs); the validator sets mechanism 1 before mechanism 0 (the SDK checks the rate limit against mechanism 0); the 90/10 split and burn via the owner's UID confirmed with money | report 11 |
| 2026-11-13 | D-14, D-58, D-66, Q-13 | Owner's decisions: 128 UIDs × 2 mechanisms; one submission per hotkey registration, rollback when a king is deregistered; before the reveal the miner sees the level distribution (overall and per gate) and each detector's contribution (E, AUROC, P_human), but not the texts of the judges' reasons | owner's decisions 2026-11-13 |
| 2026-11-20 | D-43 | G3 calibration: severity on a single scale of impact on the reader (not tied to the defect type), only findings shared by both judges count (the milder rating), "gross" — by agreement of both or by the arbiter; no exception lists tailored to the samples | calibration 24×15, $3.7; Q-12 samples |
| 2026-11-21 | D-13 | Launch window: 3 days after the first miner crown the kings get 25% of the competition's share, 75% is burned; everything is burned while the baseline is king | owner's decision |
| 2026-11-21 | D-10, D-14, D-92 | 50/50 share (detection and humanizer); when the leaderboard is unavailable, the last kings are paid rather than burn; the evaluator code is the `humanizer/` folder in It-s-AI/llm-detection, branch `v4` | owner's decisions |
| 2026-11-21 | D-10, D-14, D-92, Q-13 | No second mechanism: validator v4 sets one weight vector, 90% detection and 10% humanizer from the signed leaderboard; without the embedded key, v4 sets weights like v3. Q-13 (128 UIDs × 2 mechanisms) withdrawn. Verified on the testnet 87 metagraph with a live leaderboard: after weight processing the humanizer share is 10%; the SN32 owner is registered (UID 129), so there is somewhere to send the burn | owner's decision |
| 2026-11-20 | D-57 | The commitment and the Merkle root are written on chain as a `remark` record from the notary wallet; `verify_period.py` checks that the commitment landed before block B, and the Merkle root before the first evaluation | owner's decision; verified on testnet |
| 2026-11-20 | D-56, D-57, D-70 | Automatic period rotation: commitment and block B at the start of the period, assembly after B is finalized, preparation of the next period on the server (H is scored once and cached, the king is run in advance), switchover, reveal of the closed period in `/v1/periods/<id>/` and a check with `verify_period.py`. The bundle archive is cleaned up: kings with a live registration stay, the rest are mirrored first | tasks for production; behavior covered by tests |
| 2026-11-20 | D-22, D-42, D-43 | G2 checks only language and genre (person/tense are no longer penalized). G3 v3: judges list defects with a quote and severity; new level 0.75 — only a few minor flaws; 0.5 — a noticeable defect or more minor ones than max(3, ⌈words/75⌉); the arbiter decides within the range between the judges | owner's decisions on the Q-12 samples |
| 2026-11-17 | D-61 | Bundle ≤ 50 GB instead of 100 GB: checked at submission, on the pod and when copying to the archive; the server disk is 300 GB | owner's decision |
| 2026-11-17 | D-80 | Payment without a website: the submit command issues a Stripe link valid for 60 minutes (card only), the miner pays and confirms with the `paid` command; the server also checks the links itself; a payment made before the link expires counts even after the quota expires; a webhook is not required | owner's decision; report 10 |
| 2026-11-17 | D-70 | Publishing the king after 48 h: a link to the miner's repository at the pinned commit; if the commit cannot be downloaded without a token — our copy from the archive in our HF repository | owner's decision; `publish/kings.py` |
| 2026-11-13 | D-60, D-70, Q-14 | Bundle in the miner's own HF, on chain `<repo>@<commit>`; only that commit, branches are forbidden; an identical bundle from different hotkeys — once; the crown — after the bundle is copied into our archive, publication and re-runs of the king — from the archive | owner's decision; report 12 |
| 2026-11-09 | D-22 | Quality levels 1 / 0.5 / 0 per output instead of "caught or not"; the 95% threshold for the crown removed; G2/G3: a flag counts by agreement of the judges or by the arbiter's decision | owner's decision: a humanizer that improves some of the texts is useful to the product, even if it breaks others |
| 2026-11-09 | D-52, D-55 | Topics — LLM from the cluster label (no user data); length 150–350 words | owner's decisions Q-1, Q-9 |
| 2026-11-09 | Q-11, Q-2, D-61 | Letters removed from the genre mix; miner resources confirmed; the owner measures the genre distribution in production | owner's decisions |
| 2026-11-09 | Q-2, D-55 | Genre mix — by the distribution from production; in academic — paragraphs of full paper texts, in essays — expository/personal/literary analysis; new genres application, business_marketing, homework_answer | owner's measurement: 2000 texts, Haiku + Flash-Lite + Sonnet as reference |
| 2026-11-09 | D-42, D-43, D-44 | G2/G3: levels 1/0.5/0, flag by agreement of the judges or by the arbiter; calibration 24 inputs × 12 variants | $5.86, report 04c |
| 2026-11-09 | D-42 | Register and format excluded from G2 (`g2_ignore`); reference v1: 51/78 pass fully (was 36), Phrasly unchanged (all of its outputs have violations in G1/G3) | owner's decision |
| 2026-11-09 | Q-12 | G1 diagnostics (output = input, 20 texts): 17/20 pass; 21 misses out of ~1100 checks; 11 — checker without context, 6 — quote matching, 3 — bad facts, 1 — technical. Plan for G1 v2 | paid run $0.06 |
| 2026-11-09 | D-63 | The crown is instant, without confirmation on D_{t+1}; added the free signal "the king's S on the next period" | owner's decision: only the current period's inputs can leak, and that gives no permanent access |
| 2026-11-09 | D-41, D-44 | G1 v2 instead of v1; judges Luna + Gemini, arbiter Sonnet 5.5; judge response cache | calibration 24 inputs × 10 variants ($4.87), Phrasly on 78 outputs ($2.69 judges, $3.55 Phrasly) |
| 2026-11-09 | D-50, Q-11 | Full pool: 239 431 texts, 8 genres × ~30 000, 259 clusters ≥ 300 (D-51); new sources of essays and letters | assembly ~14 min, free |
| 2026-11-09 | Q-2, D-50, D-55 | Genre proportions per production; new genres application, business_marketing, homework_answer; academic = paper paragraphs (88%) + abstracts (12%); essays + expository/narrative/literary; candidate cache 100–500 words, the pool window is an assembly parameter (default 150–350). Pool v2: 320 021 texts, 11 genres, 344 clusters ≥ 300 | owner's measurement; assembly free (HF + local) |
| 2026-11-09 | D-20 | `E_d` = mean of the AUROC term and the mean "human" probability over outputs (weight 0.5) | owner's decision: AUROC against humans gives no gradient while the outputs are above all humans; on It's AI, Phrasly without gates: S 0.08 → 0.32 |
