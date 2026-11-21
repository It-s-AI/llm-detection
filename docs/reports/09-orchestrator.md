# 09. Full-cycle orchestrator — report

## What was done
- **One submission from start to finish** (`src/humanizer_eval/evaluation/`):
  quote → hold → bundle run on a fresh Lium pod → G0 → gates G1–G3 (levels 1 / 0.5 / 0) → crown detectors →
  S and the crown test against the king → statement of actual costs → charge against the hold.
- **Records.**
  - The miner sees `status.json`: stage, decision, S, the king's S and the lower bound, all rounded to 4 digits. They can also see `statement.json`.
  - Outputs and the full record (judge verdicts, raw scores, pod logs) are stored encrypted until the period is revealed.
- **Failures.**
  - Our failure (pod could not be rented, image failed to build, pod lost): the hold is released, nothing is charged.
  - Bundle failure (did not download, wrong hash, code crashed, no outputs): the submission is evaluated as is; missing outputs get level 0. Only pod time is charged.
  - Judges or detectors unavailable (more than 5% of calls failed after a retry, or the spending limit was reached): the evaluation is **paused**, the hold is kept. A repeated call resumes from the saved outputs without a new pod.
- **Bundle by link.** The miner provides an https link and a sha256. The pod downloads the file itself, verifies the hash and the tar headers, and only then unpacks it. The 8 GB do not pass through our server.
- **Pod.**
  - The job is started in the background (`nohup`) and polled every 30 s, so a dropped ssh connection does not kill an hour-long run.
  - If a node refuses the rental (for example, "rented only as a whole"), the next one is taken, up to three attempts. Single-GPU nodes go first.

## Mapping to decisions
| Decision | Where in code | How verified | Status |
|---|---|---|---|
| D-11, D-23 crown against the king, paired bootstrap | `pipeline.py` → `crown_decision`, seed derived from the period seed | test, pilot | ✅ |
| D-22 levels; a level-0 output counts as caught | `pipeline.py` | test: such outputs are not sent to detectors | ✅ |
| D-40 G0 (normalization, 1.3× truncation) | `_g0` | pilot: levels matched report 07 | ✅ |
| D-57 reproducibility | harness and bootstrap seed from `seed_t` | — | ✅ |
| D-58 only aggregates before reveal | `SubmissionStore`, public fields by allowlist | test: error text from the miner's code does not get into the status | ✅ |
| D-60–D-62 fresh pod, hardened container | `runner/evaluate.py: run_pod_job` | Lium: no network, 78 outputs | ✅ |
| D-80 hold → itemized statement → charge ≤ hold | `pipeline.py` + `billing` | Stripe in test mode | ✅ |
| D-90 pause when judges are unavailable | `EvaluationPaused` | test: pause, then resume without a second run | ✅ |

## Deviations and new rules
- **Level-0 outputs are not sent to detectors.**
  - Their contribution to both terms is zero anyway, so the score does not change.
  - The miner does not pay for broken texts.
  - The price of the rule: the word total in the statement shows how many words went to detectors. This is an aggregate, roughly the same information as in S.
- **One harness seed per period.** The same bundle gives the same outputs: you cannot "reroll the dice" with a repeated submission. All calls of such a repeated submission hit the cache, and only the pod is paid for.
- **The cache is free for the miner.**
  - A text already scored by a detector and a judge verdict already obtained are not paid for a second time (as was already the case in the policy).
  - After a pause, whatever was paid for before it hits the cache on resume. The owner absorbs that money.
- **Pod security.**
  - The miner's code can write anything into `/out`, for example a link to `/dev/zero` or a 100 GB file.
  - So a trusted external script copies only regular files from there, with a size limit (outputs up to 64 MB, meta up to 1 MB).
  - `pod.json` is written outside the mounted folder.
- **Miner code errors do not get into the status.** A traceback may contain the period's hidden inputs, so the status only says "entrypoint failed".

## Live example (pilot 1, 78 inputs, It's AI detector: demo, it does not decide the crown, D-33)
| Submission | Where the outputs come from | Levels 1 / 0.5 / 0 | S | Crown | Charged |
|---|---|---|---|---|---|
| identity baseline (king at start) | inputs unchanged | 78 / 0 / 0 | 0.0086 | — | owner |
| reference v2 | saved outputs (replay) | 36 / 37 / 5 | 0.0424 | **yes**: ΔS = +0.034, lower bound +0.024 | $0 (all from cache) |
| identity bundle on Lium | RTX 3090, 8.4 min | 78 / 0 / 0 | 0.0086 | no: ΔS = −0.034 | $0.35 (0.14 h at the H100 rate) |

- The hold for 78 texts is $27.84. The pod cost for the owner is $0.035. The hold was topped up with a Stripe test card.
- The S of reference v2 here is 0.042, whereas report 07 had 0.044. The reason: the pilot was built from the old pool, and `score_outputs.py` silently took only 44 of the 78 human texts. Now 71 texts from two pools are used, and the script prints how many it found. The production period is strict: if even one text of H is missing, it fails.

## Test results
`tests/evaluation/` — 12 tests:
- crown and statement;
- a repeated submission pays only for what is new;
- insufficient balance: no pod is rented;
- our failure: hold released;
- judge pause and resume;
- detector limit;
- crashed bundle without leaks;
- owner run without billing;
- bundle link rules;
- safe copying from the pod;
- rental retry on another node.

The full test suite is green.

## How to check it yourself
```
python scripts/evaluate_submission.py --replay ref_v2 --detectors its_ai --billing test --allow-missing-human
python scripts/evaluate_submission.py --bundle var/lium/identity.tar.gz --cheap-gpu --yes --detectors its_ai \
    --billing test --allow-missing-human --no-update-king
```

## What you should look at
- Do you agree with the rule "level 0 is not sent to detectors": it is cheaper for the miner, and the statement shows the word total.
- What to show the miner before reveal. Currently it is S, the decision, the king's S, ΔS, the lower bound and the amount charged. Each detector's contribution is hidden until reveal.

## Risks and what next
- The crown on real detectors (GPTZero, ZeroGPT, Pangram) is waiting for GPTZero and ZeroGPT to be topped up.
- The submission queue and the miner API (accepting a link, quote, payment, status) are the next step, together with integration into the validator.
- The statement after a pause is understated by what was paid for before it. If there are many pauses, intermediate accounting needs to be saved.
