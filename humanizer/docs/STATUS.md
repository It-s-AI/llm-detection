# Status (2026-11-20)

Branch `integration` = all phases together, offline tests green. PRs by phase: #1 detectors, #2 LLM gateway,
#3 scorer, #4 Lium client, #5 gates, #6 pool, #7 harness, #8 period builder.

| Part | Status | Where |
|---|---|---|
| Detectors (Pangram, GPTZero, ZeroGPT, It's AI) | ✅ cache, spending caps | report 01 |
| OpenRouter gateway, verdict cache | ✅ | report 01b |
| Pool v2: 320k texts from before 2022, genres matching production | ✅ application and letters excluded from the mix | report 02 |
| Period builder (topics by cluster, facts) | ✅ pilot of 78 texts; **n = 500** in production | report 03 |
| Gates G0–G3, levels 1 / 0.75 / 0.5 / 0 | ✅ G1 v2; G2 — language and genre only; G3 v3: integrity of the whole text, only defects found by both judges count; calibration 89%; injection protection 45/45 | reports 04b, 04c, `reports/data/13b-*` |
| Metric D-20/D-22, bootstrap, crown | ✅ power recomputed (A/A 5.5% ± 0.4%) | report 05 |
| Lium + harness | ✅ regular nodes (no CVM) | reports 06, 06b |
| Billing (Stripe, test mode) | ✅ quote → payment → hold → charge → refund | report 08 |
| Reference humanizer v2 (submission format) | ✅ trained on Lium ($0.60), levels 36/37/5 | `examples/ref_humanizer`, report 07 |
| Full-cycle orchestrator | ✅ quote → hold → pod → gates → detectors → crown → statement; pause on failures | report 09 |
| Submission intake: API for miners, queue, worker | ✅ hotkey signature, SN32 hotkeys only, Stripe Checkout, king on a new period; e2e over HTTP | report 10 |
| Humanizer weights in the validator | ✅ v4: a single weight vector, detection 50% + humanizer 50% from the signed leaderboard (on failure — the last kings); branch `v4` in It-s-AI/llm-detection; previously — branch `llm-detection-mechanism1`; localnet: weights, commit-reveal, 90/10 split, burn ✅ | report 11 |
| Bundles on the miner's HF + on-chain commitment; crown after copying into our archive; one submission per registration | ✅ | report 12 |
| King publication after 48 h: the miner's repository, otherwise our copy on HF (`GET /v1/kings`) | ✅ tests; copy to our HF mirror verified live (report 14) | `publish/kings.py`, report 12 |
| Humanizer section in the Space `sergak0/ai-detection-leaderboard`: payouts, period results, kings; public `GET /v1/results` | ✅ PR #1 merged 2026-11-17, Space running, both parts in place; data will appear after `HUMANIZER_API` | `space/humanizer_tab.py`, `scripts/publish_space.py` |
| Live check of the miner path on testnet 325: HF (public and private), commitment, signed submission, Stripe link + `paid`, Lium pod, gates, It's AI, statement | ✅ two submissions passed | report 13 |
| Test launch: `test` mode (flag in `/v1/info` and banner in the Space, test Stripe only, cap on evaluations per day), `deploy/` via env variables (testnet 87 / finney 32), HTTPS by IP; English READMEs, rules, validator guide, draft announcement | ✅ | `docs/rules.md`, `docs/validators.md`, `docs/announcements/` |
| Test server at `188.245.168.78` (volume, no apt): HTTPS by IP (Let's Encrypt), testnet 87, pilot-1, It's AI, test Stripe, 10 evaluations/day; Space section connected (`HUMANIZER_API`) | ✅ running | `deploy/README.md` |
| Miner guide, deployment kit (systemd, Caddy, env template), state backups | ✅ verified locally | `docs/miners.md`, `deploy/` |
| Telegram alerts (@alarmer_bot, key in env) and a watchdog every 10 minutes: API, TLS, metagraph, disk, stuck evaluations, backups, caps, OpenRouter, Lium | ✅ running on the test server | `ops/alerts.py`, `scripts/watchdog.py` |
| Parallel evaluations (N pods), the crown is decided one at a time against the current king | ✅ tests; rehearsal: two submissions on two pods simultaneously | `intake/worker.py`, report 14 |
| Period rotation: commitment and block B, build, preparation on the server (H once, king in advance), switchover, reveal `/v1/periods/<id>/`, check with `verify_period.py`; cleanup of the king archive | ✅ tests; rehearsal on the server: 3 periods, the pilot-2 reveal passed 25/25 checks; timer disabled until production | `period/lifecycle.py`, `period/reveal.py`, `scripts/rotate_period.py`, report 14 |
| On-chain notary: commitment and Merkle root as a `remark` record from a separate wallet, check of ordering relative to block B and the first evaluation | ✅ tests; testnet and rehearsal on the server | `period/notary.py`, `scripts/notary_post.py` |
| Write the leaderboard URL and key into the validator branch, after first checking the live leaderboard | ✅ script | `scripts/release_validator.py` |

## Remaining
Waiting for the owner:
- company accounts (Lium, OpenRouter, detectors, Stripe live) and top-ups;
- production server; state migration — per `deploy/README.md`, section "Moving to the production server";
- launch on finney: a second mechanism is not needed (v4 — a single weight vector); the owner releases v4 with the leaderboard key and URL built in (`release_validator.py`), validators update;
- merge the validator branch into llm-detection, after first writing in the URL and key with the `release_validator.py` script;
- open the repository, publish the announcement.

Switching to production (together with the owner): `--mode prod`, finney 32, live Stripe, three detectors, `--parallel N`, `--mirror-repo`, top up the notary wallet (~τ0.1) and enable the `humeval-rotate` timer (manually: every period costs money).

The public copy of the evaluator code is the `humanizer/` folder in branch `v4` of It-s-AI/llm-detection (squashed subtree). Work happens here; after changes: `git subtree pull --prefix=humanizer <this repository> integration --squash` in llm-detection on `v4`, then push.

Can be done without the owner: an off-machine copy of the backups (storage needed).

## Spending (ledger, upper-bound estimates)
Pangram $188.90 (cap $314), GPTZero $55.56 (cap $65.5 = quota until December 8 + approved overspend),
ZeroGPT $25.76 (cap $30.4), OpenRouter $59.98 (cap $140), Phrasly $4.34, Lium $0.95 (cap $50).
Other people's Lium pods were not touched.

## Period p1 (2026-11-17)
484 texts (of 500 requested), 12 generators, the secret commitment was published before block B = 9526464
(`docs/periods/README.md`). Generation $5.63, G1 facts ~$11 (OpenRouter). The human texts and the "no changes"
king were scored on GPTZero / ZeroGPT / Pangram: AUROC 1.00 / 1.00 / 1.00, king's S = 0.
Detectors on p1: GPTZero $34.6, ZeroGPT $16.1, Pangram $137.3.

Challenger test on p1 (It's AI, detectors free): a single-pass LLM rewrite
(`make_replay.py`, $0.86) against the "no changes" king. S = 0.0442 vs 0.0109, ΔS = 0.0334, lower
bound 0.0296 > 0 → **crown awarded**. Levels 1 / 0.5 / 0: 457 / 25 / 2. Judges cost $6.66 (the quote
had budgeted $19), 666 cents charged from a hold of 2311, the remainder was returned. The evaluation took 45 minutes, almost all of it judges.

## Crown on the pilot with real detectors (2026-11-17)
Reference v2 against the "no changes" baseline, 78 texts: AUROC GPTZero 1.00, Pangram 1.00, ZeroGPT 0.997,
S ≈ 0, crown not awarded (correct: the reference was trained against It's AI and does not evade the real detectors).
The full cycle on three paid detectors works; it cost $16.45 (GPTZero $2.85, Pangram $12.25, ZeroGPT $1.35).
