# 03. Period builder + pilot + detector admission test — report

## What was done
- **Commit-reveal seed** (`period/seed.py`): `c_t = sha256(secret)`, `seed = sha256(secret ‖ blockhash)`,
  named sub-seeds for each random choice (the release can be re-verified).
- **Merkle root** (`period/merkle.py`) over all dataset items, inclusion proofs.
- **Encryption** (`period/crypto.py`): the dataset is on disk only in encrypted form (SecretBox); inputs for the pod —
  SealedBox to its key.
- **Generation** (`period/generate.py`): topics (the model sees only the genre and the cluster label), 12 generators by
  weight, "user-like" prompts across 8 genres, wrapper cleanup, length control per slot, custom
  reasoning settings for some of the models. Everything goes through HIDDEN routing.
- **Period build** (`period/build.py`): slots from the pool → topics → inputs → facts → human reference set H with the same groups →
  Merkle → encrypted dataset + public manifest. Adapter to the phase 2 pool (`period/pool_adapter.py`).

## Pilot (real phase 2 pool, 80 slots)
| Metric | Value |
|---|---|
| AI inputs generated | 78 of 80 (2 did not fit the length on the second attempt) |
| Human reference set H | 78 texts, all with an exact group match (genre × cluster × length × register) |
| Mean length | AI 260 words, humans 240 |
| Genres | essays 23, articles/blogs 12, letters 8, reviews 8, forums 8, news 8, academic 7, fiction 4 |
| Generators | all 12: DeepSeek V4 Pro 15, GPT-5.4 14, GPT-5.4 mini 11, Qwen 3.8 7, Gemini 3.6 Flash 6, Claude Haiku 5.5 6, GPT-6 Sol 5, Qwen 3.7 Plus 5, Mistral Large 4 3, Claude Sonnet 5.5 2, Grok 4.7 2, Gemini 3.5 Flash 2 |
| Generation cost (topics + texts + facts) | **$0.44** for 80 slots (~$0.0056 per input) |

## Detector admission test (D-31): AUROC of raw AI inputs against the human reference set
| Detector | AUROC (by group) | Admitted (≥ 0.9) | Weak genres | Mean score AI / humans |
|---|---|---|---|---|
| GPTZero | **1.000** | ✅ | — | 1.00 / 0.00 |
| Pangram | **1.000** | ✅ | — | 1.00 / 0.00 |
| ZeroGPT | **0.967** | ✅ | fiction 0.83, forums 0.88, news 0.92 | 0.64 / 0.03 |
| It's AI (proxy, not in the crown) | 1.000 | — | — | 0.98 / 0.03 |

Conclusions:
- All three crown detectors are admitted; GPTZero and Pangram separate raw AI texts perfectly — so at the start
  of the competition a "change-nothing" king will get S ≈ 0, and any real evasion will be visible.
- ZeroGPT is noticeably weaker (especially on fiction and forums) — in the soft minimum it will be the "easy"
  detector; this is fine, but it is worth watching whether it becomes the only one being evaded.
- For choosing the primary score: for GPTZero `p_ai_plus_mixed` = `p_ai` (the mixed class does not occur on raw AI);
  for Pangram all variants give 1.0 — the difference will show up only on humanized texts.

## Pilot spending
Detectors: Pangram ~$23, GPTZero ~$5.9, ZeroGPT ~$2.7 (per the ledger's upper-bound estimates); generation $0.44.

## Not done / limitations
- The block hash is a stub for now (commits are not written to the chain — a testnet wallet is needed).
- The topic source is for now only the model working from the cluster label; It's AI logs / fresh content — awaiting a decision (Q-1).
- The pool leakage test (closeness of inputs to the pool vs "production-like" texts) — not run yet.

## How to reproduce
```bash
python scripts/pilot_period.py --pool <pool_dir> --n 8          # cheap run: It's AI only
python scripts/pilot_period.py --pool <pool_dir> --n 80 --yes    # with paid detectors (~$30)
```
