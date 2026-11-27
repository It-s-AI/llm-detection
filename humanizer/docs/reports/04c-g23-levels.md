# 04c. G2/G3: quality levels and the arbiter — calibration

`scripts/calibrate_g23.py`: the same 24 inputs as in the G1 calibration (3 per genre), variants with a known level,
judges from `configs/gates.json` (GPT-6 Luna + Gemini 3.8 Flash, arbiter Claude Sonnet 5.5). Cost $5.86.
Unchanged outputs and typography-only edits never reach the judges (shortcut in `GateRunner`).

The production rule is **arbiter**: a flag counts if both judges set it, and on disagreement the arbiter decides.
For comparison: **both** — agreement of both only (no arbiter), **any** — a flag from either judge (the v1 rule).

## G2 (style): share of the correct level, in parentheses — share of "has a violation"
| Variant | Expected | arbiter | both | any |
|---|---|---|---|---|
| light synonym edit | 1 | 0.96 | 1.00 | 0.96 |
| paraphrase by a strong LLM | 1 | **1.00** | 1.00 | 1.00 |
| register change | 0.5 | 0.92 | 0.92 | 1.00 |
| person/tense change | 0.5 | 1.00 | 1.00 | 1.00 |
| format change | 0.5 | 1.00 | 1.00 | 1.00 |
| translation into Spanish | 0 | 1.00 | 1.00 | 1.00 |
| genre change | 0 | 0.58 (0.92) | 0.46 (0.88) | 0.67 (0.92) |

## G3 (naturalness)
| Variant | Expected | arbiter | both | any |
|---|---|---|---|---|
| light synonym edit | 1 → actually 0.5 | 0.21 (flag 0.79) | 0.71 | 0.17 |
| paraphrase by a strong LLM | 1 | **0.88** (0.12 — minor) | 1.00 | 0.58 |
| typos | 0.5 | 1.00 | 1.00 | 1.00 |
| odd synonyms | 0.5 | 0.96 | 0.96 | 0.92 |
| truncated phrases | 0 | 1.00 | 1.00 | 1.00 |
| shuffled sentences | 0 | 1.00 | 0.96 | 1.00 |
| garbage ("As an AI language model…", markup) | 0 | 1.00 | 1.00 | 1.00 |

## Conclusions
- **Gross violations are always caught** (language, truncated phrases, incoherence, garbage — 24/24); minor ones — 92–100%.
- **The "light edit" turned out not to be clean:** Grok replaced words mechanically ("notice → register", "trap → snare",
  "uses terrible → employs awful"). The judges flag "odd words" — level 0.5, not 0. This is exactly the intended
  behavior: thesaurus substitution is a typical trick of crude humanizers, and it earns half the contribution.
- **Natural paraphrase:** 21/24 pass, 3/24 get a minor violation (half the contribution) — the price of the rule
  with the arbiter. The "both judges only" rule is softer here (24/24), but lets through more odd words (0.71 vs
  0.21 on the mechanical edit). We keep the arbiter — same as in G1.
- **Genre change** is recognized as a violation in 92% of cases, but as a gross one only in 58% (in the rest the judges see a change of
  format/register → 0.5). For the miner this is still a loss of contribution; genre substitutions are rare.

## Change 2026-11-09: register and format excluded from G2
Owner's decision: changing formality and format (restructuring paragraphs, a list rewritten as comma-separated
text) is the essence of a humanizer's work; this is level 1. G2 keeps language and genre (0) and person/tense (0.5).
The excluded items do not count as judge disagreements, and the arbiter is not called for them (`g2_ignore` in `configs/gates.json`).
Garbage markup is still caught by G3 ("garbage"). Effect (recomputed from the cache of judge responses, $0):

| | Fully passed (level 1) | G2 = 1 |
|---|---|---|
| Phrasly | 0 → 0 (all outputs have violations in G1 or G3) | 34 → 56 |
| Reference humanizer v1 | 36 → **51** (0.5: 18, 0: 9) | 54 → 70 |

## Change 2026-11-10: G3 v2 — form only, comma splices → 0.5
Owner's decision. What changed in the G3 prompt (`NATURAL_SYSTEM`, flags `DEFECT_KEYS`):
- the judges assess **only form**: changed, added or omitted facts are G1's job;
- **only defects that are absent from the input** are flagged;
- new flag `run_ons` (two sentences joined by a comma without a conjunction, comma splice) — **minor violation 0.5**;
  "truncated phrases" (0) still covers fragments and unfinished or unintelligible sentences;
- conversational style is not a defect: contractions, short sentences, a short phrase for emphasis.

Calibration with `scripts/calibrate_g23.py` on the same 24 inputs + 3 new variants: `splice` (three pairs of adjacent
sentences joined with a comma), `contradict` and `insert` from the G1 calibration (one fact inverted / one
invented detail). Cost $2.43 (G2 responses from cache).

| G3 variant | Expected | v1 (arbiter) | **v2 (arbiter)** |
|---|---|---|---|
| light synonym edit | 0.5 in practice | 0.5: 19/24 | 0.5: 20/24 |
| paraphrase by a strong LLM | 1 | 21/24 | **21/24** (3 — "odd words") |
| typos | 0.5 | 24/24 | 24/24 |
| odd synonyms | 0.5 | 23/24 | 24/24 |
| **comma splices** | 0.5 | — | **23/24** (1 — level 0) |
| inserted invented detail | 1 (G1's job) | — | **24/24** |
| inverted fact | 1 (G1's job) | — | 8/24; 15/24 — "incoherence" (0) |
| truncated phrases | 0 | 24/24 | 24/24 |
| shuffled sentences | 0 | 24/24 | 24/24 |
| garbage | 0 | 24/24 | 24/24 |

- **Splices** now cost half the contribution, not all of it.
- G3 no longer touches **added details** (24/24 clean).
- G3 treats an **inverted fact** as incoherence in 15 cases out of 24. This is not a mix-up with meaning: the inverted
  sentence contradicts its neighbors *within* the output ("willpower is underrated", and then the text argues the
  opposite; "Pytest does not provide… which reports the fastest tests"). The reader sees broken logic —
  level 0 is fair. The expectation of "1" for this variant was wrong.
- G2 is unchanged (responses from cache): register and format are not checked; in 9/24 "register changes" the judges
  also see a change of person (the conversational version switches to "you") → 0.5.

Recheck of the pilot outputs with G3 v2 (G1/G2 from cache, $1.38):

| | Levels before (1 / 0.5 / 0) | After | "Splices" flag |
|---|---|---|---|
| Phrasly | 0 / 23 / 55 | 0 / 26 / 52 | 11 |
| Reference humanizer v1 | 51 / 18 / 9 | **53 / 22 / 3** | 15 |

For Phrasly, 12 outputs rose from 0 to 0.5 in G3 and 5 dropped; there are still no fully clean ones — G1 holds them back.
For reference v1, there are three times fewer zeros: previously splices were counted as "truncated phrases".
