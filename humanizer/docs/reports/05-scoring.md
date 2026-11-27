# 05. Scorer and crown statistics — report

Branch `phase-5-scoring`. Design note: [`docs/design/05-scoring.md`](../design/05-scoring.md)
(implemented autonomously; the note should be confirmed together with this report).

## What was done (3–5 points in plain language)
1. **Metric** (`scoring/metric.py`): each detector's AUROC is computed within "genre × length" strata
   and averaged with weights by the number of outputs. This is the exact Mann–Whitney statistic (ties = 0.5), computed via
   sorting, without enumerating all pairs. Then `E_d`, soft-min with `p = −4`, and the final `S`. An output
   that did not pass a gate gets the score `+inf`, i.e. it is caught by all detectors.
2. **Paired bootstrap** (`scoring/bootstrap.py`): in each replicate, inputs are resampled within
   strata jointly for the king and the challenger; H is also resampled within strata. The code
   is vectorized: a single decision at n = 750, m = 1000, B = 2000 and 3 detectors takes ≈ 0.1 s.
   With the same seed the result is identical.
3. **Crown and payouts** (`scoring/crown.py`): `crown_decision` returns the decision and a
   PASS/FAIL line for each criterion: bootstrap bound > δ_min, challenger's gates ≥ 95%.
   `KingState` stores the coronation history and computes 85/10/5 over the current king and the two previous kings
   with distinct hotkeys; it gives the missing shares of previous kings (and baseline shares in those slots) to the current king,
   and the share of a current baseline king to `__burn__` (D-12, decision 2026-11-09); it is saved to JSON.
4. **Simulator** (`scoring/simulate.py`, `scripts/simulate_crown.py`) on synthetic periods:
   shared text difficulty, model skill, correlation φ, detector noise, score rounding (ties),
   stratum effects. The script computes A/A (the false crown rate), power curves, and shows
   where the bootstrap variance comes from.
5. **Tests:** 82 new tests, plus 1 long one under the `slow` marker. AUROC is checked against `sklearn`,
   the fast bootstrap against naive resampling (agreement to 1e-12). A regular `pytest` run
   passes in ≈ 17–21 s.

## Mapping to decisions
| Decision | Where in code | Verified by | Status |
|---|---|---|---|
| D-20 AUROC by strata, weights by number of texts, `E_d` | `metric.stratified_auroc_detail`, `metric.evasion` | `test_auroc_matches_sklearn_random`, `test_auroc_hand_computed_ties`, `test_stratified_weights_by_outputs`, `test_stratified_matches_sklearn_per_stratum`, `test_stratification_removes_genre_offsets`, `test_strata_without_humans_are_excluded_and_reported`, `test_evasion` | ✅ |
| D-21 soft-min `p = −4`, any `E_d = 0` ⇒ `S = 0` | `metric.softmin`, `metric.submission_score` | `test_softmin_*`, `test_submission_score_combines_detectors` | ✅ |
| D-22 gate failure = caught; crown only with ≥ 95% passed | `metric.apply_gate`, `SubmissionRun.gated_scores`, `crown.crown_decision(min_gate_pass=0.95)` | `test_gate_failures_are_caught_by_every_detector`, `test_gate_pass_rate_blocks_crown`, `test_gate_threshold_is_inclusive`, `test_king_gate_rate_is_reported_not_used` | ✅ |
| D-23 paired bootstrap (inputs jointly, H too), one-sided α = 5% | `bootstrap.paired_bootstrap`, `bootstrap.ResamplingPlan` | `test_fast_replicates_equal_naive_resampling`, `test_resampling_is_within_strata`, `test_deterministic_given_seed`, `test_identical_outputs_never_crown`, `test_small_aa_false_crown_rate_and_se_calibration`, `test_coverage_of_lower_bound_under_alternative`, `test_aa_false_crown_rate_at_production_size` (slow), A/A in the script | ✅ |
| D-24 all texts through all detectors, n ≈ 750 | `metric.check_detectors` (the submission's detector set = the H detector set); n = 750 in the simulation | `test_detector_mismatch_raises`; A/A at n = 750 | ✅ |
| D-25 δ_min | parameter `delta_min` (default 0, Q-4), strict `>` | `test_delta_min_raises_the_bar`, `test_crown_is_strict_inequality`; the `--delta-min` flag in the script | ⏳ value — after the first periods |
| D-11 challenger vs. king on the same inputs | `crown_decision(king, challenger, human)`, `bootstrap.check_paired` (strata and `input_ids` must match) | `test_misaligned_runs_rejected` | ✅ (queue by commit — not in this phase) |
| D-12 85/10/5: current + two previous kings | `KingState.payout_slots`, `KingState.payout_weights` | `test_85_10_5_with_growing_history`, `test_previous_kings_are_distinct_hotkeys`, `test_same_hotkey_crowning_itself_twice` | ✅ (interpretation of "distinct" — below) |
| D-13 the baseline share is burned | `KingState.initial`, `KingEntry.is_baseline` → `__burn__` | `test_baseline_king_share_is_burned`, `test_empty_state_burns_everything` | ✅ |

## Deviations from the specification and why
1. **`crown_decision` signature.** The spec had `(king, challenger, alpha, delta_min, min_gate_pass)`.
   Added the mandatory `human` (AUROC cannot be computed without the human reference set H) and the bootstrap parameters
   `n_boot`, `seed`, `p`. `king` and `challenger` are `SubmissionRun` (per-detector scores,
   `gate_pass`, strata, optionally `input_ids`).
2. **"Two previous distinct kings"**: I treat them as distinct by hotkey. If one hotkey is crowned
   twice in a row, second place goes to the previous *other* hotkey, so 95% does not go to a single miner.
   If it should be different, the change is in one place (`payout_slots`).
3. **Strata without H** are excluded from AUROC: a warning is written to the log, and the list is in `excluded_strata`.
   Strata that have H but no outputs simply get weight 0. Gates (95%) are counted over all outputs,
   including the excluded strata.
4. **`pyproject.toml`:** added the `slow` marker; it is excluded by default
   (`addopts = -m 'not live and not slow'`). The long A/A is run via `pytest -m slow`.
5. **The bootstrap is percentile-based**, as recorded in D-23 (no BCa and no studentization). The default seed is 0.
   In production the seed must be derived from the period seed (D-57); this is not wired up yet.
6. **From the phase 5 plan line, "baselines" and "simulations on the It's AI proxy" were not done:** they
   need the detector gateway (phase 1). For now there is only synthetic data. The two-stage crown (D-63)
   and freezing on a detector version change (D-32) are also not in this phase.

## Live example (input → output)
**A single decision** (synthetic period, n = 750, m = 1000, 24 strata, B = 2000, φ = 0.5):
```
--- A/A (equal models)
king       AUROC {'det_a': 0.7914, 'det_b': 0.7507, 'det_c': 0.8513} S = 0.3608
challenger AUROC {'det_a': 0.8033, 'det_b': 0.7443, 'det_c': 0.8456} S = 0.3665
crown = False
   FAIL: q_0.05(dS*) = -0.0211 <= delta_min = +0.0000 (dS = +0.0057, se = 0.0169, B = 2000)
   PASS: challenger gate pass rate 0.985 >= 0.950
--- challenger better by true dS = 0.05
challenger AUROC {'det_a': 0.7762, 'det_b': 0.7202, 'det_c': 0.8199} S = 0.4221
crown = True
   PASS: q_0.05(dS*) = +0.0335 > delta_min = +0.0000 (dS = +0.0613, se = 0.0173, B = 2000)
   PASS: challenger gate pass rate 0.985 >= 0.950
```
**Payouts over the course of the history** (baseline → A → B → A again → C):
```
baseline king: {'__burn__': 1.0}
after A:       {'5Fminer-A': 1.0}
after B:       {'5Fminer-B': 0.9, '5Fminer-A': 0.1}
after A again: {'5Fminer-A': 0.9, '5Fminer-B': 0.1}
after C:       {'5Fminer-C': 0.85, '5Fminer-A': 0.1, '5Fminer-B': 0.05}
```

**Simulation** (`python scripts/simulate_crown.py --decompose`, seed 20261108). Parameters:
n = 750, m = 1000, 24 strata (Q-2 proportions × 3 length ranges), 2% gate failures, B = 2000,
α = 0.05, δ_min = 0. Synthetic detectors: king AUROC 0.80 / 0.75 / 0.85, rounding to
4 / 2 / 3 digits. Population S of the king = 0.353.

A/A (the challenger is distributionally the same as the king):

| φ | periods | false crowns | rate | MC se | 95% CI (Wilson) | sd(ΔŜ) | mean bootstrap se |
|---|---|---|---|---|---|---|---|
| 0.0 | 2000 | 95 | **0.0475** | 0.0048 | [0.0390, 0.0577] | 0.0207 | 0.0210 |
| 0.5 | 2000 | 81 | **0.0405** | 0.0044 | [0.0327, 0.0501] | 0.0163 | 0.0169 |
| 0.8 | 2000 | 81 | **0.0405** | 0.0044 | [0.0327, 0.0501] | 0.0126 | 0.0133 |

Power: P(crown) as a function of the true ΔS, 300 periods per cell. Row 0 is the A/A
above. The "ΔS check" column is ΔS on a sample of 2·10⁵ texts with ties from rounding; it
confirms the analytic calibration.

| true ΔS | ΔS check | φ = 0 | φ = 0.5 | φ = 0.8 |
|---|---|---|---|---|
| 0 | 0 | 0.048 | 0.041 | 0.041 |
| 0.01 | 0.0101 | 0.133 | 0.153 | 0.157 |
| 0.02 | 0.0201 | 0.250 | 0.333 | 0.463 |
| 0.03 | 0.0300 | 0.437 | 0.563 | 0.763 |
| 0.04 | 0.0400 | 0.640 | 0.783 | 0.933 |
| 0.05 | 0.0499 | 0.780 | 0.923 | 0.997 |
| 0.06 | 0.0599 | 0.883 | 0.977 | 1.000 |
| 0.08 | 0.0798 | 0.987 | 1.000 | 1.000 |

80% power is reached at ΔS ≈ 0.05 (φ = 0), ≈ 0.04 (φ = 0.5) and ≈ 0.033 (φ = 0.8). With a king S
of ≈ 0.35 this is an improvement of 9–14% relative to the king.

Where the bootstrap se comes from (A/A, 1000 periods per φ):

| φ | true sd(ΔŜ) | se: inputs + H (D-23) | se: inputs only | se: H only |
|---|---|---|---|---|
| 0 | 0.0206 | 0.0210 (+2%) | 0.0205 | 0.0034 |
| 0.5 | 0.0162 | 0.0169 (+4%) | 0.0163 | 0.0031 |
| 0.8 | 0.0126 | 0.0133 (+6%) | 0.0127 | 0.0028 |

Resampling only the inputs gives an se equal to the true spread of ΔŜ. Resampling H (D-23) adds
a few percent: under A/A the contribution of H to the true variance is zero, but the bootstrap "sees" second-order
noise, which is more noticeable with 24 small strata. The test is therefore slightly conservative:
4.0–4.75% false crowns instead of 5%.

## Test results (numbers against acceptance criteria)
- `pytest`: **90 passed, 1 deselected** (82 new in `tests/scoring/` + 8 old), 17–21 s.
  The criterion was < 60 s.
- `pytest -m slow`: **1 passed** in 33 s. This is A/A at n = 750, B = 2000, 1000 periods: rate ≤ 0.05 + 2·MC se.
- `ruff check .` and `ruff format --check .` — clean.
- AUROC matches `sklearn.metrics.roc_auc_score` to 1e-12, including on data with ties.
  Fast bootstrap replicates match materialized resampling to 1e-12.
- A/A at n = 750: **0.0475 / 0.0405 / 0.0405** (φ = 0 / 0.5 / 0.8) against a target of ≤ 0.05 + MC error. ✅
- Coverage of the lower bound under the alternative (test, n = 240, ΔS = 0.03): the bound is above the true ΔS
  no more often than the expected ~5%. ✅
- Speed: ≈ 0.1 s per decision. A full script run takes 110 s on 16 processes, with
  `--decompose` 148 s. `--quick` takes about 5 s. Two full runs with the same seed produced
  byte-identical A/A and power tables: the result is deterministic and does not depend on `--jobs`.

## How to check it yourself (2–3 commands)
```bash
export UV_CACHE_DIR=/workspace/sn32_owner/.uv-cache
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest && .venv/bin/pytest -m slow                         # tests (regular + long A/A)
.venv/bin/python scripts/simulate_crown.py --quick                   # smoke test in seconds
.venv/bin/python scripts/simulate_crown.py --decompose --csv var/sim/crown.csv   # tables above
```

## What you should review (checklist)
- [ ] "Previous distinct kings" = distinct **hotkeys** (D-12). Do you agree?
- [ ] Power at n = 750: a crown with 80% probability requires ΔS ≈ 0.04–0.05 (+9–14% over the king's S).
  Is that enough, or are more texts needed? The se width scales as 1/√n.
- [ ] δ_min = 0 (D-25): the chance of a false crown is ≈ 4–5% **per** challenger equal to the king.
  With K such challengers per period, ≈ K·α false crowns are expected; ~$200 per submission (D-04)
  turns into roughly $4000 per "won lottery".
- [ ] Strata without H are excluded with a warning. Would it be better to fail with an error? The phase 3 builder must
  guarantee that H covers all strata (D-54).
- [ ] Is the synthetic data plausible? King AUROC 0.75–0.85, detector noise 0.5–0.9 σ, 2% gate
  failures. φ here is the correlation of *latent* outcomes. The observed score correlation is lower because of detector
  noise, so even φ = 0.8 reduces se by only ~40%.
- [ ] Bootstrap seed in production: I propose `SeedSequence` from `H(seed_t ‖ challenger_id)` (D-57).

## Risks, open questions, what is needed next
- **Conservativeness due to resampling H** (see the decomposition above): slightly fewer false crowns than α,
  slightly lower power. This is the safe side, so it is left as in D-23. The alternative —
  resampling only the inputs — is exact under A/A but does not account for H noise under the alternative.
- **Small strata.** "Fiction" (5%) gives ~12 outputs per length range. At m_s ≤ 2 the bootstrap
  underestimates the variance. A guarantee of a minimum stratum size in H from the builder (phase 3) is needed.
- **Ties in real detectors** (saturation at 0/1, rounding to whole percent) are handled correctly,
  but reduce discriminating power. To be checked on recorded responses after phase 1.
- **Next:** calibrate σ and φ on the It's AI proxy and real responses (phase 1, D-24, D-33);
  tie the bootstrap seed to the period seed; define the format of the "period package" (scores + gates + strata +
  `input_ids`) that the scorer receives from phases 1/3/4; choose δ_min after the first periods (D-25).

## Change 2026-11-09: probability term in `E_d` (D-20)
`E_d = (1 − w)·2·min(0.5, 1 − AUROC_d) + w·P_d`, `w = 0.5` (`metric.DEFAULT_PROB_WEIGHT`), where `P_d` is the mean
"human" probability (1 − score) over outputs from strata that have H (the same outputs as in AUROC); an output that
failed a gate contributes 0. The bootstrap resamples both terms together (`ResamplingPlan.scores`); the fast path
is checked against naive recomputation. The power simulation (`simulate.py`) is still at `w = 0`: its analytic helpers
model only AUROC.

On pilot 1, It's AI only (free), 78 outputs:

| Submission | AUROC | P(human) | S at w = 0 | S at w = 0.5 |
|---|---|---|---|---|
| identity (AI input) | 1.000 | 0.017 | 0.000 | 0.009 |
| Phrasly without gates | 0.960 | 0.562 | 0.080 | 0.321 |
| Phrasly, rejection by G1 only | 0.997 | 0.082 | 0.005 | 0.044 |
| Phrasly, all gates (current) | 1.000 | 0.000 | 0.000 | 0.000 |

## Change 2026-11-09: per-output quality levels (D-22)
Gates give each output a level of 1 / 0.5 / 0 (`GateResult.level`; for G2/G3, a flag by agreement of two judges or by
the arbiter's decision in a dispute). An output's contribution to both terms of `E_d` is multiplied by its level; level 0 is equivalent to
the former "+inf" (verified by a test), and 0.5 gives exactly half. A penalty `λ · share of major` exists, default 0.
The "≥ 95% passed gates" requirement for the crown is disabled (`min_gate_pass = 0`, kept as an option). The fast bootstrap
with levels and the penalty is checked against naive recomputation.

Pilot 1, It's AI, new gate rules (arbiter — Claude Sonnet 5.5, extra cost $0.66):

| Submission | AUROC | P(human) | major / minor | S |
|---|---|---|---|---|
| identity (AI input) | 1.000 | 0.017 | 0 / 0 | 0.009 |
| strong-LLM paraphrase (24 inputs, G1 passed) | 1.000 | 0.040 | 0 / 0 | 0.020 |
| Phrasly, all gates with levels | 0.999 | 0.081 | 71% / 29% | **0.042** |
| Phrasly, G1 only with levels | 0.995 | 0.149 | 56% / 21% | 0.079 |
| Phrasly without gates | 0.960 | 0.562 | — | 0.321 |

Phrasly levels by gate: G1 — 18 passed, 16 minor, 44 major; G2 — 34 / 44 / 0 (register 32, person/tense 22,
format 7); G3 — 2 / 38 / 38 (cut-off phrases 30, incoherence 21). G2/G3 are not calibrated yet.

## Change 2026-11-10: power under the D-20/D-22 metric
The simulation (`scoring/simulate.py`) now models both terms of `E_d`: the analytic `P_d = m_q · E[1 − σ(gain·x − bias)]`
(Gauss–Hermite, `x ~ N(μ, 1 + stratum_sd² + σ_d²)`) and quality levels 0 / 0.5 / 1 (`gate_fail_rate`, `half_rate`;
mean level `m_q`). The analytic S matches the mean over periods (test `test_expected_score_matches_simulation`).
Along the way a rounding bug was found and fixed: when in all strata the outputs are above all humans, the sum of stratum weights
gave AUROC 1.0000000000000002 and `evasion()` crashed (the value is now clamped to [0, 1]; test with weights 1/13, 6/13, 3/13, 3/13).

`scripts/simulate_crown.py --mde …`: the smallest true ΔS at which the challenger is crowned with 80% probability
(linear interpolation over the ΔS grid), α = 0.05, B = 1000, 300 periods per cell, H = n. Quality levels as for
the reference humanizer v1 (0: 12%, 0.5: 23%), `w = 0.5`. In parentheses — ΔS relative to the king's S. False crowns
(A/A) in all runs are 0.033–0.067 with an MC error of ±0.013, i.e. ≈ α.

| n (= H) | Phrasly-level king, S = 0.017, φ = 0.5 | average, S = 0.069, φ = 0.5 | mature, S = 0.26, φ = 0.5 |
|---|---|---|---|
| 200 | 0.014 (86%) | 0.034 (50%) | 0.069 (26%) |
| 300 | 0.012 (69%) | 0.028 (40%) | 0.056 (21%) |
| 500 | 0.008 (47%) | 0.020 (29%) | 0.043 (17%) |
| 750 | 0.007 (40%) | 0.015 (22%) | 0.032 (12%) |
| 1000 | 0.005 (32%) | 0.014 (20%) | 0.028 (11%) |

Full tables (φ = 0 / 0.5 / 0.8): king AUROC 0.99/0.98/0.995 — φ=0: 0.016, 0.013, 0.009, 0.007, 0.006;
φ=0.8: 0.012, 0.010, 0.007, 0.006, 0.005. Average (0.95/0.93/0.97) — φ=0: 0.042, 0.031, 0.023, 0.018, 0.016;
φ=0.8: 0.028, 0.025, 0.018, 0.013, 0.012. Mature (0.80/0.75/0.85) — φ=0: > 0.08, 0.064, 0.048, 0.038, 0.034;
φ=0.8: 0.059, 0.048, 0.038, 0.028, 0.025 (for n = 200, 300, 500, 750, 1000).

- The detectable ΔS falls as 1/√n: going from 750 → 500 texts worsens sensitivity by ~25%, 750 → 300 by ~70%.
- Submission cost is linear in n (detectors + judges + GPU): at 750 ≈ $170, at 500 ≈ $115, at 300 ≈ $70.
- Caveat: the detectors are synthetic (a sigmoid of a latent scale); real ones have extreme score distributions (GPTZero
  ≥ 0.99 on most AI texts), so the absolute numbers are a guide, not a promise.
- False crowns at n = H = 750, φ = 0.5, 3000 periods: **5.5% ± 0.4%** at `w = 0.5` (5.2% at `w = 0`);
  the bootstrap se matches the true spread of ΔŜ (0.0129 vs 0.0130). The default H size in the simulation
  is now equal to n (that is how a period is assembled, D-54), not 1000.
