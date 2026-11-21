# 05. Scorer and crown statistics
Status: draft, implemented autonomously (without separate approval of the note)
Implements decisions: D-20, D-21, D-22, D-23, D-24, D-25, D-11, D-12, D-13

## Goal
A pure (no network, no paid APIs) library that, from detector responses already obtained, decides:
what score `S` a submission has, whether a challenger takes the crown from the king, and how to split payouts 85/10/5.
Plus a simulator on synthetic data, to check that false crowns ≈ α and to estimate power.

## Inputs and outputs
- **Input per detector d:** "this is AI" probabilities (higher = more AI) for the `n` outputs of the submission and for the `m`
  texts of the human reference set H; the group (stratum) label of each text `"genre|length"` (a string); a
  `gate_pass` mask for the outputs. An output that fails any gate gets the score `+inf` for all detectors
  (D-22) — it is "caught" by all of them.
- **AUROC_d** = P(score(output) > score(human)), ties = 0.5; computed **within each stratum**
  (Mann–Whitney via sorting, O((n+m) log(n+m))) and averaged with weights `n_s` (the number of outputs
  in the stratum). Strata without human texts are excluded and listed in the result (D-20).
- `E_d = 2·min(0.5, 1 − AUROC_d)`; `S = (mean_d E_d^p)^(1/p)`, `p = −4`; any `E_d = 0` ⇒ `S = 0` (D-21).
- **Crown test (D-23):** paired bootstrap `ΔS = S_ch − S_king`. In each replicate the input indices
  are resampled with replacement **within strata and jointly** for the king and the challenger; H is also
  resampled within strata (the same H sample for both). B = 2000, α = 0.05.
  Crown ⇔ `q_α(ΔS*) > δ_min` **and** the challenger's gate pass rate ≥ 0.95 (D-22). δ_min = 0
  by default (D-25, Q-4). For the king the gate pass rate is only reported (gates are absolute).
- **Payouts (D-12, D-13):** coronation history; 85/10/5 — the current king and the two previous
  **distinct** (by hotkey) kings. If there are fewer than three kings, the remainder goes to burn (`"__burn__"`);
  the share of the owner's starting baseline also goes to burn.

## Interfaces
Package `humanizer_eval.scoring`:
- `metric.py`: `apply_gate`, `auroc`, `stratified_auroc` / `stratified_auroc_detail`, `evasion`,
  `softmin`; containers `SubmissionRun` (per-detector scores, `gate_pass`, strata, optional `input_ids`),
  `HumanReference` (per-detector H scores, strata); `submission_score(run, human) -> SubmissionScore`.
- `bootstrap.py`: `paired_bootstrap(king, challenger, human, n_boot=2000, alpha=0.05, seed=0)`
  → `PairedBootstrapResult` (ΔS, lower bound, SE, S of both, optional array of replicates). Vectorized:
  resampling = count weights, replicate AUROC via cumulative sums over the sorted H —
  O(B·(n+m)) per detector, no O(n·m). Deterministic for a given seed (in production — the period seed, D-57).
- `crown.py`: `crown_decision(king, challenger, human, ...) -> CrownDecision` (decision, reasons,
  ΔS, bound, gate pass rates); `KingState` (history, `apply_crown`, `payout_weights`, JSON).
- `simulate.py` + `scripts/simulate_crown.py`: synthetic periods (shared text "difficulty",
  model skill, correlation φ between model outcomes on the same text, detector noise, rounding of
  scores → ties, stratum effects), an A/A experiment and power curves.

## What we do not do (boundaries)
We do not call detectors or judges (that is phases 1 and 4), do not build H (phase 3), do not write to the chain and do not
set weights (phases 8–9). No two-stage crown (D-63), no freeze on a detector version change
(D-32) and no baseline humanizers — only the interfaces they will plug into. Simulations on the It's AI
proxy (D-33) come after phase 1; for now synthetic data only.

## Tests and acceptance criteria
- AUROC matches `sklearn.metrics.roc_auc_score` (random data, ties, `+inf`); hand-written
  examples with ties; stratum weights; exclusion of strata without H; gate mask; soft-min edge cases.
- Fast bootstrap replicates match "naive" resampling (materialized samples).
- Determinism with a single seed; king = challenger (identical outputs) ⇒ crown impossible.
- Small A/A in the regular tests; long A/A under the `slow` marker. A regular `pytest` run < 60 s.
- Crown logic (both reasons), 85/10/5 with 1, 2, 3+ kings, repeated hotkey, burn, JSON round-trip.
- Script: A/A at n = 750 gives a false-crown rate ≤ 0.05 + Monte Carlo error.

## Risks and open questions
- The percentile bootstrap is calibrated asymptotically; with small strata (m_s ≤ 2) the variance
  is underestimated — H under D-54 has the same proportions as the slots, so strata should not be empty.
- α controls the false crown **per challenger**; with K challengers against an equal king,
  ≈ K·α false crowns are expected. Protection: the submission cost (D-04) and δ_min (D-25).
- "Distinct" previous kings — I interpret this as distinct hotkeys (otherwise a single miner who improved on themselves would
  take 95%). Needs the owner's confirmation.
- The `crown_decision` signature was extended with a required `human` (human reference set H) and bootstrap parameters.
