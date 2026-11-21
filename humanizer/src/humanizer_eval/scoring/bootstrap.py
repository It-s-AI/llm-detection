"""Paired bootstrap of ``dS = S_challenger - S_king`` (spec D-23).

King and challenger are evaluated on the *same* period inputs (index-aligned arrays) against the
*same* human reference set H. Every replicate

* resamples the input indices with replacement **within each stratum**, jointly for king and
  challenger (the pair of outputs on one input stays together), and
* resamples H with replacement within each stratum (one H resample shared by both models).

Stratum sizes are preserved, so the stratum weights of D-20 are the same in every replicate.

Quality levels (D-22) weight each output: an output's share of beaten humans counts ``q_i`` times, so a
stratum's AUROC is ``1 - (1/n_s) * sum_i w_i q_i (1 - beaten_i / m_s)``; ``q`` in {0, 1} reproduces
the old "+inf for a failed gate" exactly.

Implementation: a resample is represented by multiplicity counts ``w`` (outputs) and ``v`` (humans).
For one stratum and detector, with H sorted by score and ``C = cumsum(v)`` in that order, output
``i`` beats ``C[lo_i]`` resampled humans and ties ``C[hi_i] - C[lo_i]`` of them, where
``lo_i``/``hi_i`` are the left/right insertion points of its score in sorted H. Those insertion
points do not depend on the resample, so they are computed once; each replicate then costs
O(n_s + m_s) per stratum and detector instead of O(n_s * m_s) or a re-sort.

Replicates are processed in fixed-size blocks, each with its own child of the seed's
``SeedSequence``, so results are deterministic for a given seed and do not depend on memory limits;
the first ``k`` blocks are identical for any ``n_boot >= k * BLOCK``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from humanizer_eval.scoring.metric import (
    DEFAULT_P,
    DEFAULT_PROB_WEIGHT,
    DEFAULT_SEVERE_PENALTY,
    HumanReference,
    SubmissionRun,
    check_detectors,
    detector_evasion,
    evasion,
    group_indices,
    softmin,
    submission_score,
)

DEFAULT_N_BOOT = 2000
DEFAULT_ALPHA = 0.05
BLOCK = 250  # replicates per vectorized block


@dataclass(frozen=True)
class PairedBootstrapResult:
    delta: float  # point estimate S_challenger - S_king on the observed data
    lower: float  # alpha-quantile of the replicate distribution (one-sided lower bound)
    se: float  # standard deviation of the replicates
    alpha: float
    n_boot: int
    seed: int | None
    s_king: float
    s_challenger: float
    excluded_strata: tuple[str, ...]
    replicates: np.ndarray | None = None  # shape (n_boot,), only if requested


def check_paired(king: SubmissionRun, challenger: SubmissionRun) -> None:
    """King and challenger must be scored on the same inputs, in the same order."""
    if king.n != challenger.n:
        raise ValueError(f"paired test needs the same inputs: king n={king.n}, challenger n={challenger.n}")
    if not np.array_equal(king.strata, challenger.strata):
        raise ValueError("king and challenger strata differ: arrays are not aligned on the same inputs")
    ki, ci = king.input_ids, challenger.input_ids
    if ki is not None and ci is not None and not np.array_equal(ki, ci):
        raise ValueError("king and challenger input_ids differ: arrays are not aligned")


class _Stratum:
    __slots__ = ("hi_t", "label", "lo", "lo_t", "m", "n", "order", "out_idx", "q", "qual", "severe", "tie")

    def __init__(self, label: str, out_idx: np.ndarray, m: int):
        self.label = label
        self.out_idx = out_idx
        self.n = out_idx.size
        self.m = m
        self.order: list[np.ndarray] = []  # [detector] -> argsort of H scores in this stratum
        # [detector][run] -> insertion points; ties stored sparsely (most outputs have none)
        self.lo: list[list[np.ndarray]] = []
        self.tie: list[list[np.ndarray]] = []
        self.lo_t: list[list[np.ndarray]] = []
        self.hi_t: list[list[np.ndarray]] = []
        self.q: list[list[np.ndarray]] = []  # [detector][run] -> per-output q_i * (1 - score_i)
        self.qual: list[np.ndarray] = []  # [run] -> quality levels of this stratum's outputs
        self.severe: list[np.ndarray] = []  # [run] -> 1.0 where the output is level 0


class ResamplingPlan:
    """Search positions of every run's outputs inside sorted H, per stratum and detector.

    Shared by all replicates; ``auroc(w, v)`` evaluates the stratified AUROC of every run under
    multiplicity counts ``w`` (outputs) and ``v`` (humans), for a whole block of replicates.
    """

    def __init__(self, runs: Sequence[SubmissionRun], human: HumanReference):
        if not runs:
            raise ValueError("no runs")
        self.detectors = check_detectors(runs, human)
        for r in runs[1:]:
            check_paired(runs[0], r)
        self.n_runs = len(runs)
        out_groups = group_indices(runs[0].strata)
        hum_groups = group_indices(human.strata)
        self.excluded = tuple(s for s in out_groups if s not in hum_groups)
        labels = [s for s in out_groups if s in hum_groups]
        if not labels:
            raise ValueError("no stratum has both submission outputs and human texts")
        gated = [{d: r.raw_scores(d) for d in self.detectors} for r in runs]
        quality = [r.quality for r in runs]
        self._probs_ok = all(
            ((g[d][np.isfinite(g[d])] >= 0) & (g[d][np.isfinite(g[d])] <= 1)).all()
            for g in gated
            for d in self.detectors
        )
        self.strata: list[_Stratum] = []
        for s in labels:
            st = _Stratum(s, out_groups[s], hum_groups[s].size)
            st.qual = [q[st.out_idx] for q in quality]
            st.severe = [(q[st.out_idx] == 0.0).astype(np.float64) for q in quality]
            for d in self.detectors:
                hs = human.scores(d)[hum_groups[s]]
                order = np.argsort(hs, kind="stable")
                hs_sorted = hs[order]
                st.order.append(order)
                los, ties, lo_ts, hi_ts = [], [], [], []
                for g in gated:
                    x = g[d][st.out_idx]
                    lo = np.searchsorted(hs_sorted, x, side="left")
                    hi = np.searchsorted(hs_sorted, x, side="right")
                    t = np.flatnonzero(hi != lo)
                    los.append(lo)
                    ties.append(t)
                    lo_ts.append(lo[t])
                    hi_ts.append(hi[t])
                st.lo.append(los)
                st.tie.append(ties)
                st.lo_t.append(lo_ts)
                st.hi_t.append(hi_ts)
                st.q.append([st.qual[k] * _p_human_unchecked(g[d][st.out_idx]) for k, g in enumerate(gated)])
            self.strata.append(st)
        self.n_total = sum(st.n for st in self.strata)

    def unit_counts(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Counts of the original (non-resampled) data: one replicate with every multiplicity 1."""
        return (
            [np.ones((1, st.n)) for st in self.strata],
            [np.ones((1, st.m)) for st in self.strata],
        )

    def draw_counts(self, rng: np.random.Generator, b: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Multiplicities of ``b`` within-stratum resamples (with replacement) of outputs and of H."""
        ws, vs = [], []
        for st in self.strata:
            ws.append(_counts(rng, st.n, b))
            vs.append(_counts(rng, st.m, b))
        return ws, vs

    def auroc(self, w: Sequence[np.ndarray], v: Sequence[np.ndarray]) -> np.ndarray:
        """Stratified AUROC for every run, detector and replicate: shape ``(n_runs, n_det, b)``.

        ``w[s]`` has shape ``(b, n_s)`` and must sum to ``n_s`` per row; ``v[s]`` has shape
        ``(b, m_s)`` and must sum to ``m_s`` per row (within-stratum resampling).
        """
        b = w[0].shape[0]
        out = np.zeros((self.n_runs, len(self.detectors), b))
        for s, st in enumerate(self.strata):
            ws, vs = w[s], v[s]
            scale = 0.5 / (self.n_total * st.m)
            cum = np.zeros((b, st.m + 1))
            wq = [ws * st.qual[k][None, :] for k in range(self.n_runs)]
            # outputs' weight not covered by quality counts as fully caught (AUROC contribution 1)
            caught = [(st.n - x.sum(axis=1)) / self.n_total for x in wq]
            for d in range(len(self.detectors)):
                np.cumsum(vs[:, st.order[d]], axis=1, out=cum[:, 1:])
                for k in range(self.n_runs):
                    # sum_i w_i (C[lo_i] + C[hi_i]) = 2 sum_i w_i C[lo_i] + sum_ties w_i (C[hi_i] - C[lo_i])
                    num = 2.0 * np.einsum("bi,bi->b", wq[k], cum[:, st.lo[d][k]])
                    t = st.tie[d][k]
                    if t.size:
                        num += np.einsum(
                            "bi,bi->b", wq[k][:, t], cum[:, st.hi_t[d][k]] - cum[:, st.lo_t[d][k]]
                        )
                    out[k, d] += num * scale + caught[k]
        return out

    def p_human(self, w: Sequence[np.ndarray]) -> np.ndarray:
        """Mean P(human) for every run, detector and replicate: shape ``(n_runs, n_det, b)``."""
        if not self._probs_ok:
            raise ValueError("the probability term needs detector scores in [0, 1]")
        b = w[0].shape[0]
        out = np.zeros((self.n_runs, len(self.detectors), b))
        for s, st in enumerate(self.strata):
            for d in range(len(self.detectors)):
                for k in range(self.n_runs):
                    out[k, d] += w[s] @ st.q[d][k]
        return out / self.n_total

    def severe_share(self, w: Sequence[np.ndarray]) -> np.ndarray:
        """Share of level-0 outputs for every run and replicate: shape ``(n_runs, b)``."""
        out = np.zeros((self.n_runs, w[0].shape[0]))
        for s, st in enumerate(self.strata):
            for k in range(self.n_runs):
                out[k] += w[s] @ st.severe[k]
        return out / self.n_total

    def scores(
        self,
        w: Sequence[np.ndarray],
        v: Sequence[np.ndarray],
        p: float = DEFAULT_P,
        w_prob: float = DEFAULT_PROB_WEIGHT,
        penalty: float = DEFAULT_SEVERE_PENALTY,
    ) -> np.ndarray:
        """Submission score ``S`` for every run and replicate: shape ``(n_runs, b)``."""
        auc = np.clip(self.auroc(w, v), 0.0, 1.0)  # round-off just outside [0, 1]
        ph = np.clip(self.p_human(w), 0.0, 1.0) if w_prob > 0 else np.zeros_like(auc)
        s = np.asarray(softmin(detector_evasion(auc, ph, w_prob), p=p, axis=1))
        return s - penalty * self.severe_share(w) if penalty else s


def _p_human_unchecked(x: np.ndarray) -> np.ndarray:
    # range is checked lazily in ``p_human`` so an AUROC-only plan accepts any score scale
    return np.clip(1.0 - np.where(np.isfinite(x), x, 1.0), 0.0, 1.0)


def _counts(rng: np.random.Generator, size: int, b: int) -> np.ndarray:
    idx = rng.integers(0, size, size=(b, size))
    idx += (np.arange(b, dtype=idx.dtype) * size)[:, None]
    return np.bincount(idx.ravel(), minlength=b * size).reshape(b, size).astype(np.float64)


def scores_from_auroc(auc: np.ndarray, p: float = DEFAULT_P) -> np.ndarray:
    """``(n_runs, n_det, b)`` AUROCs -> ``(n_runs, b)`` scores S with the AUROC term only (w_prob = 0)."""
    # Guard against round-off just outside [0, 1] in the vectorized sums.
    return np.asarray(softmin(evasion(np.clip(auc, 0.0, 1.0)), p=p, axis=1))


def paired_bootstrap(
    king: SubmissionRun,
    challenger: SubmissionRun,
    human: HumanReference,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int | np.random.SeedSequence = 0,
    p: float = DEFAULT_P,
    w_prob: float = DEFAULT_PROB_WEIGHT,
    penalty: float = DEFAULT_SEVERE_PENALTY,
    return_replicates: bool = False,
) -> PairedBootstrapResult:
    """Paired, stratified bootstrap of ``dS = S_challenger - S_king`` (D-23).

    ``lower`` is the ``alpha``-quantile of the replicates: the challenger is better than the king
    with one-sided confidence ``1 - alpha`` if ``lower > 0`` (or ``> delta_min``). In production
    ``seed`` is derived from the period seed (D-57) so anyone can reproduce the decision.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    check_paired(king, challenger)
    sk = submission_score(king, human, p=p, w_prob=w_prob, penalty=penalty)
    sc = submission_score(challenger, human, p=p, w_prob=w_prob, penalty=penalty)
    plan = ResamplingPlan([king, challenger], human)

    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    n_blocks = -(-n_boot // BLOCK)
    reps = np.empty(n_boot)
    for blk, child in enumerate(ss.spawn(n_blocks)):
        lo = blk * BLOCK
        b = min(BLOCK, n_boot - lo)
        w, v = plan.draw_counts(np.random.default_rng(child), b)
        s = plan.scores(w, v, p=p, w_prob=w_prob, penalty=penalty)
        reps[lo : lo + b] = s[1] - s[0]

    return PairedBootstrapResult(
        delta=sc.s - sk.s,
        lower=float(np.quantile(reps, alpha)),
        se=float(reps.std(ddof=1)) if n_boot > 1 else 0.0,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed if isinstance(seed, int) else None,
        s_king=sk.s,
        s_challenger=sc.s,
        excluded_strata=plan.excluded,
        replicates=reps if return_replicates else None,
    )
