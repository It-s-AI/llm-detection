"""Per-submission metric: stratified AUROC + mean P(human) per detector, soft-min (spec D-20..D-22).

Conventions
-----------
* A detector score is the detector's probability that the text is AI, in [0, 1] (the primary score
  of each vendor client). The AUROC term only uses the order of scores; the probability term uses the
  values, so outputs and the human reference set H must be on the same [0, 1] scale.
* ``AUROC_d = P(score(output) > score(human)) + 0.5 * P(score(output) == score(human))``. It is computed
  inside each stratum (genre x length bin) and averaged with weights proportional to the number of
  submission outputs in the stratum. Strata that have outputs but no human texts are excluded and
  reported. An AUROC of 0.5 means the detector cannot tell the outputs from human text.
* Every output has a quality level ``q`` in {1, 0.5, 0} from the gates (D-22): 1 passed, 0.5 minor
  violation, 0 severe. Its contribution to both terms below is multiplied by ``q``: a severe failure
  counts as caught by every detector (as if its score were ``+inf``), a minor one earns half.
  Booleans are accepted (True = 1, False = 0).
* ``E_d = (1 - w) * 2 * min(0.5, 1 - AUROC_d) + w * P_d`` with ``w = 0.5`` (D-20), where
  ``AUROC_d = 1 - mean_i q_i * u_i`` (``u_i``: share of same-stratum humans scored above output ``i``,
  ties 0.5, averaged over strata with weights by outputs) and ``P_d = mean_i q_i * (1 - score_i)``
  is the quality-weighted mean predicted probability of the human class. A miner gains from
  lowering the detector's probabilities even while the outputs still rank above every human text
  (no AUROC change), and the AUROC term rewards outputs that become truly indistinguishable.
  ``S = softmin_p(E_1..E_D) - penalty * share of severe outputs`` with ``p = -4`` (D-21) and
  ``penalty = 0`` by default.

Every AUROC here is an exact Mann-Whitney U statistic computed by sorting the human scores and
binary-searching the output scores: O((n + m) log m) per stratum, never O(n * m).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)

#: Score assigned to an output that failed a quality gate: maximally AI for every detector.
GATE_FAIL_SCORE = math.inf

#: Soft-min exponent from D-21.
DEFAULT_P = -4.0

#: Weight of the mean-probability term in ``E_d`` (D-20, owner decision 2026-11-09).
DEFAULT_PROB_WEIGHT = 0.5

#: ``S`` is reduced by ``penalty * share of severe outputs``; 0 = a broken output is worth nothing
#: (like an unchanged one) but costs nothing extra (owner decision 2026-11-09).
DEFAULT_SEVERE_PENALTY = 0.0

QUALITY_LEVELS = (0.0, 0.5, 1.0)


# --- input validation -------------------------------------------------------------------------
def _as_scores(x: ArrayLike, name: str) -> np.ndarray:
    a = np.array(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if np.isnan(a).any():
        raise ValueError(f"{name} contains NaN (a missing detector response must be retried, not scored)")
    if np.isneginf(a).any():
        raise ValueError(f"{name} contains -inf")
    return a


def _as_labels(x: ArrayLike, name: str, n: int) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f"{name} must be one-dimensional with {n} labels, got shape {a.shape}")
    return a.astype(str)


def _as_mask(x: ArrayLike, name: str, n: int) -> np.ndarray:
    a = np.array(x)
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f"{name} must be one-dimensional with {n} entries, got shape {a.shape}")
    if a.dtype != np.bool_:
        if not np.isin(a, (0, 1)).all():
            raise ValueError(f"{name} must be boolean")
        a = a.astype(bool)
    return a


def group_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    """Map each distinct label to the (sorted) positions where it occurs."""
    uniq, inv = np.unique(labels, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    return {str(u): order[bounds[k] : bounds[k + 1]] for k, u in enumerate(uniq)}


def _as_quality(x: ArrayLike, name: str, n: int) -> np.ndarray:
    a = np.array(x)
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f"{name} must be one-dimensional with {n} entries, got shape {a.shape}")
    a = a.astype(np.float64)
    if not np.isin(a, QUALITY_LEVELS).all():
        raise ValueError(f"{name} must hold quality levels 0, 0.5, 1 (or booleans)")
    return a


# --- gates ------------------------------------------------------------------------------------
def apply_gate(scores: ArrayLike, gate_pass: ArrayLike) -> np.ndarray:
    """Return a copy of ``scores`` with every gate-failed output set to ``+inf`` (D-22)."""
    s = _as_scores(scores, "scores")
    g = _as_mask(gate_pass, "gate_pass", s.shape[0])
    out = s.copy()
    out[~g] = GATE_FAIL_SCORE
    return out


# --- AUROC ------------------------------------------------------------------------------------
def _mw_numerator(ai_scores: np.ndarray, human_sorted: np.ndarray) -> float:
    """Sum over outputs of (#humans strictly below + 0.5 * #humans tied).

    Divided by ``n * m`` this is the Mann-Whitney AUROC with ties counted 0.5.
    """
    lo = np.searchsorted(human_sorted, ai_scores, side="left")
    hi = np.searchsorted(human_sorted, ai_scores, side="right")
    return 0.5 * float(np.sum(lo, dtype=np.int64) + np.sum(hi, dtype=np.int64))


def auroc(ai_scores: ArrayLike, human_scores: ArrayLike) -> float:
    """Unstratified AUROC = P(ai > human) + 0.5 * P(ai == human)."""
    x = _as_scores(ai_scores, "ai_scores")
    y = _as_scores(human_scores, "human_scores")
    if x.size == 0 or y.size == 0:
        raise ValueError("AUROC needs at least one output and one human score")
    return _mw_numerator(x, np.sort(y)) / (x.size * y.size)


@dataclass(frozen=True)
class StratifiedAUROC:
    value: float
    per_stratum: dict[str, float]  # AUROC inside each included stratum
    weights: dict[str, float]  # n_s / N over included strata (sums to 1)
    excluded: tuple[str, ...]  # strata with outputs but no human texts


def stratified_auroc_detail(
    ai_scores: ArrayLike,
    ai_strata: ArrayLike,
    human_scores: ArrayLike,
    human_strata: ArrayLike,
    quality: ArrayLike | None = None,
) -> StratifiedAUROC:
    """Stratified AUROC with per-stratum values, weights and excluded strata.

    With ``quality``, an output's "human-likeness" ``u_i`` counts ``q_i`` times: the stratum value is
    ``1 - mean_i q_i * u_i`` (``q = 1`` everywhere gives the plain Mann-Whitney AUROC; ``q = 0`` is the
    same as scoring the output ``+inf``).
    """
    x = _as_scores(ai_scores, "ai_scores")
    y = _as_scores(human_scores, "human_scores")
    xs = _as_labels(ai_strata, "ai_strata", x.shape[0])
    ys = _as_labels(human_strata, "human_strata", y.shape[0])
    q = np.ones(x.shape[0]) if quality is None else _as_quality(quality, "quality", x.shape[0])
    if x.size == 0:
        raise ValueError("no submission outputs")
    out_groups = group_indices(xs)
    hum_groups = group_indices(ys)
    included = [s for s in out_groups if s in hum_groups]
    excluded = tuple(s for s in out_groups if s not in hum_groups)
    if not included:
        raise ValueError("no stratum has both submission outputs and human texts")
    n_total = sum(out_groups[s].size for s in included)
    per: dict[str, float] = {}
    weights: dict[str, float] = {}
    value = 0.0
    for s in included:
        xi = x[out_groups[s]]
        qi = q[out_groups[s]]
        yi = np.sort(y[hum_groups[s]])
        below = 0.5 * (np.searchsorted(yi, xi, side="left") + np.searchsorted(yi, xi, side="right"))
        a = 1.0 - float(np.sum(qi * (1.0 - below / yi.size))) / xi.size
        w = xi.size / n_total
        per[s] = a
        weights[s] = w
        value += w * a
    # the weights sum to 1 only up to round-off: all-perfect strata can give 1.0000000000000002
    value = min(1.0, max(0.0, value))
    return StratifiedAUROC(value=value, per_stratum=per, weights=weights, excluded=excluded)


def stratified_auroc(
    ai_scores: ArrayLike,
    ai_strata: ArrayLike,
    human_scores: ArrayLike,
    human_strata: ArrayLike,
) -> float:
    """AUROC inside each stratum, averaged with weights proportional to outputs per stratum (D-20).

    Use :func:`stratified_auroc_detail` to see per-stratum values and the excluded strata.
    """
    r = stratified_auroc_detail(ai_scores, ai_strata, human_scores, human_strata)
    _warn_excluded(r.excluded)
    return r.value


def _warn_excluded(excluded: tuple[str, ...]) -> None:
    if excluded:
        log.warning("strata without human texts excluded from AUROC: %s", ", ".join(excluded))


# --- evasion and soft-min ---------------------------------------------------------------------
def evasion(auroc_value: float | ArrayLike) -> float | np.ndarray:
    """AUROC term ``2 * min(0.5, 1 - AUROC_d)`` in [0, 1]; 1 = indistinguishable from human."""
    a = np.asarray(auroc_value, dtype=np.float64)
    if np.isnan(a).any() or (a < 0).any() or (a > 1).any():
        raise ValueError("AUROC must be in [0, 1]")
    e = 2.0 * np.minimum(0.5, 1.0 - a)
    return float(e) if e.ndim == 0 else e


def prob_human(gated_scores: ArrayLike, name: str = "scores") -> np.ndarray:
    """Per-output ``1 - score``; ``+inf`` (caught / failed request) gives 0. Scores must be in [0, 1]."""
    s = np.asarray(gated_scores, dtype=np.float64)
    finite = np.isfinite(s)
    if ((s[finite] < 0) | (s[finite] > 1)).any():
        raise ValueError(f"{name}: the probability term needs detector scores in [0, 1]")
    q = np.zeros_like(s)
    q[finite] = 1.0 - s[finite]
    return q


def detector_evasion(
    auroc_value: float | ArrayLike, p_human: float | ArrayLike, w_prob: float = DEFAULT_PROB_WEIGHT
) -> float | np.ndarray:
    """``E_d = (1 - w) * evasion(AUROC_d) + w * P_d`` in [0, 1] (D-20)."""
    if not 0 <= w_prob <= 1:
        raise ValueError("w_prob must be in [0, 1]")
    ph = np.asarray(p_human, dtype=np.float64)
    if np.isnan(ph).any() or (ph < 0).any() or (ph > 1).any():
        raise ValueError("P(human) must be in [0, 1]")
    e = (1.0 - w_prob) * np.asarray(evasion(auroc_value)) + w_prob * ph
    return float(e) if e.ndim == 0 else e


def softmin(values: ArrayLike, p: float = DEFAULT_P, axis: int | None = None) -> float | np.ndarray:
    """Power mean ``(mean v^p)^(1/p)`` with ``p < 0``; exactly 0 if any value is 0 (D-21).

    With ``axis=None`` all values are pooled and a float is returned; otherwise the mean is taken
    along ``axis`` (used by the bootstrap to combine detectors for every replicate at once).
    """
    if not p < 0:
        raise ValueError("soft-min needs a negative exponent p")
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("soft-min of no values")
    if np.isnan(v).any() or (v < 0).any() or np.isinf(v).any():
        raise ValueError("soft-min values must be finite and non-negative")
    any_zero = (v == 0).any(axis=axis)
    with np.errstate(divide="ignore"):
        powered = np.where(v > 0, v, 1.0) ** p
    m = powered.mean(axis=axis)
    out = np.where(any_zero, 0.0, m ** (1.0 / p))
    return float(out) if out.ndim == 0 else out


# --- submission-level containers --------------------------------------------------------------
def _freeze(*arrays: np.ndarray | None) -> None:
    for a in arrays:
        if a is not None:
            a.setflags(write=False)


def _scores_by_detector(scores: Mapping[str, ArrayLike], what: str) -> dict[str, np.ndarray]:
    if not scores:
        raise ValueError(f"{what}: no detectors")
    out = {str(d): _as_scores(v, f"{what}[{d!r}]") for d, v in scores.items()}
    sizes = {v.shape[0] for v in out.values()}
    if len(sizes) != 1:
        raise ValueError(f"{what}: detectors disagree on the number of texts: {sorted(sizes)}")
    return out


class SubmissionRun:
    """One submission's raw detector scores on the period inputs (index-aligned across detectors).

    ``scores[d][i]`` is detector ``d``'s AI probability for the output on input ``i`` *before* gating;
    ``gate_pass[i]`` is that output's quality level (1, 0.5, 0; booleans allowed); ``strata[i]`` is the
    stratum of input ``i``. ``input_ids`` (optional) lets the paired test verify that king and
    challenger arrays are aligned on the same inputs. Arrays are copied and validated once.
    """

    def __init__(
        self,
        scores: Mapping[str, ArrayLike],
        gate_pass: ArrayLike,
        strata: ArrayLike,
        input_ids: ArrayLike | None = None,
    ):
        self._scores = _scores_by_detector(scores, "SubmissionRun.scores")
        n = next(iter(self._scores.values())).shape[0]
        if n == 0:
            raise ValueError("SubmissionRun has no outputs")
        self._quality = _as_quality(gate_pass, "gate_pass", n)
        self._gate = self._quality == 1.0
        self._strata = _as_labels(strata, "strata", n)
        self._ids: np.ndarray | None = None
        if input_ids is not None:
            self._ids = _as_labels(input_ids, "input_ids", n)
            if np.unique(self._ids).size != n:
                raise ValueError("input_ids must be unique")
        _freeze(*self._scores.values(), self._quality, self._gate, self._strata, self._ids)

    def __repr__(self) -> str:
        return (
            f"SubmissionRun(n={self.n}, detectors={self.detectors}, gate_pass_rate={self.gate_pass_rate:.3f})"
        )

    @property
    def n(self) -> int:
        return self._gate.shape[0]

    @property
    def detectors(self) -> tuple[str, ...]:
        return tuple(sorted(self._scores))

    @property
    def gate_pass(self) -> np.ndarray:
        """True where the output passed every gate (quality level 1)."""
        return self._gate

    @property
    def quality(self) -> np.ndarray:
        return self._quality

    @property
    def severe_rate(self) -> float:
        return float((self._quality == 0.0).mean())

    @property
    def strata(self) -> np.ndarray:
        return self._strata

    @property
    def input_ids(self) -> np.ndarray | None:
        return self._ids

    @property
    def gate_pass_rate(self) -> float:
        return float(self._gate.mean())

    def raw_scores(self, detector: str) -> np.ndarray:
        return self._scores[detector]

    def gated_scores(self, detector: str) -> np.ndarray:
        """Detector scores with severe (level 0) outputs set to ``+inf`` (D-22)."""
        out = self._scores[detector].copy()
        out[self._quality == 0.0] = GATE_FAIL_SCORE
        return out


class HumanReference:
    """Detector scores of the period's human reference set H (D-54), index-aligned across detectors."""

    def __init__(self, scores: Mapping[str, ArrayLike], strata: ArrayLike):
        self._scores = _scores_by_detector(scores, "HumanReference.scores")
        m = next(iter(self._scores.values())).shape[0]
        if m == 0:
            raise ValueError("HumanReference is empty")
        if any(np.isinf(v).any() for v in self._scores.values()):
            raise ValueError("human scores must be finite")
        self._strata = _as_labels(strata, "strata", m)
        _freeze(*self._scores.values(), self._strata)

    def __repr__(self) -> str:
        return f"HumanReference(m={self.m}, detectors={self.detectors})"

    @property
    def m(self) -> int:
        return self._strata.shape[0]

    @property
    def detectors(self) -> tuple[str, ...]:
        return tuple(sorted(self._scores))

    @property
    def strata(self) -> np.ndarray:
        return self._strata

    def scores(self, detector: str) -> np.ndarray:
        return self._scores[detector]


def check_detectors(runs: Iterable[SubmissionRun], human: HumanReference) -> tuple[str, ...]:
    """Every run must be scored by exactly the detectors that scored H (D-24)."""
    dets = human.detectors
    for r in runs:
        if r.detectors != dets:
            raise ValueError(f"detector mismatch: run has {r.detectors}, human reference has {dets}")
    return dets


@dataclass(frozen=True)
class SubmissionScore:
    s: float  # soft-min of the per-detector E_d, the submission score S
    auroc: dict[str, float]
    p_human: dict[str, float]  # mean predicted probability of the human class over the outputs
    evasion: dict[str, float]  # E_d (both terms)
    gate_pass_rate: float
    n: int
    excluded_strata: tuple[str, ...]
    w_prob: float = DEFAULT_PROB_WEIGHT
    severe_rate: float = 0.0  # share of level-0 outputs (strata with H)
    minor_rate: float = 0.0  # share of level-0.5 outputs
    penalty: float = DEFAULT_SEVERE_PENALTY


def submission_score(
    run: SubmissionRun,
    human: HumanReference,
    p: float = DEFAULT_P,
    w_prob: float = DEFAULT_PROB_WEIGHT,
    penalty: float = DEFAULT_SEVERE_PENALTY,
) -> SubmissionScore:
    """Combine all detectors into ``S`` (D-20, D-21), with gate failures caught (D-22).

    Both terms use the outputs of the strata that have human texts (the AUROC needs them), so the
    point estimate and every bootstrap replicate average over the same outputs.
    """
    dets = check_detectors([run], human)
    aucs: dict[str, float] = {}
    phs: dict[str, float] = {}
    excluded: tuple[str, ...] = ()
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    q = run.quality
    keep = np.ones(run.n, bool)
    for d in dets:
        raw = run.raw_scores(d)
        r = stratified_auroc_detail(raw, run.strata, human.scores(d), human.strata, quality=q)
        aucs[d] = r.value
        excluded = r.excluded  # same strata for every detector
        keep = ~np.isin(run.strata, np.array(excluded, dtype=str)) if excluded else keep
        phs[d] = float((q * prob_human(raw, d))[keep].mean()) if w_prob > 0 else 0.0
    _warn_excluded(excluded)
    ev = {d: float(detector_evasion(aucs[d], phs[d], w_prob)) for d in dets}
    severe = float((q[keep] == 0.0).mean())
    s = float(softmin(list(ev.values()), p=p)) - penalty * severe
    return SubmissionScore(
        s=s,
        auroc=aucs,
        p_human=phs,
        evasion=ev,
        gate_pass_rate=run.gate_pass_rate,
        n=run.n,
        excluded_strata=excluded,
        w_prob=w_prob,
        severe_rate=severe,
        minor_rate=float((q[keep] == 0.5).mean()),
        penalty=penalty,
    )
