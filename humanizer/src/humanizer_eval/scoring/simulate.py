"""Synthetic periods for checking the crown test: A/A false-crown rate and power curves.

Generative model (all on a latent "AI-ness" scale; larger = more AI-like):

* Inputs and human texts are laid out over strata (genre x length bin) with fixed counts, as the
  period builder does (D-54). Each stratum has an offset ``c_{s,d}`` per detector that shifts
  outputs *and* human texts alike (some genres look more AI to a detector for everyone) - this is
  what stratified AUROC removes.
* Per input ``i`` a difficulty ``eta_i ~ N(0, 1)`` is shared by king and challenger; model ``k``'s
  outcome on that input is ``a_{k,i} = sqrt(phi) * eta_i + sqrt(1 - phi) * eps_{k,i}``, so
  ``corr(a_king, a_challenger) = phi``.
* Detector ``d`` sees ``x_{k,i,d} = mu_{k,d} + c_{s(i),d} + a_{k,i} + sigma_d * xi`` for outputs and
  ``y_{j,d} = c_{s(j),d} + h_j + sigma_d * xi'`` for human texts, ``h_j ~ N(0, human_sd^2)``.
  ``mu_{k,d}`` is the model's (lack of) skill against detector ``d``. Reported scores are
  ``sigmoid(gain * latent - bias)`` rounded to the detector's precision, which creates ties.
* Each output independently gets quality level 0 with probability ``gate_fail_rate``, 0.5 with
  probability ``half_rate`` and 1 otherwise (D-22).

Within a stratum ``x - y ~ N(mu, 1 + human_sd^2 + 2 sigma_d^2)``, so the population AUROC (ignoring
rounding) is ``1 - m_q * (1 - Phi(mu / sd))`` with mean quality level ``m_q``. The probability term is
``P_d = m_q * E[1 - sigmoid(gain * x - bias)]`` with ``x ~ N(mu, 1 + stratum_sd^2 + sigma_d^2)``
(Gauss-Hermite). This gives an analytic population score ``S`` (D-20 with ``w_prob``) for any skill
vector and lets the power curves be indexed by the *true* ``dS``.
The challenger with advantage ``delta`` has ``mu_ch = mu_king - delta`` for every detector.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from statistics import NormalDist

import numpy as np

from humanizer_eval.scoring.bootstrap import ResamplingPlan
from humanizer_eval.scoring.crown import DEFAULT_MIN_GATE_PASS, crown_decision
from humanizer_eval.scoring.metric import (
    DEFAULT_P,
    DEFAULT_PROB_WEIGHT,
    HumanReference,
    SubmissionRun,
    softmin,
    submission_score,
)

_N01 = NormalDist()

#: Genre proportions from spec Q-2 (defaults until It's AI logs are available).
DEFAULT_GENRES: tuple[tuple[str, float], ...] = (
    ("essay", 0.30),
    ("article", 0.15),
    ("academic", 0.10),
    ("news", 0.10),
    ("letter", 0.10),
    ("review", 0.10),
    ("forum", 0.10),
    ("fiction", 0.05),
)
DEFAULT_LENGTH_BINS: tuple[str, ...] = ("150-200", "200-250", "250-300", "300-350")  # D-55: 150-350


@dataclass(frozen=True)
class DetectorSim:
    name: str
    king_auroc: float  # population AUROC of the king's outputs vs H (before gates and rounding)
    noise_sd: float  # detector-specific per-text noise, latent units
    decimals: int  # reported probability rounded to this many decimals (ties)


DEFAULT_DETECTORS: tuple[DetectorSim, ...] = (
    DetectorSim("det_a", 0.80, 0.6, 4),
    DetectorSim("det_b", 0.75, 0.9, 2),
    DetectorSim("det_c", 0.85, 0.5, 3),
)


@dataclass(frozen=True)
class SimConfig:
    n_inputs: int = 500  # D-24 (owner decision 2026-11-10)
    n_human: int = 500  # D-54: H has the same counts per stratum as the slots (human_factor 1)
    genres: tuple[tuple[str, float], ...] = DEFAULT_GENRES
    length_bins: tuple[str, ...] = DEFAULT_LENGTH_BINS
    detectors: tuple[DetectorSim, ...] = DEFAULT_DETECTORS
    human_sd: float = 1.0
    stratum_sd: float = 0.5
    gate_fail_rate: float = 0.02  # share of outputs at quality level 0
    half_rate: float = 0.0  # share of outputs at quality level 0.5 (D-22)
    score_gain: float = 1.5
    score_bias: float = 1.0
    p: float = DEFAULT_P
    #: weight of the mean-probability term (D-20); the analytic helpers model both terms
    w_prob: float = DEFAULT_PROB_WEIGHT

    @property
    def mean_quality(self) -> float:
        return 1.0 - self.gate_fail_rate - 0.5 * self.half_rate

    @property
    def strata(self) -> tuple[str, ...]:
        return tuple(f"{g}|{b}" for g, _ in self.genres for b in self.length_bins)

    @property
    def stratum_weights(self) -> np.ndarray:
        w = np.array([pg / len(self.length_bins) for _, pg in self.genres for _ in self.length_bins])
        return w / w.sum()


def allocate(n: int, weights: np.ndarray) -> np.ndarray:
    """Integer counts summing to ``n`` proportional to ``weights`` (largest remainder)."""
    w = np.asarray(weights, dtype=float)
    raw = n * w / w.sum()
    counts = np.floor(raw).astype(int)
    rest = n - counts.sum()
    counts[np.argsort(-(raw - counts), kind="stable")[:rest]] += 1
    return counts


def layout(cfg: SimConfig, n: int) -> np.ndarray:
    """Deterministic stratum label per text: same counts per stratum every period (D-54)."""
    return np.repeat(np.array(cfg.strata), allocate(n, cfg.stratum_weights))


def _sd_total(cfg: SimConfig, det: DetectorSim) -> float:
    return math.sqrt(1.0 + cfg.human_sd**2 + 2.0 * det.noise_sd**2)


def king_skill(cfg: SimConfig) -> np.ndarray:
    """Latent mean ``mu_d`` of the king's outputs that yields each detector's ``king_auroc``."""
    return np.array([_sd_total(cfg, d) * _N01.inv_cdf(d.king_auroc) for d in cfg.detectors])


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)
_GH_W = _GH_W / _GH_W.sum()


def expected_p_human(cfg: SimConfig, m: float, det: DetectorSim) -> float:
    """Population mean ``1 - score`` of ungated outputs with latent mean ``m`` (rounding ignored)."""
    sd = math.sqrt(1.0 + cfg.stratum_sd**2 + det.noise_sd**2)
    z = cfg.score_gain * (m + sd * _GH_X) - cfg.score_bias
    return float((_GH_W / (1.0 + np.exp(z))).sum())


def expected_evasion(cfg: SimConfig, mu: np.ndarray) -> list[float]:
    """Population ``E_d`` per detector (D-20, quality levels D-22)."""
    mq = cfg.mean_quality
    ev = []
    for m, d in zip(mu, cfg.detectors, strict=True):
        one_minus_auc = mq * (1.0 - _N01.cdf(m / _sd_total(cfg, d)))
        ph = mq * expected_p_human(cfg, float(m), d) if cfg.w_prob > 0 else 0.0
        ev.append((1.0 - cfg.w_prob) * 2.0 * min(0.5, one_minus_auc) + cfg.w_prob * ph)
    return ev


def expected_score(cfg: SimConfig, mu: np.ndarray) -> float:
    """Population S of a model with latent means ``mu`` (rounding ties ignored)."""
    return float(softmin(expected_evasion(cfg, mu), p=cfg.p))


def advantage_for_delta(cfg: SimConfig, target_delta: float) -> float:
    """Latent advantage ``delta`` such that the challenger's population S exceeds the king's by ``target``."""
    if target_delta <= 0:
        return 0.0
    mu = king_skill(cfg)
    s0 = expected_score(cfg, mu)
    lo, hi = 0.0, float(mu.max()) + 10.0
    if expected_score(cfg, mu - hi) - s0 < target_delta:
        raise ValueError(f"target dS {target_delta} is not reachable from king S = {s0:.4f}")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if expected_score(cfg, mu - mid) - s0 < target_delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class PeriodSample:
    king: SubmissionRun
    challenger: SubmissionRun
    human: HumanReference


def _to_prob(cfg: SimConfig, latent: np.ndarray, decimals: int) -> np.ndarray:
    z = cfg.score_gain * latent - cfg.score_bias
    return np.round(1.0 / (1.0 + np.exp(-z)), decimals)


def simulate_period(
    cfg: SimConfig, rng: np.random.Generator, *, phi: float, advantage: float = 0.0
) -> PeriodSample:
    """One period: king and challenger outputs on the same inputs, plus the human reference set."""
    if not 0 <= phi <= 1:
        raise ValueError("phi must be in [0, 1]")
    out_strata = layout(cfg, cfg.n_inputs)
    hum_strata = layout(cfg, cfg.n_human)
    n, m, n_det = cfg.n_inputs, cfg.n_human, len(cfg.detectors)
    s_index = {s: k for k, s in enumerate(cfg.strata)}
    oi = np.array([s_index[s] for s in out_strata])
    hi = np.array([s_index[s] for s in hum_strata])
    offsets = rng.normal(0.0, cfg.stratum_sd, size=(len(cfg.strata), n_det))

    eta = rng.normal(size=n)
    mu_king = king_skill(cfg)
    runs = []
    for mu in (mu_king, mu_king - advantage):
        a = math.sqrt(phi) * eta + math.sqrt(1.0 - phi) * rng.normal(size=n)
        scores = {}
        for d, det in enumerate(cfg.detectors):
            lat = mu[d] + offsets[oi, d] + a + det.noise_sd * rng.normal(size=n)
            scores[det.name] = _to_prob(cfg, lat, det.decimals)
        r = rng.random(n)
        g, h = cfg.gate_fail_rate, cfg.half_rate
        level = np.where(r < g, 0.0, np.where(r < g + h, 0.5, 1.0))
        runs.append(SubmissionRun(scores, level, out_strata))

    h = rng.normal(0.0, cfg.human_sd, size=m)
    hscores = {}
    for d, det in enumerate(cfg.detectors):
        lat = offsets[hi, d] + h + det.noise_sd * rng.normal(size=m)
        hscores[det.name] = _to_prob(cfg, lat, det.decimals)
    return PeriodSample(runs[0], runs[1], HumanReference(hscores, hum_strata))


def measured_delta(cfg: SimConfig, advantage: float, *, n: int = 200_000, seed: int = 12345) -> float:
    """Large-sample S_ch - S_king including rounding ties (checks the analytic calibration)."""
    big = replace(cfg, n_inputs=n, n_human=n)
    ps = simulate_period(big, np.random.default_rng(seed), phi=0.5, advantage=advantage)
    return (
        submission_score(ps.challenger, ps.human, p=cfg.p, w_prob=cfg.w_prob).s
        - submission_score(ps.king, ps.human, p=cfg.p, w_prob=cfg.w_prob).s
    )


# --- experiments ------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrialSummary:
    phi: float
    target_delta: float  # population dS (analytic)
    advantage: float  # latent advantage that produces it
    n_periods: int
    crowns: int
    mean_delta_hat: float  # mean observed dS over periods
    sd_delta_hat: float  # spread of observed dS over periods (true sampling sd)
    mean_boot_se: float  # mean bootstrap se (should match sd_delta_hat)

    @property
    def rate(self) -> float:
        return self.crowns / self.n_periods

    @property
    def mc_se(self) -> float:
        r = self.rate
        return math.sqrt(r * (1 - r) / self.n_periods)

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        n, r = self.n_periods, self.rate
        den = 1 + z * z / n
        c = (r + z * z / (2 * n)) / den
        h = z * math.sqrt(r * (1 - r) / n + z * z / (4 * n * n)) / den
        return max(0.0, c - h), min(1.0, c + h)


@dataclass(frozen=True)
class _Job:
    cfg: SimConfig
    phi: float
    advantage: float
    periods: tuple[int, ...]
    seed: int
    n_boot: int
    alpha: float
    delta_min: float
    min_gate_pass: float


def _run_job(job: _Job) -> list[tuple[bool, float, float]]:
    out = []
    for r in job.periods:
        # Period r uses the same random stream in every cell (common random numbers), so curves over
        # delta and phi are smooth; the bootstrap stream is independent of the data stream.
        data_ss, boot_ss = np.random.SeedSequence([job.seed, r]).spawn(2)
        ps = simulate_period(job.cfg, np.random.default_rng(data_ss), phi=job.phi, advantage=job.advantage)
        dec = crown_decision(
            ps.king,
            ps.challenger,
            ps.human,
            alpha=job.alpha,
            delta_min=job.delta_min,
            min_gate_pass=job.min_gate_pass,
            n_boot=job.n_boot,
            seed=boot_ss,
            p=job.cfg.p,
            w_prob=job.cfg.w_prob,
        )
        out.append((dec.crown, dec.delta, dec.se))
    return out


def _chunks(n_periods: int, jobs: int) -> list[tuple[int, ...]]:
    return [
        tuple(int(i) for i in c) for c in np.array_split(np.arange(n_periods), max(1, jobs * 4)) if c.size
    ]


def _map(fn, work: list[_Job], jobs: int, pool: ProcessPoolExecutor | None) -> list:
    if pool is not None:
        parts = list(pool.map(fn, work))
    elif jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            parts = list(ex.map(fn, work))
    else:
        parts = [fn(j) for j in work]
    return [x for part in parts for x in part]


def run_trials(
    cfg: SimConfig,
    *,
    phi: float,
    target_delta: float = 0.0,
    n_periods: int = 1000,
    n_boot: int = 2000,
    alpha: float = 0.05,
    delta_min: float = 0.0,
    min_gate_pass: float = DEFAULT_MIN_GATE_PASS,
    seed: int = 0,
    jobs: int = 1,
    pool: ProcessPoolExecutor | None = None,
) -> TrialSummary:
    """Simulate ``n_periods`` independent periods and count crowns. ``target_delta = 0`` is A/A."""
    adv = advantage_for_delta(cfg, target_delta)
    work = [
        _Job(cfg, phi, adv, c, seed, n_boot, alpha, delta_min, min_gate_pass)
        for c in _chunks(n_periods, jobs)
    ]
    res = _map(_run_job, work, jobs, pool)
    crowns = sum(c for c, _, _ in res)
    deltas = np.array([d for _, d, _ in res])
    ses = np.array([s for _, _, s in res])
    return TrialSummary(
        phi=phi,
        target_delta=target_delta,
        advantage=adv,
        n_periods=n_periods,
        crowns=int(crowns),
        mean_delta_hat=float(deltas.mean()),
        sd_delta_hat=float(deltas.std(ddof=1)) if n_periods > 1 else 0.0,
        mean_boot_se=float(ses.mean()),
    )


def default_jobs() -> int:
    return max(1, min(16, os.cpu_count() or 1))


def power_grid(
    cfg: SimConfig,
    deltas: Sequence[float],
    phis: Sequence[float],
    *,
    n_periods: int,
    n_boot: int,
    seed: int = 0,
    jobs: int = 1,
    **kw,
) -> list[TrialSummary]:
    """Crown probability for every (target dS, phi) cell, sharing one process pool."""
    ctx = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else contextlib.nullcontext()
    with ctx as pool:
        return [
            run_trials(
                cfg,
                phi=phi,
                target_delta=delta,
                n_periods=n_periods,
                n_boot=n_boot,
                seed=seed,
                jobs=jobs,
                pool=pool,
                **kw,
            )
            for delta in deltas
            for phi in phis
        ]


# --- where does the bootstrap variance come from? ---------------------------------------------
@dataclass(frozen=True)
class SeDecomposition:
    phi: float
    n_periods: int
    sd_delta_hat: float  # true sampling sd of dS-hat across periods
    se_both: float  # mean bootstrap se, inputs and H resampled (D-23, what the crown test uses)
    se_inputs: float  # mean bootstrap se, only inputs resampled
    se_human: float  # mean bootstrap se, only H resampled


def _decomp_job(job: _Job) -> list[tuple[float, float, float, float]]:
    out = []
    for r in job.periods:
        data_ss, boot_ss = np.random.SeedSequence([job.seed, r]).spawn(2)
        ps = simulate_period(job.cfg, np.random.default_rng(data_ss), phi=job.phi, advantage=job.advantage)
        plan = ResamplingPlan([ps.king, ps.challenger], ps.human)
        w, v = plan.draw_counts(np.random.default_rng(boot_ss), job.n_boot)
        w1, v1 = plan.unit_counts()
        fixed_w = [np.broadcast_to(x, (job.n_boot, x.shape[1])) for x in w1]
        fixed_v = [np.broadcast_to(x, (job.n_boot, x.shape[1])) for x in v1]
        se = []
        for ww, vv in ((w, v), (w, fixed_v), (fixed_w, v)):
            s = plan.scores(ww, vv, p=job.cfg.p, w_prob=job.cfg.w_prob)
            se.append(float((s[1] - s[0]).std(ddof=1)))
        s0 = plan.scores(w1, v1, p=job.cfg.p, w_prob=job.cfg.w_prob)
        out.append((float(s0[1, 0] - s0[0, 0]), *se))
    return out


def se_decomposition(
    cfg: SimConfig, *, phi: float, n_periods: int, n_boot: int, seed: int = 0, jobs: int = 1
) -> SeDecomposition:
    """Compare the true sd of dS-hat (A/A) with bootstrap se under three resampling schemes."""
    work = [_Job(cfg, phi, 0.0, c, seed, n_boot, 0.05, 0.0, 0.0) for c in _chunks(n_periods, jobs)]
    res = np.array(_map(_decomp_job, work, jobs, None))
    return SeDecomposition(
        phi=phi,
        n_periods=n_periods,
        sd_delta_hat=float(res[:, 0].std(ddof=1)),
        se_both=float(res[:, 1].mean()),
        se_inputs=float(res[:, 2].mean()),
        se_human=float(res[:, 3].mean()),
    )
