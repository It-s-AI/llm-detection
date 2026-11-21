from dataclasses import replace

import numpy as np
import pytest

from humanizer_eval.scoring.bootstrap import BLOCK, ResamplingPlan, paired_bootstrap
from humanizer_eval.scoring.metric import HumanReference, SubmissionRun, group_indices, submission_score
from humanizer_eval.scoring.simulate import (
    SimConfig,
    advantage_for_delta,
    expected_score,
    king_skill,
    run_trials,
    simulate_period,
)

SMALL = replace(SimConfig(), n_inputs=240, n_human=320)


def period(seed=0, phi=0.5, advantage=0.0, cfg=SMALL):
    return simulate_period(cfg, np.random.default_rng(seed), phi=phi, advantage=advantage)


def _materialize(run, idx):
    return SubmissionRun(
        {d: run.raw_scores(d)[idx] for d in run.detectors}, run.quality[idx], run.strata[idx]
    )


def test_fast_replicates_equal_naive_resampling():
    ps = period(seed=1)
    plan = ResamplingPlan([ps.king, ps.challenger], ps.human)
    w, v = plan.draw_counts(np.random.default_rng(5), 6)
    fast = plan.scores(w, v)
    hum_groups = group_indices(ps.human.strata)
    for b in range(6):
        oi = np.concatenate([np.repeat(st.out_idx, w[s][b].astype(int)) for s, st in enumerate(plan.strata)])
        hi = np.concatenate(
            [np.repeat(hum_groups[st.label], v[s][b].astype(int)) for s, st in enumerate(plan.strata)]
        )
        human_b = HumanReference({d: ps.human.scores(d)[hi] for d in ps.human.detectors}, ps.human.strata[hi])
        assert fast[0, b] == pytest.approx(submission_score(_materialize(ps.king, oi), human_b).s, abs=1e-12)
        assert fast[1, b] == pytest.approx(
            submission_score(_materialize(ps.challenger, oi), human_b).s, abs=1e-12
        )


def test_unit_counts_reproduce_point_estimate():
    ps = period(seed=2)
    plan = ResamplingPlan([ps.king, ps.challenger], ps.human)
    s = plan.scores(*plan.unit_counts())
    assert s[0, 0] == pytest.approx(submission_score(ps.king, ps.human).s, abs=1e-12)
    assert s[1, 0] == pytest.approx(submission_score(ps.challenger, ps.human).s, abs=1e-12)


def test_resampling_is_within_strata():
    ps = period(seed=3)
    plan = ResamplingPlan([ps.king, ps.challenger], ps.human)
    w, v = plan.draw_counts(np.random.default_rng(0), 50)
    for s, st in enumerate(plan.strata):
        assert (w[s].sum(axis=1) == st.n).all()
        assert (v[s].sum(axis=1) == st.m).all()
        assert w[s].shape == (50, st.n) and v[s].shape == (50, st.m)


def test_deterministic_given_seed():
    ps = period(seed=4)
    a = paired_bootstrap(ps.king, ps.challenger, ps.human, n_boot=600, seed=11, return_replicates=True)
    b = paired_bootstrap(ps.king, ps.challenger, ps.human, n_boot=600, seed=11, return_replicates=True)
    c = paired_bootstrap(ps.king, ps.challenger, ps.human, n_boot=600, seed=12, return_replicates=True)
    assert np.array_equal(a.replicates, b.replicates)
    assert a.lower == b.lower
    assert not np.array_equal(a.replicates, c.replicates)
    assert a.delta == c.delta  # the point estimate does not depend on the seed


def test_block_prefix_is_stable_across_n_boot():
    ps = period(seed=5)
    short = paired_bootstrap(
        ps.king, ps.challenger, ps.human, n_boot=2 * BLOCK, seed=3, return_replicates=True
    )
    long = paired_bootstrap(
        ps.king, ps.challenger, ps.human, n_boot=4 * BLOCK, seed=3, return_replicates=True
    )
    assert np.array_equal(short.replicates, long.replicates[: 2 * BLOCK])


def test_identical_outputs_never_crown():
    ps = period(seed=6)
    r = paired_bootstrap(ps.king, ps.king, ps.human, n_boot=500, seed=0, return_replicates=True)
    assert r.delta == 0.0
    assert np.all(r.replicates == 0.0)
    assert r.lower == 0.0


def test_point_estimate_and_lower_bound_order():
    ps = period(seed=7, advantage=advantage_for_delta(SMALL, 0.15))
    r = paired_bootstrap(ps.king, ps.challenger, ps.human, n_boot=1000, seed=0)
    assert r.delta == pytest.approx(r.s_challenger - r.s_king)
    assert r.lower < r.delta
    assert r.lower > 0  # a large true advantage is detected
    assert r.se > 0


def test_misaligned_runs_rejected():
    ps = period(seed=8)
    k = ps.king
    perm = np.random.default_rng(0).permutation(k.n)
    shuffled = _materialize(ps.challenger, perm)
    with pytest.raises(ValueError, match="not aligned"):
        paired_bootstrap(k, shuffled, ps.human, n_boot=10)
    shorter = _materialize(ps.challenger, np.arange(k.n - 1))
    with pytest.raises(ValueError, match="same inputs"):
        paired_bootstrap(k, shorter, ps.human, n_boot=10)
    ids = [f"in{i}" for i in range(k.n)]
    k_ids = SubmissionRun({d: k.raw_scores(d) for d in k.detectors}, k.gate_pass, k.strata, ids)
    c_ids = SubmissionRun(
        {d: ps.challenger.raw_scores(d) for d in k.detectors}, ps.challenger.gate_pass, k.strata, ids[::-1]
    )
    with pytest.raises(ValueError, match="input_ids"):
        paired_bootstrap(k_ids, c_ids, ps.human, n_boot=10)


def test_bad_parameters_rejected():
    ps = period(seed=9)
    with pytest.raises(ValueError):
        paired_bootstrap(ps.king, ps.challenger, ps.human, n_boot=0)
    with pytest.raises(ValueError):
        paired_bootstrap(ps.king, ps.challenger, ps.human, alpha=1.5)


def test_small_aa_false_crown_rate_and_se_calibration():
    # Quick A/A: n=240, B=400, 200 periods. Expected false crowns ~ 10 (5%); the bound is loose
    # (P[Binomial(200, 0.05) > 20] < 1e-3) but catches a broken test. Deterministic seeds.
    r = run_trials(SMALL, phi=0.5, n_periods=200, n_boot=400, seed=99)
    assert r.crowns <= 20
    assert abs(r.mean_delta_hat) < 3 * r.sd_delta_hat / np.sqrt(r.n_periods)
    # bootstrap se tracks the true sampling sd of dS-hat (within 25%)
    assert r.mean_boot_se == pytest.approx(r.sd_delta_hat, rel=0.25)


def test_coverage_of_lower_bound_under_alternative():
    # With a real advantage the one-sided bound must still sit below the true dS ~95% of the time.
    target = 0.03
    adv = advantage_for_delta(SMALL, target)
    true_delta = expected_score(SMALL, king_skill(SMALL) - adv) - expected_score(SMALL, king_skill(SMALL))
    assert true_delta == pytest.approx(target, abs=1e-6)
    misses = 0
    for r in range(150):
        ps = period(seed=1000 + r, phi=0.5, advantage=adv)
        b = paired_bootstrap(ps.king, ps.challenger, ps.human, n_boot=400, seed=r)
        misses += b.lower > true_delta
    assert misses <= 16  # expected ~7.5; P[Binomial(150, .05) > 16] < 1e-3


@pytest.mark.slow
def test_aa_false_crown_rate_at_production_size():
    # D-23/D-24: n = H = 500, B = 2000, alpha = 0.05, metric D-20 (w = 0.5). Measured on 3000 periods (n = 750):
    # 0.055 +- 0.004 (w = 0: 0.052), so the bound is 3 MC sd to keep the test from flaking.
    r = run_trials(SimConfig(), phi=0.5, n_periods=1000, n_boot=2000, seed=2026, jobs=8)
    assert r.rate <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / r.n_periods)


def _with_levels(run, seed):
    q = np.random.default_rng(seed).choice([0.0, 0.5, 1.0], size=run.n, p=[0.2, 0.3, 0.5])
    return SubmissionRun({d: run.raw_scores(d) for d in run.detectors}, q, run.strata)


def test_fast_replicates_equal_naive_with_quality_levels_and_penalty():
    ps = period(seed=4)
    king, ch = _with_levels(ps.king, 1), _with_levels(ps.challenger, 2)
    plan = ResamplingPlan([king, ch], ps.human)
    w, v = plan.draw_counts(np.random.default_rng(9), 5)
    fast = plan.scores(w, v, penalty=0.3)
    hum_groups = group_indices(ps.human.strata)
    for b in range(5):
        oi = np.concatenate([np.repeat(st.out_idx, w[s][b].astype(int)) for s, st in enumerate(plan.strata)])
        hi = np.concatenate(
            [np.repeat(hum_groups[st.label], v[s][b].astype(int)) for s, st in enumerate(plan.strata)]
        )
        human_b = HumanReference({d: ps.human.scores(d)[hi] for d in ps.human.detectors}, ps.human.strata[hi])
        for k, run in enumerate((king, ch)):
            naive = submission_score(_materialize(run, oi), human_b, penalty=0.3).s
            assert fast[k, b] == pytest.approx(naive, abs=1e-12)


def test_minor_level_earns_half_and_severe_matches_plus_inf():
    ps = period(seed=5)
    run = ps.challenger
    sc = {d: run.raw_scores(d) for d in run.detectors}
    full = submission_score(SubmissionRun(sc, np.ones(run.n), run.strata), ps.human)
    half = submission_score(SubmissionRun(sc, np.full(run.n, 0.5), run.strata), ps.human)
    for d in run.detectors:
        assert 1 - half.auroc[d] == pytest.approx(0.5 * (1 - full.auroc[d]))
        assert half.p_human[d] == pytest.approx(0.5 * full.p_human[d])
    gate = np.random.default_rng(0).random(run.n) > 0.3
    as_bool = submission_score(SubmissionRun(sc, gate, run.strata), ps.human, w_prob=0.0)
    inf = {d: np.where(gate, sc[d], np.inf) for d in sc}
    as_inf = submission_score(SubmissionRun(inf, np.ones(run.n), run.strata), ps.human, w_prob=0.0)
    assert as_bool.s == pytest.approx(as_inf.s, abs=1e-12)
