import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from humanizer_eval.scoring.metric import submission_score
from humanizer_eval.scoring.simulate import (
    SimConfig,
    advantage_for_delta,
    allocate,
    expected_score,
    king_skill,
    layout,
    measured_delta,
    run_trials,
    simulate_period,
)

ROOT = Path(__file__).resolve().parents[2]


def test_allocate_sums_and_is_proportional():
    w = np.array([0.3, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    c = allocate(750, w)
    assert c.sum() == 750
    assert np.abs(c - 750 * w).max() < 1
    assert allocate(7, np.array([1.0, 1.0, 1.0])).tolist() == [3, 2, 2]


def test_layout_is_fixed_per_period():
    cfg = SimConfig()
    a, b = layout(cfg, 750), layout(cfg, 750)
    assert np.array_equal(a, b)
    assert len(set(a)) == len(cfg.strata) == 32  # 8 genres x 4 length bins (D-55: 150-350)


def test_king_auroc_matches_configuration():
    cfg = replace(SimConfig(), n_inputs=100_000, n_human=100_000, gate_fail_rate=0.0)
    ps = simulate_period(cfg, np.random.default_rng(0), phi=0.5)
    r = submission_score(ps.king, ps.human, w_prob=cfg.w_prob)
    for det in cfg.detectors:
        assert r.auroc[det.name] == pytest.approx(det.king_auroc, abs=0.006)


@pytest.mark.parametrize("w_prob", [0.0, 0.5])
def test_expected_score_matches_simulation(w_prob):
    # P(human) depends on the period's stratum offsets (sd of S ~0.008 per period): average 8 periods
    cfg = replace(SimConfig(), n_inputs=20_000, n_human=20_000, gate_fail_rate=0.04, half_rate=0.1,
                  w_prob=w_prob)  # fmt: skip
    s = [
        submission_score(ps.king, ps.human, w_prob=w_prob).s
        for ps in (simulate_period(cfg, np.random.default_rng(k), phi=0.5) for k in range(8))
    ]
    assert np.mean(s) == pytest.approx(expected_score(cfg, king_skill(cfg)), abs=0.008)


def test_advantage_for_delta_hits_target():
    cfg = SimConfig()
    mu = king_skill(cfg)
    for target in (0.01, 0.05, 0.1):
        adv = advantage_for_delta(cfg, target)
        assert expected_score(cfg, mu - adv) - expected_score(cfg, mu) == pytest.approx(target, abs=1e-9)
    assert advantage_for_delta(cfg, 0.0) == 0.0
    assert measured_delta(cfg, advantage_for_delta(cfg, 0.05), n=100_000) == pytest.approx(0.05, abs=0.01)
    with pytest.raises(ValueError):
        advantage_for_delta(cfg, 2.0)


def test_simulate_period_is_deterministic_and_paired():
    cfg = replace(SimConfig(), n_inputs=200, n_human=300)
    a = simulate_period(cfg, np.random.default_rng(1), phi=0.5)
    b = simulate_period(cfg, np.random.default_rng(1), phi=0.5)
    for d in a.king.detectors:
        assert np.array_equal(a.king.raw_scores(d), b.king.raw_scores(d))
        assert np.array_equal(a.human.scores(d), b.human.scores(d))
    assert np.array_equal(a.king.strata, a.challenger.strata)
    assert a.human.m == 300 and a.king.n == 200


def test_phi_controls_paired_correlation():
    cfg = replace(SimConfig(), n_inputs=20_000, n_human=100)

    def corr(phi):
        ps = simulate_period(cfg, np.random.default_rng(2), phi=phi)
        d = cfg.detectors[0].name
        return np.corrcoef(ps.king.raw_scores(d), ps.challenger.raw_scores(d))[0, 1]

    c0, c5, c8 = corr(0.0), corr(0.5), corr(0.8)
    assert c0 < c5 < c8
    assert c0 > 0.1  # shared stratum offsets correlate scores even at phi = 0
    with pytest.raises(ValueError):
        simulate_period(cfg, np.random.default_rng(0), phi=1.5)


def test_run_trials_is_reproducible_across_jobs():
    cfg = replace(SimConfig(), n_inputs=120, n_human=150)
    a = run_trials(cfg, phi=0.5, target_delta=0.05, n_periods=8, n_boot=100, seed=3, jobs=1)
    b = run_trials(cfg, phi=0.5, target_delta=0.05, n_periods=8, n_boot=100, seed=3, jobs=2)
    assert a == b
    assert 0 <= a.crowns <= 8


def test_script_quick_smoke(capsys):
    spec = importlib.util.spec_from_file_location("simulate_crown", ROOT / "scripts" / "simulate_crown.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    argv = ["--quick", "--aa-periods", "6", "--power-periods", "4", "--n-boot", "60", "--jobs", "1"]
    argv += ["--deltas", "0.05", "--phis", "0,0.8", "--n-inputs", "120", "--n-human", "150"]
    argv += ["--decompose", "--decompose-periods", "4"]
    assert mod.main(argv) == 0
    out = capsys.readouterr().out
    assert "## A/A" in out and "## Power" in out
    assert "| 0.05 |" in out
    assert "## A/A bootstrap se by resampling scheme" in out
