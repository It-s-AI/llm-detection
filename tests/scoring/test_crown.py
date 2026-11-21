import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from humanizer_eval.scoring.crown import BURN_KEY, KingEntry, KingState, crown_decision
from humanizer_eval.scoring.metric import SubmissionRun
from humanizer_eval.scoring.simulate import SimConfig, advantage_for_delta, simulate_period

CFG = replace(SimConfig(), n_inputs=300, n_human=400, gate_fail_rate=0.0)
T0 = datetime(2026, 11, 8, tzinfo=UTC)


def period(seed=0, delta=0.0, phi=0.5):
    adv = advantage_for_delta(CFG, delta)
    return simulate_period(CFG, np.random.default_rng(seed), phi=phi, advantage=adv)


def with_gate_failures(run: SubmissionRun, rate: float, seed: int = 0) -> SubmissionRun:
    gate = np.random.default_rng(seed).random(run.n) >= rate
    return SubmissionRun({d: run.raw_scores(d) for d in run.detectors}, gate, run.strata)


# --- crown decision ----------------------------------------------------------------------------
def test_clear_winner_is_crowned():
    ps = period(seed=1, delta=0.15)
    d = crown_decision(ps.king, ps.challenger, ps.human, n_boot=500, seed=0)
    assert d.crown
    assert d.lower > 0 and d.delta > d.lower
    assert d.s_challenger - d.s_king == pytest.approx(d.delta)
    assert all(r.startswith("PASS") for r in d.reasons)
    assert d.challenger_gate_pass_rate == 1.0


def test_identical_challenger_is_not_crowned():
    ps = period(seed=2)
    d = crown_decision(ps.king, ps.king, ps.human, n_boot=300, seed=0)
    assert not d.crown
    assert d.lower == 0.0
    assert d.reasons[0].startswith("FAIL")


def test_optional_gate_floor_blocks_crown():
    ps = period(seed=3, delta=0.25)
    ch = with_gate_failures(ps.challenger, 0.08, seed=1)  # ~92% pass
    d = crown_decision(ps.king, ch, ps.human, n_boot=500, seed=0, min_gate_pass=0.95)
    assert d.lower > 0  # the statistical criterion alone would crown
    assert d.challenger_gate_pass_rate < 0.95
    assert not d.crown
    assert d.reasons[0].startswith("PASS") and d.reasons[1].startswith("FAIL")


def test_no_aggregate_gate_floor_by_default():
    # D-22 (2026-11-09): failures cost the challenger per output inside S, not via a 95% floor
    ps = period(seed=3, delta=0.25)
    ch = with_gate_failures(ps.challenger, 0.08, seed=1)
    d = crown_decision(ps.king, ch, ps.human, n_boot=500, seed=0)
    full = crown_decision(ps.king, ps.challenger, ps.human, n_boot=500, seed=0)
    assert d.crown and d.min_gate_pass == 0.0
    assert d.s_challenger < full.s_challenger  # the failed outputs lost their contribution


def test_gate_threshold_is_inclusive():
    ps = period(seed=3, delta=0.25)
    gate = np.ones(ps.challenger.n, bool)
    gate[: int(0.05 * ps.challenger.n)] = False  # exactly 95% pass
    ch = SubmissionRun(
        {d: ps.challenger.raw_scores(d) for d in ps.challenger.detectors}, gate, ps.king.strata
    )
    d = crown_decision(ps.king, ch, ps.human, n_boot=300, seed=0)
    assert d.challenger_gate_pass_rate == pytest.approx(0.95)
    assert d.reasons[1].startswith("PASS")


def test_king_gate_rate_is_reported_not_used():
    ps = period(seed=4, delta=0.15)
    weak_king = with_gate_failures(ps.king, 0.3)
    d = crown_decision(weak_king, ps.challenger, ps.human, n_boot=300, seed=0)
    assert d.king_gate_pass_rate < 0.8
    assert d.crown


def test_delta_min_raises_the_bar():
    ps = period(seed=5, delta=0.15)
    base = crown_decision(ps.king, ps.challenger, ps.human, n_boot=500, seed=0)
    strict = crown_decision(ps.king, ps.challenger, ps.human, n_boot=500, seed=0, delta_min=base.lower + 1e-9)
    assert base.crown and not strict.crown
    assert strict.lower == base.lower  # same seed, same replicates


def test_crown_is_strict_inequality():
    ps = period(seed=6)
    d = crown_decision(ps.king, ps.king, ps.human, n_boot=200, seed=0, delta_min=0.0)
    assert d.lower == 0.0 and not d.crown


# --- king state and payouts --------------------------------------------------------------------
def test_baseline_king_share_is_burned():
    ks = KingState.initial("baseline-v1", "owner-hk", T0)
    assert ks.current.is_baseline
    assert ks.payout_weights() == {BURN_KEY: 1.0}


def test_empty_state_burns_everything():
    assert KingState().payout_weights() == {BURN_KEY: 1.0}


def test_85_10_5_with_growing_history():
    ks = KingState.initial("baseline-v1", "owner-hk", T0)
    ks.apply_crown("sub-a", "hk-a", T0 + timedelta(days=1), 0.03)
    # baseline / missing previous-king slots roll to the current king (D-12, as in SN34)
    assert ks.payout_weights() == pytest.approx({"hk-a": 1.0})
    ks.apply_crown("sub-b", "hk-b", T0 + timedelta(days=2), 0.02)
    assert ks.payout_weights() == pytest.approx({"hk-b": 0.90, "hk-a": 0.10})
    ks.apply_crown("sub-c", "hk-c", T0 + timedelta(days=3), 0.01)
    assert ks.payout_weights() == pytest.approx({"hk-c": 0.85, "hk-b": 0.10, "hk-a": 0.05})
    ks.apply_crown("sub-d", "hk-d", T0 + timedelta(days=4), 0.04)
    w = ks.payout_weights()
    assert w == pytest.approx({"hk-d": 0.85, "hk-c": 0.10, "hk-b": 0.05})
    assert BURN_KEY not in w
    assert sum(w.values()) == pytest.approx(1.0)


def test_miners_without_baseline_fewer_than_three():
    ks = KingState()
    ks.apply_crown("sub-a", "hk-a", T0, 0.03)
    assert ks.payout_weights() == pytest.approx({"hk-a": 1.0})
    ks.apply_crown("sub-b", "hk-b", T0 + timedelta(hours=1), 0.01)
    assert ks.payout_weights() == pytest.approx({"hk-b": 0.90, "hk-a": 0.10})


def test_previous_kings_are_distinct_hotkeys():
    ks = KingState.initial("baseline-v1", "owner-hk", T0)
    ks.apply_crown("sub-a1", "hk-a", T0 + timedelta(days=1), 0.03)
    ks.apply_crown("sub-b1", "hk-b", T0 + timedelta(days=2), 0.02)
    ks.apply_crown("sub-a2", "hk-a", T0 + timedelta(days=3), 0.01)  # A dethrones B with a new model
    # slots: hk-a (current), hk-b, then the next distinct hotkey is the baseline -> rolls to hk-a
    assert ks.payout_weights() == pytest.approx({"hk-a": 0.90, "hk-b": 0.10})
    assert [e.submission_id for e in ks.payout_slots()] == ["sub-a2", "sub-b1", "baseline-v1"]


def test_same_hotkey_crowning_itself_twice():
    ks = KingState()
    ks.apply_crown("sub-a1", "hk-a", T0, 0.03)
    ks.apply_crown("sub-a2", "hk-a", T0 + timedelta(days=1), 0.01)
    assert ks.payout_weights() == pytest.approx({"hk-a": 1.0})


def test_apply_crown_validation():
    ks = KingState.initial("baseline-v1", "owner-hk", T0)
    ks.apply_crown("sub-a", "hk-a", T0 + timedelta(days=1), 0.03)
    with pytest.raises(ValueError, match="already crowned"):
        ks.apply_crown("sub-a", "hk-x", T0 + timedelta(days=2), 0.01)
    with pytest.raises(ValueError, match="earlier"):
        ks.apply_crown("sub-b", "hk-b", T0, 0.01)
    with pytest.raises(ValueError, match="timezone"):
        ks.apply_crown("sub-b", "hk-b", datetime(2026, 12, 1), 0.01)
    with pytest.raises(ValueError, match="reserved"):
        ks.apply_crown("sub-b", BURN_KEY, T0 + timedelta(days=2), 0.01)
    with pytest.raises(ValueError, match="finite"):
        ks.apply_crown("sub-b", "hk-b", T0 + timedelta(days=2), float("nan"))
    assert len(ks.history) == 2


def test_history_must_be_ordered_and_unique():
    e1 = KingEntry("s1", "h1", T0 + timedelta(days=1), 0.1)
    e2 = KingEntry("s2", "h2", T0, 0.1)
    with pytest.raises(ValueError, match="ordered"):
        KingState([e1, e2])
    with pytest.raises(ValueError, match="duplicate"):
        KingState([e2, KingEntry("s2", "h3", T0 + timedelta(days=1), 0.1)])


def test_json_round_trip():
    ks = KingState.initial("baseline-v1", "owner-hk", T0)
    ks.apply_crown("sub-a", "hk-a", T0 + timedelta(days=1, minutes=3), 0.031)
    ks.apply_crown("sub-b", "hk-b", T0 + timedelta(days=5), 0.0125)
    s = ks.to_json()
    back = KingState.from_json(s)
    assert back == ks
    assert back.payout_weights() == ks.payout_weights()
    assert back.history[0].delta_s is None and back.history[0].is_baseline
    assert back.history[1].crowned_at == T0 + timedelta(days=1, minutes=3)
    d = json.loads(s)
    assert d["version"] == 1 and len(d["history"]) == 3
    d["version"] = 99
    with pytest.raises(ValueError, match="version"):
        KingState.from_dict(d)
