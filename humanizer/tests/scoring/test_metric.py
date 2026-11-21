import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from humanizer_eval.scoring.metric import (
    HumanReference,
    SubmissionRun,
    apply_gate,
    auroc,
    evasion,
    softmin,
    stratified_auroc,
    stratified_auroc_detail,
    submission_score,
)


def sk_auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return roc_auc_score(np.r_[np.ones(x.size), np.zeros(y.size)], np.r_[x, y])


# --- AUROC -------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("decimals", [None, 1])
def test_auroc_matches_sklearn_random(seed, decimals):
    rng = np.random.default_rng(seed)
    n, m = rng.integers(1, 400, size=2)
    x = rng.normal(0.7, 1.0, n)
    y = rng.normal(0.0, 1.0, m)
    if decimals is not None:  # coarse rounding -> many ties
        x, y = np.round(x, decimals), np.round(y, decimals)
    assert auroc(x, y) == pytest.approx(sk_auc(x, y), abs=1e-12)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        ([1, 2, 3], [2], 0.5),  # below, tie, above
        ([1, 1], [1, 1], 0.5),  # all ties
        ([0, 1], [0, 0, 1], 3.5 / 6),  # x=0: two ties + one loss; x=1: two wins + one tie
        ([5], [1, 2, 3], 1.0),
        ([0], [1, 2, 3], 0.0),
    ],
)
def test_auroc_hand_computed_ties(x, y, expected):
    assert auroc(x, y) == pytest.approx(expected)
    assert sk_auc(x, y) == pytest.approx(expected)


def test_auroc_with_inf_equals_large_finite():
    rng = np.random.default_rng(3)
    x = rng.random(50)
    y = rng.random(60)
    x_inf = x.copy()
    x_inf[:7] = np.inf
    x_big = x.copy()
    x_big[:7] = 10.0
    assert auroc(x_inf, y) == pytest.approx(sk_auc(x_big, y))


def test_auroc_rejects_bad_input():
    with pytest.raises(ValueError):
        auroc([np.nan, 1.0], [0.0])
    with pytest.raises(ValueError):
        auroc([], [0.0])
    with pytest.raises(ValueError):
        auroc([1.0], [-np.inf])


def test_auroc_scales_to_large_inputs():
    # O((n+m) log m): a million pairs-worth of data would be 1e11 pairs for an O(n*m) method.
    rng = np.random.default_rng(0)
    x = rng.normal(1, 1, 300_000)
    y = rng.normal(0, 1, 300_000)
    expected = 0.5 * (1 + math.erf(1 / 2))  # Phi(1/sqrt(2))
    assert auroc(x, y) == pytest.approx(expected, abs=3e-3)


# --- stratification ----------------------------------------------------------------------------
def test_stratified_weights_by_outputs():
    # stratum a: 3 outputs, AUROC 1; stratum b: 1 output, AUROC 0  ->  3/4 * 1 + 1/4 * 0
    xs = [5, 6, 7, 0]
    xl = ["a", "a", "a", "b"]
    ys = [1, 2, 9, 9]
    yl = ["a", "a", "b", "b"]
    r = stratified_auroc_detail(xs, xl, ys, yl)
    assert r.per_stratum == {"a": 1.0, "b": 0.0}
    assert r.weights == {"a": 0.75, "b": 0.25}
    assert r.value == pytest.approx(0.75)
    assert r.excluded == ()


def test_stratified_matches_sklearn_per_stratum():
    rng = np.random.default_rng(7)
    labels = np.array(["essay|short", "essay|long", "news|short", "forum|long"])
    xl = labels[rng.integers(0, 4, 500)]
    yl = labels[rng.integers(0, 4, 700)]
    x = np.round(rng.normal(0.5, 1, 500), 1)
    y = np.round(rng.normal(0.0, 1, 700), 1)
    expected = sum((xl == s).mean() * sk_auc(x[xl == s], y[yl == s]) for s in labels)
    assert stratified_auroc(x, xl, y, yl) == pytest.approx(expected, abs=1e-12)


def test_stratification_removes_genre_offsets():
    # Within each genre outputs and humans are identically distributed (AUROC 0.5), but outputs are
    # concentrated in the genre that every detector scores high: pooling would wrongly say "caught".
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(3, 1, 900), rng.normal(0, 1, 100)]
    xl = ["academic"] * 900 + ["forum"] * 100
    y = np.r_[rng.normal(3, 1, 100), rng.normal(0, 1, 900)]
    yl = ["academic"] * 100 + ["forum"] * 900
    assert auroc(x, y) > 0.85
    assert stratified_auroc(x, xl, y, yl) == pytest.approx(0.5, abs=0.06)


def test_strata_without_humans_are_excluded_and_reported(caplog):
    xs = [5, 6, 0, 0]
    xl = ["a", "a", "orphan", "orphan"]
    ys = [1, 2, 3]
    yl = ["a", "a", "human_only"]
    r = stratified_auroc_detail(xs, xl, ys, yl)
    assert r.excluded == ("orphan",)
    assert r.weights == {"a": 1.0}
    assert r.value == 1.0
    with caplog.at_level("WARNING"):
        assert stratified_auroc(xs, xl, ys, yl) == 1.0
    assert "orphan" in caplog.text


def test_no_common_stratum_raises():
    with pytest.raises(ValueError, match="no stratum"):
        stratified_auroc([1.0], ["a"], [0.0], ["b"])


def test_label_length_mismatch_raises():
    with pytest.raises(ValueError):
        stratified_auroc([1.0, 2.0], ["a"], [0.0], ["a"])


# --- gates -------------------------------------------------------------------------------------
def test_apply_gate_sets_inf_and_copies():
    s = np.array([0.1, 0.2, 0.3])
    g = np.array([True, False, True])
    out = apply_gate(s, g)
    assert out.tolist() == [0.1, math.inf, 0.3]
    assert s.tolist() == [0.1, 0.2, 0.3]
    with pytest.raises(ValueError):
        apply_gate(s, [True, False])
    with pytest.raises(ValueError):
        apply_gate(s, [0, 2, 1])


def test_gate_failures_are_caught_by_every_detector():
    # The output fooling every detector (score 0) failed a gate: it must count as AI, not human.
    human = HumanReference({"d1": [0.2, 0.4], "d2": [0.3, 0.5]}, ["s", "s"])
    scores = {"d1": [0.0, 0.9], "d2": [0.0, 0.9]}
    passed = submission_score(SubmissionRun(scores, [True, True], ["s", "s"]), human, w_prob=0.0)
    failed = submission_score(SubmissionRun(scores, [False, True], ["s", "s"]), human, w_prob=0.0)
    assert passed.auroc == {"d1": 0.5, "d2": 0.5}
    assert passed.s == pytest.approx(1.0)
    assert failed.auroc == {"d1": 1.0, "d2": 1.0}
    assert failed.s == 0.0
    assert failed.gate_pass_rate == 0.5
    # probability term: the failed output counts as P(human) = 0, not 1
    mixed = submission_score(SubmissionRun(scores, [False, True], ["s", "s"]), human)
    assert mixed.p_human == {"d1": pytest.approx(0.05), "d2": pytest.approx(0.05)}


# --- evasion and soft-min ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("a", "e"), [(0.5, 1.0), (1.0, 0.0), (0.75, 0.5), (0.9, 0.2), (0.3, 1.0), (0.0, 1.0)]
)
def test_evasion(a, e):
    assert evasion(a) == pytest.approx(e)


def test_evasion_vectorized_and_range():
    assert np.allclose(evasion(np.array([0.5, 0.8])), [1.0, 0.4])
    with pytest.raises(ValueError):
        evasion(1.2)
    with pytest.raises(ValueError):
        evasion(np.nan)


def test_softmin_basic_values():
    assert softmin([0.4, 0.4, 0.4]) == pytest.approx(0.4)
    v = [0.4, 0.5, 0.3]
    expected = (np.mean(np.array(v) ** -4)) ** (-1 / 4)
    assert softmin(v) == pytest.approx(expected)
    assert min(v) < softmin(v) < np.mean(v)
    assert softmin([0.2, 1.0], p=-1) == pytest.approx(1 / ((5 + 1) / 2))  # harmonic mean


def test_softmin_any_zero_is_zero():
    assert softmin([0.0, 0.9, 1.0]) == 0.0
    assert softmin([0.0]) == 0.0
    assert softmin([1e-12, 1.0]) > 0.0


def test_softmin_tends_to_min_as_p_decreases():
    v = [0.3, 0.6, 0.9]
    assert softmin(v, p=-200) == pytest.approx(0.3, rel=0.01)


def test_softmin_axis_matches_scalar():
    rng = np.random.default_rng(0)
    m = rng.random((3, 6))
    m[1, 2] = 0.0
    out = softmin(m, axis=0)
    assert out.shape == (6,)
    for j in range(6):
        assert out[j] == pytest.approx(softmin(m[:, j]))
    assert out[2] == 0.0


@pytest.mark.parametrize("bad", [[], [-0.1, 0.5], [np.nan], [np.inf]])
def test_softmin_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        softmin(bad)


@pytest.mark.parametrize("p", [0, 1, 2.5])
def test_softmin_needs_negative_p(p):
    with pytest.raises(ValueError):
        softmin([0.5], p=p)


# --- submission score and containers -----------------------------------------------------------
def test_submission_score_combines_detectors():
    human = HumanReference({"a": [0.0, 1.0], "b": [0.0, 1.0]}, ["s", "s"])
    # detector a: outputs (0.5, 2.0) vs humans (0, 1): wins 1+2 of 4 -> AUROC 0.75 -> E 0.5
    # detector b: outputs (0.5, 0.5): wins 1+1 of 4 -> AUROC 0.5 -> E 1.0
    run = SubmissionRun({"a": [0.5, 2.0], "b": [0.5, 0.5]}, [True, True], ["s", "s"])
    r = submission_score(run, human, w_prob=0.0)  # AUROC term only (scores here are not probabilities)
    assert r.auroc == {"a": 0.75, "b": 0.5}
    assert r.evasion == {"a": 0.5, "b": 1.0}
    assert r.s == pytest.approx(((0.5**-4 + 1.0) / 2) ** -0.25)
    assert r.n == 2
    assert r.gate_pass_rate == 1.0


def test_detector_mismatch_raises():
    human = HumanReference({"a": [0.0], "b": [0.0]}, ["s"])
    run = SubmissionRun({"a": [0.5]}, [True], ["s"])
    with pytest.raises(ValueError, match="detector mismatch"):
        submission_score(run, human)


def test_containers_validate_and_freeze():
    with pytest.raises(ValueError):
        SubmissionRun({"a": [0.1, 0.2], "b": [0.1]}, [True, True], ["s", "s"])
    with pytest.raises(ValueError):
        SubmissionRun({"a": [0.1, 0.2]}, [True, True], ["s", "s"], input_ids=["x", "x"])
    with pytest.raises(ValueError):
        HumanReference({"a": [np.inf]}, ["s"])
    src = np.array([0.1, 0.2])
    run = SubmissionRun({"a": src}, [True, False], ["s", "s"])
    src[0] = 9.0  # the run keeps its own copy
    assert run.raw_scores("a")[0] == 0.1
    with pytest.raises(ValueError):
        run.raw_scores("a")[0] = 1.0  # read-only
    assert run.gated_scores("a").tolist() == [0.1, math.inf]


# --- probability term (D-20, 2026-11-09) ---------------------------------------------------------
def test_prob_term_hand_computed():
    human = HumanReference({"a": [0.1, 0.2]}, ["s", "s"])
    r = submission_score(SubmissionRun({"a": [0.3, 0.9]}, [True, True], ["s", "s"]), human)
    assert r.auroc == {"a": 1.0}  # both outputs rank above every human
    assert r.p_human == {"a": pytest.approx(0.4)}  # mean(1 - 0.3, 1 - 0.9)
    assert r.evasion == {"a": pytest.approx(0.5 * 0.0 + 0.5 * 0.4)}
    assert r.s == pytest.approx(0.2)


def test_prob_term_rewards_lower_probabilities_without_auroc_change():
    human = HumanReference({"a": [0.01, 0.02, 0.03]}, ["s"] * 3)
    raw = submission_score(SubmissionRun({"a": [0.99, 0.98, 0.97]}, [True] * 3, ["s"] * 3), human)
    softer = submission_score(SubmissionRun({"a": [0.40, 0.50, 0.60]}, [True] * 3, ["s"] * 3), human)
    assert raw.auroc == softer.auroc == {"a": 1.0}
    assert softer.s > raw.s > 0


def test_prob_weight_bounds_and_auroc_only():
    human = HumanReference({"a": [0.1, 0.2]}, ["s", "s"])
    run = SubmissionRun({"a": [0.15, 0.9]}, [True, True], ["s", "s"])
    only_auc = submission_score(run, human, w_prob=0.0)
    only_prob = submission_score(run, human, w_prob=1.0)
    assert only_auc.s == pytest.approx(evasion(only_auc.auroc["a"]))
    assert only_prob.s == pytest.approx(only_prob.p_human["a"])
    with pytest.raises(ValueError):
        submission_score(run, human, w_prob=1.5)


def test_prob_term_needs_probabilities():
    human = HumanReference({"a": [0.0, 1.0]}, ["s", "s"])
    run = SubmissionRun({"a": [0.5, 2.0]}, [True, True], ["s", "s"])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        submission_score(run, human)


def test_prob_term_uses_same_outputs_as_auroc():
    # stratum "x" has no human texts: excluded from both terms
    human = HumanReference({"a": [0.1, 0.2]}, ["s", "s"])
    run = SubmissionRun({"a": [0.3, 0.0]}, [True, True], ["s", "x"])
    r = submission_score(run, human)
    assert r.excluded_strata == ("x",) and r.p_human == {"a": pytest.approx(0.7)}


def test_stratified_auroc_never_exceeds_one_by_round_off():
    # stratum weights 1/13, 6/13, 3/13, 3/13 add up to 1.0000000000000002 in floating point
    sizes = (1, 6, 3, 3)
    strata = [f"s{k}" for k, n in enumerate(sizes) for _ in range(n)]
    human_strata = [f"s{k}" for k in range(len(sizes))]
    r = stratified_auroc_detail([0.99] * len(strata), strata, [0.01] * len(sizes), human_strata)
    assert r.value == 1.0
    run = SubmissionRun({"d": [0.99] * len(strata)}, [1.0] * len(strata), strata)
    res = submission_score(run, HumanReference({"d": [0.01] * len(sizes)}, human_strata))
    assert res.auroc["d"] == 1.0
