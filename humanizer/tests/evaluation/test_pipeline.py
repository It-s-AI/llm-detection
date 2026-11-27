import json
import math

import numpy as np
import pytest

from humanizer_eval.billing import Accounts, BillingService, PriceCard
from humanizer_eval.detectors.base import DetectorScore, text_hash
from humanizer_eval.detectors.fake import FakeDetector
from humanizer_eval.evaluation import (
    EvalConfig,
    EvaluationPaused,
    InsufficientBalance,
    KingRun,
    PeriodContext,
    ReplayRunner,
    RunOutput,
    Submission,
    SubmissionStore,
    evaluate_submission,
)
from humanizer_eval.gates.judges import GateResult
from humanizer_eval.llm.openrouter import ChatResult
from humanizer_eval.runner.evaluate import fetch_script, pod_script
from humanizer_eval.scoring.crown import KingState

KEY = bytes(range(32))
JUDGE = "openai/gpt-6-luna"
DETS = ("gptzero", "pangram", "zerogpt")
N = 40


def _ctx() -> PeriodContext:
    rng = np.random.default_rng(0)
    items, hs, hstrata = [], {d: [] for d in DETS}, []
    for i in range(N):
        g = ("essay", "academic")[i % 2]
        items.append(
            {
                "id": f"in{i:02d}",
                "text": f"Input number {i} has a few words in it about topic {i}.",
                "slot": {"genre": g, "length_bin": "150-199"},
                "facts": [],
            }
        )
        hstrata.append(f"{g}|150-199")
        for d in DETS:
            hs[d].append(float(rng.uniform(0.0, 0.6)))
    return PeriodContext("p-test", b"\x01" * 32, items, [f"h{i}" for i in range(N)], hstrata, hs)


class FakeLLM:
    def chat(self, model, messages, **kw):
        return ChatResult("{}", model, "fake", 100, 20, 0.0001, None, json={})


class FakeGates:
    """Level by marker in the output; one judge call per output (metered)."""

    def __init__(self, llm, broken=False):
        self.llm = llm
        self.broken = broken

    def check(self, source, output, facts):
        if not output.strip():
            return GateResult("fail", {"G0": False}, level=0.0, levels={"G0": 0.0})
        self.llm.chat(JUDGE, [{"role": "user", "content": output}])
        if self.broken:
            return GateResult("error", reasons=["judge error"])
        if "BROKEN" in output:
            return GateResult("fail", level=0.0, levels={"G3": 0.0})
        if "ODD" in output:
            return GateResult("fail", level=0.5, levels={"G3": 0.5})
        if "SLIP" in output:  # G3 minor slips (D-22)
            return GateResult("fail", level=0.75, levels={"G3": 0.75})
        return GateResult("pass", level=1.0, levels={"G1": 1.0, "G2": 1.0, "G3": 1.0})


def _detectors():
    return {d: FakeDetector(d, rules=[(lambda t: "human" in t, 0.05)], lo=0.9, hi=1.0) for d in DETS}


def _identity_king(ctx):
    return KingRun("baseline", "owner", ctx.input_ids, [1.0] * N, {d: [0.97] * N for d in DETS})


def _billing(tmp_path, credit=30000):
    svc = BillingService(PriceCard.load("configs/price_card.json"), Accounts(tmp_path / "acc.sqlite"))
    if credit:
        svc.accounts.record_payment("pi_1", "hk1", credit, 0)
    return svc


def _outputs(ctx, broken=(), odd=()):
    outs = {}
    for k, it in enumerate(ctx.items):
        tag = " BROKEN" if k in broken else " ODD" if k in odd else ""
        outs[it["id"]] = f"A human wrote this about topic {k}{tag}."
    return outs


def _eval(tmp_path, runner, *, billing=None, gates=FakeGates, dets=None, king="identity", cfg=None, sid="s1"):
    ctx = _ctx()
    store = SubmissionStore(tmp_path / "subs", KEY)
    k = _identity_king(ctx) if king == "identity" else king
    res = evaluate_submission(
        Submission(sid, "hk1"),
        ctx,
        runner=runner,
        llm=FakeLLM(),
        gate_factory=gates,
        detectors=dets or _detectors(),
        store=store,
        king=k,
        billing=billing,
        king_state_path=tmp_path / "king_state.json",
        king_run_path=tmp_path / "king_run.enc",
        cfg=cfg or EvalConfig(workers=4, crown_kw={"n_boot": 400}, retry_wait_s=0),
    )
    return res, store, ctx


def test_good_challenger_is_crowned_and_billed_for_what_was_used(tmp_path):
    ctx = _ctx()
    bill = _billing(tmp_path)
    dets = _detectors()
    runner = ReplayRunner(_outputs(ctx, broken={0, 1, 2}, odd={3}), pod_seconds=900)
    res, store, _ = _eval(tmp_path, runner, billing=bill, dets=dets)
    assert res.status == "done" and res.crowned
    assert res.private["levels_count"] == {"1.0": 36, "0.75": 0, "0.5": 1, "0.0": 3}
    # level-0 outputs are not sent to detectors (they count as caught anyway)
    assert all(d.calls == N - 3 for d in dets.values())
    assert all(math.isinf(res.private["scores"][d][0] or math.inf) for d in DETS)
    items = {i["item"]: i for i in res.statement["items"]}
    assert items["detector:pangram"]["quantity"] == sum(
        len(_outputs(ctx)[f"in{k:02d}"].split()) + (1 if k == 3 else 0) for k in range(3, N)
    )
    assert items[f"judge:{JUDGE}"]["quantity"] == (N - 0) * 120  # every non-empty output was judged once
    assert items["lium:h100"]["quantity"] == 0.25
    assert res.settlement["charged_cents"] == res.statement["total_cents"] <= res.settlement["hold_cents"]
    assert bill.accounts.hold("s1").status == "settled"
    # the king: state and run for the rest of the period
    state = KingState.from_json((tmp_path / "king_state.json").read_text())
    assert [e.submission_id for e in state.history] == ["baseline", "s1"]
    assert state.payout_weights() == {"hk1": 1.0}
    kr = KingRun.load(tmp_path / "king_run.enc", KEY)
    assert kr.submission_id == "s1" and kr.quality[0] == 0.0 and math.isinf(kr.scores["gptzero"][0])
    # public status: aggregates only
    pub = store.status("s1")
    assert pub["stage"] == "done" and pub["crowned"] is True and pub["S"] > pub["S_king"]
    assert set(pub) <= {
        "submission_id",
        "hotkey",
        "stage",
        "updated_at",
        "period_id",
        "S",
        "crowned",
        "bundle_ran",
        "S_king",
        "delta",
        "lower_bound",
        "alpha",
        "charged_cents",
        "levels",
        "gates",
        "judge_errors",
        "detectors",
    }
    assert pub["levels"] == {"1": 36, "0.75": 0, "0.5": 1, "0": 3}
    assert pub["gates"]["G3"] == {"1": 36, "0.75": 0, "0.5": 1, "0": 3} and pub["gates"]["G1"] == {
        "1": 36,
        "0.75": 0,
        "0.5": 0,
        "0": 0,
    }
    assert set(pub["detectors"]) == set(DETS) and pub["judge_errors"] == 0


def test_same_bundle_again_costs_only_what_is_new(tmp_path):
    ctx = _ctx()
    bill = _billing(tmp_path)

    class CachingFake(FakeDetector):
        def __init__(self, name):
            super().__init__(name, rules=[(lambda t: "human" in t, 0.05)], lo=0.9, hi=1.0)
            self.seen = set()

        def score(self, texts):
            out = super().score(texts)
            for t, r in zip(texts, out, strict=True):
                r.cached = t in self.seen
                self.seen.add(t)
            return out

    dets = {d: CachingFake(d) for d in DETS}
    outs = _outputs(ctx)
    r1, *_ = _eval(tmp_path, ReplayRunner(outs), billing=bill, dets=dets, sid="a1", king=None)
    r2, *_ = _eval(tmp_path, ReplayRunner(outs), billing=bill, dets=dets, sid="a2", king=None)
    words = {i["item"]: i["quantity"] for i in r2.statement["items"]}
    assert words["detector:gptzero"] == 0 and r2.public["S"] == r1.public["S"]


def test_insufficient_balance_stops_before_the_pod(tmp_path):
    class Boom:
        def run(self, *a, **k):
            raise AssertionError("must not run without a hold")

    with pytest.raises(InsufficientBalance) as e:
        _eval(tmp_path, Boom(), billing=_billing(tmp_path, credit=0))
    st = SubmissionStore(tmp_path / "subs", KEY).status("s1")
    assert st["stage"] == "awaiting_payment" and st["amount_due_cents"] == e.value.amount_due_cents > 0


def test_an_infra_failure_pauses_the_same_submission_with_its_hold_kept(tmp_path):
    """Our failure (no pod, a lost pod) never ends a submission: the same one, the same pinned commit, is
    run again later; the miner does not resubmit, so its first commitment stands (owner decision)."""
    from humanizer_eval.evaluation import EvaluationPaused

    class Down:
        def run(self, sub, items, seed, workdir):
            return RunOutput("infra_error", {}, 120.0, "rent failed")

    bill = _billing(tmp_path)
    with pytest.raises(EvaluationPaused, match="our side: rent failed"):
        _eval(tmp_path, Down(), billing=bill)
    assert bill.accounts.hold("s1").status == "open"
    assert SubmissionStore(tmp_path / "subs", KEY).status("s1")["stage"] == "paused"


def test_judges_down_pause_then_resume_without_a_second_run(tmp_path):
    ctx = _ctx()
    bill = _billing(tmp_path)
    calls = []

    class Once(ReplayRunner):
        def run(self, *a, **k):
            calls.append(1)
            return super().run(*a, **k)

    runner = Once(_outputs(ctx), pod_seconds=600)
    with pytest.raises(EvaluationPaused):
        _eval(tmp_path, runner, billing=bill, gates=lambda llm: FakeGates(llm, broken=True))
    assert bill.accounts.hold("s1").status == "open"
    assert SubmissionStore(tmp_path / "subs", KEY).status("s1")["stage"] == "paused"
    res, *_ = _eval(tmp_path, runner, billing=bill)
    assert res.status == "done" and len(calls) == 1
    assert {i["item"]: i["quantity"] for i in res.statement["items"]}["lium:h100"] == pytest.approx(
        1 / 6, abs=1e-4
    )


def test_detector_budget_hit_pauses(tmp_path):
    class Broke(FakeDetector):
        def score(self, texts):
            return [DetectorScore(self.name, text_hash(t), float("nan"), error="budget: cap") for t in texts]

    dets = {**_detectors(), "zerogpt": Broke("zerogpt")}
    with pytest.raises(EvaluationPaused) as e:
        _eval(tmp_path, ReplayRunner(_outputs(_ctx())), billing=_billing(tmp_path), dets=dets)
    assert e.value.stage == "detecting"


def test_crashed_bundle_scores_zero_charges_pod_time_and_leaks_nothing(tmp_path):
    secret = "Input number 7 has a few words in it about topic 7."

    class Crash:
        def run(self, sub, items, seed, workdir):
            return RunOutput(
                "miner_error", {}, 300.0, "entrypoint failed", meta={"errors": [f"ValueError: {secret}"]}
            )

    bill = _billing(tmp_path)
    res, store, _ = _eval(tmp_path, Crash(), billing=bill)
    assert res.status == "done" and res.public["S"] == 0.0 and not res.crowned
    items = {i["item"]: i for i in res.statement["items"]}
    assert items["detector:gptzero"]["quantity"] == 0 and not any(k.startswith("judge:") for k in items)
    assert items["lium:h100"]["amount_cents"] > 0
    visible = (store.dir("s1") / "status.json").read_text() + (store.dir("s1") / "statement.json").read_text()
    assert secret not in visible and "ValueError" not in visible
    assert store.status("s1")["bundle_error"] == "entrypoint failed"
    assert secret not in (store.dir("s1") / "record.enc").read_bytes().decode("latin-1")  # encrypted at rest


def test_owner_run_without_billing_and_without_king(tmp_path):
    ctx = _ctx()
    res, _, _ = _eval(tmp_path, ReplayRunner([it["text"] for it in ctx.items]), king=None, sid="baseline")
    assert res.status == "done" and res.statement is None and not res.crowned
    assert "S_king" not in res.public


def test_detector_set_must_match_the_period(tmp_path):
    with pytest.raises(ValueError, match="differ"):
        _eval(tmp_path, ReplayRunner(_outputs(_ctx())), dets={"gptzero": FakeDetector("gptzero")})


def test_bundle_url_rules():
    with pytest.raises(ValueError):
        fetch_script("http://example.com/b.tar", "a" * 64)
    with pytest.raises(ValueError):
        fetch_script("https://example.com/b.tar", "A" * 64)
    with pytest.raises(ValueError):
        Submission("s1", "hk", bundle_url="https://example.com/b.tar")
    with pytest.raises(ValueError):
        Submission("../x", "hk")
    s = fetch_script("https://example.com/b.tar; rm -rf /", "a" * 64)
    assert "'https://example.com/b.tar; rm -rf /'" in s  # quoted, not executed
    assert "--proto =https --proto-redir =https" in s  # a redirect cannot switch to http/file/ftp


def test_pod_results_are_copied_safely():
    s = pod_script(seed=1, time_limit_s=60)
    assert "[ ! -L /work/out/$n ]" in s and "head -c $cap" in s
    assert 'echo "$info" > /work/pod.json' in s
    json.dumps(s)


def test_lium_runner_moves_on_after_a_refused_rent(tmp_path, monkeypatch):
    from humanizer_eval.evaluation.runners import LiumRunner, rank_executors
    from humanizer_eval.runner import evaluate
    from humanizer_eval.runner.evaluate import PodJobResult

    exs = [
        {
            "id": "big",
            "machine_name": "NVIDIA H100 80GB HBM3",
            "price_per_gpu": 1.0,
            "gpu_count": 8,
            "available_gpu_count": 8,
            "tier": "secure",
            "reliability_score": 99,
        },
        {
            "id": "one",
            "machine_name": "NVIDIA H100 80GB HBM3",
            "price_per_gpu": 2.0,
            "gpu_count": 1,
            "available_gpu_count": 1,
            "tier": "secure",
            "reliability_score": 99,
        },
        {
            "id": "flaky",
            "machine_name": "NVIDIA H100 80GB HBM3",
            "price_per_gpu": 0.5,
            "gpu_count": 1,
            "available_gpu_count": 1,
            "tier": "secure",
            "reliability_score": 99,
            "frequent_rent_failures": True,
        },
    ]
    assert [e["id"] for e in rank_executors(exs, 2.6)] == ["one", "big"]  # single-GPU first, flaky out
    # evaluations: exactly one GPU model, never the slower PCIe variant (identical conditions)
    from humanizer_eval.evaluation.runners import EVAL_MACHINE

    variants = [{**exs[0], "id": "sxm", "machine_name": EVAL_MACHINE},
                {**exs[0], "id": "pcie", "machine_name": "NVIDIA H100 PCIe"}]  # fmt: skip
    assert [e["id"] for e in rank_executors(variants, 2.6, exact=(EVAL_MACHINE,))] == ["sxm"]
    seen = []

    def fake_job(items, *, executor, **kw):
        seen.append(executor["id"])
        if executor["id"] == "one":
            return PodJobResult("infra_error", "rent failed: taken", {}, {}, {}, None, 1.0, tmp_path)
        return PodJobResult("ok", "", {"a": "x"}, {}, {"exit": 0}, 0.1, 600.0, tmp_path)

    monkeypatch.setattr(evaluate, "run_pod_job", fake_job)

    class Client:
        def executors(self):
            return exs

    out = LiumRunner(Client(), tmp_path / "k").run(Submission("s1", "hk"), [], 0, tmp_path)
    assert seen == ["one", "big"] and out.status == "ok" and out.pod_seconds == 600.0
    assert [a["status"] for a in out.meta["attempts"]] == ["infra_error", "ok"]


# --- identical conditions for king and challenger (owner's requirement 2026-11-21) -----------------
class FlakyGates(FakeGates):
    """The judge fails on output ``bad`` the first ``fails`` times it is asked."""

    def __init__(self, llm, bad=5, fails=1, reason="judge timeout"):
        super().__init__(llm)
        self.bad, self.left, self.reason = bad, fails, reason

    def check(self, source, output, facts):
        if f"topic {self.bad}." in output and self.left > 0:
            self.left -= 1
            return GateResult("error", reasons=[self.reason])
        return super().check(source, output, facts)


def test_a_judge_failure_is_retried_never_scored_as_level_zero(tmp_path):
    ctx = _ctx()
    left = {"n": 2}  # the judge fails twice on output 5, whichever gate view asks (first pass or retries)

    class Shared(FlakyGates):
        def check(self, source, output, facts):
            if "topic 5." in output and left["n"] > 0:
                left["n"] -= 1
                return GateResult("error", reasons=["judge timeout"])
            return FakeGates.check(self, source, output, facts)

    res, _, _ = _eval(tmp_path, ReplayRunner(_outputs(ctx)), gates=lambda llm: Shared(llm))
    assert res.status == "done" and res.private["levels"][5] == 1.0  # retried, judged like every other


def test_one_output_without_a_verdict_pauses_never_gets_a_level(tmp_path):
    """'No answer' cannot tell a provider outage (ours) from an unusable reply: never a level 0 for it; the
    run pauses (and is parked for the owner past the pauses)."""
    from humanizer_eval.evaluation import EvaluationPaused

    ctx = _ctx()
    gates = lambda llm: FlakyGates(llm, fails=99, reason="no answer from a/judge")  # noqa: E731
    with pytest.raises(EvaluationPaused, match="1 of 40 outputs could not be judged"):
        _eval(tmp_path, ReplayRunner(_outputs(ctx)), gates=gates)


def test_our_detector_account_errors_never_count_as_a_refused_text(tmp_path):
    """'403 Not enough credits' (ZeroGPT on 2026-11-26) is our account, not the miner's text."""
    from humanizer_eval.evaluation.pipeline import _refusal

    assert not _refusal("parse: DetectorError: zerogpt: code=403 message='Not enough credits'")
    assert not _refusal("gptzero HTTP 403: forbidden")
    assert not _refusal("pangram HTTP 401: invalid key")
    assert not _refusal("gptzero HTTP 429: too many requests")
    assert _refusal("gptzero HTTP 400: text not supported")


def test_judges_failing_on_many_outputs_pause_the_evaluation(tmp_path):
    from humanizer_eval.evaluation import EvaluationPaused

    with pytest.raises(EvaluationPaused, match="could not be judged"):
        _eval(tmp_path, ReplayRunner(_outputs(_ctx())), gates=lambda llm: FakeGates(llm, broken=True))


class FlakyDetector(FakeDetector):
    def __init__(self, name, bad_text, fails, **kw):
        super().__init__(name, **kw)
        self.bad_text, self.left = bad_text, fails

    def score(self, texts):
        out = super().score(texts)
        for i, t in enumerate(texts):
            if t == self.bad_text and self.left > 0:
                self.left -= 1
                out[i] = DetectorScore(self.name, text_hash(t), 0.0, error="code=500 try again")
        return out


def test_one_unscored_text_pauses_never_counts_as_caught(tmp_path):
    from humanizer_eval.evaluation import EvaluationPaused

    ctx = _ctx()
    outs = _outputs(ctx)
    bad = outs[ctx.items[7]["id"]]
    rules = [(lambda t: "human" in t, 0.05)]
    dets = {**_detectors(), "zerogpt": FlakyDetector("zerogpt", bad, 2, rules=rules, lo=0.9, hi=1.0)}
    res, _, _ = _eval(tmp_path, ReplayRunner(outs), dets=dets)
    assert res.status == "done" and res.private["scores"]["zerogpt"][7] == 0.05  # retried, really scored
    dets["zerogpt"] = FlakyDetector("zerogpt", bad, 99, rules=rules, lo=0.9, hi=1.0)
    with pytest.raises(EvaluationPaused, match="zerogpt: 1 of 40 texts failed"):
        _eval(tmp_path / "b", ReplayRunner(outs), dets=dets, sid="s2")


def test_runs_scored_under_other_conditions_are_never_compared(tmp_path):
    ctx = _ctx()
    king = _identity_king(ctx)
    king.fingerprint = "old"
    with pytest.raises(ValueError, match="other conditions"):
        _eval(tmp_path, ReplayRunner(_outputs(ctx)), king=king,
              cfg=EvalConfig(workers=4, crown_kw={"n_boot": 400}, retry_wait_s=0, fingerprint="new"))  # fmt: skip


def test_a_detector_model_change_pauses_with_conditions_changed(tmp_path, monkeypatch):
    from humanizer_eval.evaluation import conditions
    from humanizer_eval.evaluation.pipeline import ConditionsChanged

    ctx = _ctx()
    store = SubmissionStore(tmp_path / "subs", KEY)
    common = dict(runner=ReplayRunner(_outputs(ctx)), llm=FakeLLM(), gate_factory=FakeGates, store=store,
                  king=_identity_king(ctx), cfg=EvalConfig(workers=4, crown_kw={"n_boot": 200}, retry_wait_s=0))  # fmt: skip
    # a vendor that reports versions: H was scored by "fake-1", this run sees "fake-2"
    ctx.meta["detector_baseline"] = {"gptzero": {"versions": ["fake-1"]}}
    dets = {**_detectors(), "gptzero": FakeDetector("gptzero", lo=0.9, hi=1.0, version="fake-2")}
    with pytest.raises(ConditionsChanged, match="gptzero: model version"):
        evaluate_submission(Submission("s1", "hk1"), ctx, detectors=dets, **common)
    # a vendor without versions: its canaries moved since H was scored
    ctx.meta["detector_baseline"] = {"zerogpt": {"versions": [], "canary": {"texts": ["t"], "scores": [0.5]}}}
    monkeypatch.setattr(conditions, "_score_texts", lambda det, texts: [0.51])
    with pytest.raises(ConditionsChanged, match="canary scores moved"):
        evaluate_submission(
            Submission("s2", "hk1"), ctx, detectors=_detectors(), watch={"zerogpt": object()}, **common
        )
    monkeypatch.setattr(conditions, "_score_texts", lambda det, texts: [0.5])  # unchanged: evaluated normally
    res = evaluate_submission(
        Submission("s3", "hk1"), ctx, detectors=_detectors(), watch={"zerogpt": object()}, **common
    )
    assert res.status == "done"


def test_fingerprint_covers_code_config_and_detectors():
    from humanizer_eval.evaluation.conditions import scoring_fingerprint

    cfg = {"judges": ["openai/a", "google/b"], "g3": {"minor_base": 3}}
    dets = _detectors()
    fp = scoring_fingerprint(cfg, dets, {"n_boot": 2000})
    assert fp == scoring_fingerprint(dict(cfg), dict(dets), {"n_boot": 2000})
    assert fp != scoring_fingerprint({**cfg, "g3": {"minor_base": 4}}, dets, {"n_boot": 2000})
    assert fp != scoring_fingerprint(cfg, dets, {"n_boot": 1000})
    assert fp != scoring_fingerprint(cfg, {k: v for k, v in dets.items() if k != "zerogpt"}, {"n_boot": 2000})


def test_canary_tolerance_absorbs_gptzeros_two_backends_only(monkeypatch):
    from humanizer_eval.evaluation import conditions

    base = {"gptzero": {"versions": [], "canary": {"texts": ["t"], "scores": [0.3892]}},
            "zerogpt": {"versions": [], "canary": {"texts": ["t"], "scores": [0.3892]}}}  # fmt: skip
    monkeypatch.setattr(conditions, "_score_texts", lambda det, texts: [0.3964])  # GPTZero's other value
    problems = conditions.check(base, {"gptzero": object(), "zerogpt": object()})
    assert problems == [p for p in problems if p.startswith("zerogpt")] and len(problems) == 1
    monkeypatch.setattr(conditions, "_score_texts", lambda det, texts: [0.45])  # a real model change
    assert len(conditions.check(base, {"gptzero": object()})) == 1


def test_an_output_at_level_075_is_scored_not_a_crash(tmp_path):
    """G3 gives 0.75 for a few minor slips: the metric must take it (it once raised and every run crashed)."""
    ctx = _ctx()
    outs = _outputs(ctx)
    first = ctx.items[0]["id"]
    outs[first] = outs[first] + " SLIP"
    res, _, _ = _eval(tmp_path, ReplayRunner(outs))
    assert res.status == "done"
    assert res.public["levels"]["0.75"] == 1


def test_work_paid_before_a_pause_is_charged_once_at_the_end(tmp_path):
    """A pause after the judges and two detectors finished used to make all of that free: on the resume
    it came back as cache hits (0 tokens, 0 words). The statement must equal a run that never paused."""

    class CachingLLM:  # the production CachedLLM: a hit costs 0 tokens
        def __init__(self):
            self.seen = set()

        def chat(self, model, messages, **kw):
            key = (model, json.dumps(messages))
            hit = key in self.seen
            self.seen.add(key)
            return ChatResult("{}", model, "fake", 0 if hit else 100, 0 if hit else 20, 0.0, None, json={})

    class CachingDet(FakeDetector):
        def __init__(self, name):
            super().__init__(name, rules=[(lambda t: "human" in t, 0.05)], lo=0.9, hi=1.0)
            self.seen, self.down = set(), False

        def score(self, texts):
            out = super().score(texts)
            for i, t in enumerate(texts):
                if self.down and t not in self.seen:
                    out[i] = DetectorScore(self.name, text_hash(t), 0.0, error="code=500 try again")
                    continue
                out[i].cached = t in self.seen
                self.seen.add(t)
            return out

    def run(root, pause):
        ctx = _ctx()
        bill = _billing(root)
        dets = {d: CachingDet(d) for d in DETS}
        llm = CachingLLM()
        store = SubmissionStore(root / "subs", KEY)
        kw = dict(runner=ReplayRunner(_outputs(ctx), pod_seconds=600), llm=llm, gate_factory=FakeGates,
                  detectors=dets, store=store, king=None, billing=bill, king_state_path=root / "ks.json",
                  king_run_path=root / "kr.enc", cfg=EvalConfig(workers=4, retry_wait_s=0))  # fmt: skip
        if pause:
            dets["zerogpt"].down = True
            with pytest.raises(EvaluationPaused):
                evaluate_submission(Submission("s1", "hk1"), ctx, **kw)
            dets["zerogpt"].down = False
        res = evaluate_submission(Submission("s1", "hk1"), ctx, **kw)
        return {i["item"]: i["quantity"] for i in res.statement["items"]}

    clean, resumed = run(tmp_path / "a", False), run(tmp_path / "b", True)
    assert resumed == clean and any("pangram" in k for k in clean)


def test_a_cached_unusable_judge_reply_is_asked_again_on_retry(tmp_path):
    """A reply that parsed but was unusable was cached and returned on every retry: a permanent judge
    error and an abandoned run (audit 3). Retries bypass the cache."""
    from humanizer_eval.llm.cache import CachedLLM

    class Flaky:
        n = 0

        def chat(self, model, messages, **kw):
            Flaky.n += 1
            body = {"bad": True} if Flaky.n == 1 else {"ok": True}
            return ChatResult(json.dumps(body), model, "fake", 10, 5, 0.0, None, json=body)

    class Picky(FakeGates):
        def check(self, source, output, facts):
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(1) as ex:  # judges ask from their own threads (G1 does)
                r = ex.submit(
                    self.llm.chat,
                    JUDGE,
                    [{"role": "user", "content": output}],
                    json_schema={"type": "object"},
                ).result()
            if r.json.get("bad"):
                return GateResult("error", reasons=["unusable reply"])
            return GateResult("pass", level=1.0, levels={"G1": 1.0, "G2": 1.0, "G3": 1.0})

    llm = CachedLLM(Flaky(), tmp_path / "c.sqlite", cache_all=True)
    ctx = _ctx()
    from humanizer_eval.evaluation.pipeline import _gate_all
    from humanizer_eval.llm.cache import Refresh

    res = _gate_all(Picky(llm), ctx, [ctx.items[0]["text"] + " x"], EvalConfig(retry_wait_s=0),
                    retry_gates=Picky(Refresh(llm)), final_gates=Picky(Refresh(llm, strict=True)))  # fmt: skip
    assert res[0].status == "pass"


def test_a_text_the_detector_refuses_is_caught_not_a_free_run(tmp_path):
    """One output a vendor answers with HTTP 400 every time paused the whole run until it was given up
    for free (audit 5); a refusal of that very text now counts it as caught."""
    ctx = _ctx()
    outs = _outputs(ctx)
    bad = outs[ctx.items[3]["id"]]

    class Refuses(FakeDetector):
        def score(self, texts):
            out = super().score(texts)
            return [DetectorScore(self.name, text_hash(t), 0.0, error="gptzero HTTP 400: text not supported")
                    if t == bad else r for t, r in zip(texts, out, strict=True)]  # fmt: skip

    dets = {
        **_detectors(),
        "gptzero": Refuses("gptzero", rules=[(lambda t: "human" in t, 0.05)], lo=0.9, hi=1.0),
    }
    res, _, _ = _eval(tmp_path, ReplayRunner(outs), dets=dets)
    v = res.private["scores"]["gptzero"][3]
    assert res.status == "done" and (v is None or math.isinf(v))  # caught (inf is stored as None)


def test_our_own_judge_failures_never_become_level_0(tmp_path):
    """A rate limit or our budget cap on a few outputs is ours: the run pauses (audit 5)."""
    from humanizer_eval.evaluation import EvaluationPaused

    class Limited(FakeGates):
        def check(self, source, output, facts):
            if "topic 1." in output:
                return GateResult("error", reasons=["judge error: openrouter HTTP 429 rate limit"])
            return super().check(source, output, facts)

    with pytest.raises(EvaluationPaused):
        _eval(tmp_path, ReplayRunner(_outputs(_ctx())), gates=Limited)


def test_rank_skips_nodes_that_refuse_a_single_gpu_rent():
    from humanizer_eval.evaluation.runners import EVAL_MACHINE, rank_executors

    base = {"machine_name": EVAL_MACHINE, "tier": "secure", "price_per_gpu": 3.5, "reliability_score": 99,
            "available_gpu_count": 8}  # fmt: skip
    whole = {**base, "id": "whole", "gpu_count": 8, "ncu_profiling_enabled": True}
    minimum = {**base, "id": "min", "gpu_count": 8, "min_gpu_count_for_rental": 6}
    single_ncu = {**base, "id": "single-ncu", "gpu_count": 1, "ncu_profiling_enabled": True}
    shared = {
        **base,
        "id": "shared",
        "gpu_count": 8,
        "ncu_profiling_enabled": False,
        "min_gpu_count_for_rental": 1,
    }
    got = [e["id"] for e in rank_executors([whole, minimum, single_ncu, shared], 4.0, exact=(EVAL_MACHINE,))]
    assert got == ["single-ncu", "shared"]


def test_rank_prefers_a_single_h100_hbm3_and_falls_back_to_h200():
    from humanizer_eval.evaluation.runners import EVAL_MACHINES, rank_executors

    base = {"tier": "secure", "reliability_score": 99, "available_gpu_count": 1, "gpu_count": 1}
    h100 = {**base, "id": "h100", "machine_name": "NVIDIA H100 80GB HBM3", "price_per_gpu": 4.4}
    h200 = {**base, "id": "h200", "machine_name": "NVIDIA H200", "price_per_gpu": 4.2}
    pcie = {**base, "id": "pcie", "machine_name": "NVIDIA H100 PCIe", "price_per_gpu": 2.2}
    whole = {**h100, "id": "whole", "gpu_count": 8, "ncu_profiling_enabled": True, "price_per_gpu": 3.5}
    pricey = {**h200, "id": "pricey", "price_per_gpu": 5.2}
    assert [e["id"] for e in rank_executors([pcie, h200, whole, h100, pricey], 4.5, exact=EVAL_MACHINES)] == [
        "h100", "h200"]  # fmt: skip
    assert [e["id"] for e in rank_executors([pcie, h200, whole], 4.5, exact=EVAL_MACHINES)] == ["h200"]
