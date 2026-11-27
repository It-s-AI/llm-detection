"""No failure on our side can make the server spend in a loop (the night of 2026-11-25 a king re-run loop
burned ~$1531). Each scenario runs the worker for two simulated weeks, one step every 10 minutes, and
bounds every paid call: pods, detector calls, judge calls."""

from dataclasses import replace

from evaluation.test_pipeline import FakeGates, FakeLLM
from humanizer_eval.detectors.fake import FakeDetector
from humanizer_eval.evaluation import RunOutput
from humanizer_eval.intake.client import SeedSigner
from humanizer_eval.intake.worker import KING_MAX_FAILED_RUNS
from intake.test_intake import URL, CountingReplay, World, _sha

STEPS = 14 * 24 * 6  # two weeks, every 10 minutes


def _run(w, steps=STEPS):
    for _ in range(steps):
        w.worker.run_once()
        w.t[0] += 600


def _challenger(w, seed=6, url=URL):
    m = SeedSigner(bytes([seed]) * 32).ss58_address
    w.pay(m)
    return w.svc.submit(m, bundle_url=url, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]


class CountingLLM(FakeLLM):
    calls = 0

    def chat(self, *a, **kw):
        CountingLLM.calls += 1
        return super().chat(*a, **kw)


def test_detector_model_change_holds_everything_without_paid_calls(tmp_path):
    """Last night's case: the canaries say a detector's model changed. The period is held; nothing is
    re-run or re-scored until the owner rescores, however long it waits."""
    w = World(tmp_path)
    base = {"zerogpt": {"versions": [], "canary": {"texts": ["a canary human text"], "scores": [0.123]}}}
    (tmp_path / "pc").mkdir()
    w.ctx = replace(w.ctx, dir=tmp_path / "pc", meta={**w.ctx.meta, "detector_baseline": base})
    watch = FakeDetector("zerogpt", lo=0.5, hi=0.9)  # the canary moved
    w.env.detector_watch = {"zerogpt": watch}
    CountingReplay.calls = 0
    for seed in (6, 7, 8):
        _challenger(w, seed)
    before = {d: det.calls for d, det in w.dets.items()}
    _run(w)
    assert watch.calls <= 2, watch.calls
    assert CountingReplay.calls <= 1, CountingReplay.calls  # at most the first bundle ran
    assert all(w.dets[d].calls - before[d] <= 1 for d in w.dets), {d: w.dets[d].calls for d in w.dets}


class InfraRunner:
    pods = 0

    def run(self, sub, items, seed, workdir):
        InfraRunner.pods += 1
        return RunOutput("infra_error", {}, 600.0, "lost the pod", cost_usd=4.5)


def test_a_king_rerun_failing_on_our_side_rents_a_bounded_number_of_pods(tmp_path):
    w = World(tmp_path)
    _challenger(w, 5)
    assert w.worker.run_once()["crowned"]  # a miner king
    base = w.env.runner_for
    w.env.runner_for = lambda sub: InfraRunner() if sub.submission_id.endswith("-kingrun") else base(sub)
    w.ctx = replace(w.ctx, period_id="p-next")  # his re-run on the new period always fails (ours)
    InfraRunner.pods = 0
    url2 = "https://huggingface.co/ch/b/resolve/main/b.tar"
    w.bundles[url2] = w.bundles[URL]
    _challenger(w, 6, url2)
    _run(w)
    assert InfraRunner.pods <= KING_MAX_FAILED_RUNS, InfraRunner.pods


class DownDetector(FakeDetector):
    def score(self, texts):
        self.calls += len(texts)
        raise RuntimeError("HTTP 503 from the vendor")


def test_a_detector_outage_pauses_then_parks_with_a_bounded_number_of_calls(tmp_path):
    w = World(tmp_path)
    down = DownDetector("zerogpt", lo=0.9, hi=1.0)
    w.dets["zerogpt"] = down
    w.env.detectors = w.dets
    CountingReplay.calls = 0
    sid = _challenger(w)
    _run(w)
    row = w.queue.get(sid)
    assert CountingReplay.calls == 1  # the bundle ran once; the retries reuse its outputs
    assert '"parked"' in (row.note or "") and w.billing.accounts.hold(sid).status == "open"
    n = len(w.ctx.items)
    assert down.calls <= (w.worker.max_pauses + 2) * n * 3, down.calls
    calls = down.calls
    _run(w)  # parked: nothing more until the owner decides
    assert down.calls == calls


def test_a_judge_outage_pauses_then_parks_with_a_bounded_number_of_calls(tmp_path):
    w = World(tmp_path)
    CountingLLM.calls = 0
    w.env.llm = CountingLLM()
    w.env.gate_factory = lambda llm: FakeGates(llm, broken=True)
    CountingReplay.calls = 0
    sid = _challenger(w)
    _run(w)
    row = w.queue.get(sid)
    assert CountingReplay.calls == 1
    assert '"parked"' in (row.note or "")
    n = len(w.ctx.items)
    # each attempt: one call per output plus up to 3 uncached retry rounds of the failed ones
    assert CountingLLM.calls <= (w.worker.max_pauses + 1) * n * 4, CountingLLM.calls
    calls = CountingLLM.calls
    _run(w)
    assert CountingLLM.calls == calls
