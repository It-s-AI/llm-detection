"""Replay real vendor responses recorded by scripts/live_detector_check.py (no network, no cost)."""

from pathlib import Path

import pytest

from humanizer_eval.budget import SpendLedger
from humanizer_eval.detectors.fake import ReplayTransport
from humanizer_eval.detectors.http import HttpClient
from humanizer_eval.detectors.vendors import GPTZero, ItsAI, Pangram, ZeroGPT

REC = Path(__file__).parents[1] / "fixtures" / "recordings"


def _texts():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "live_check", Path(__file__).parents[2] / "scripts" / "live_detector_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AI_TEXTS


def _http(name, base=""):
    path = REC / f"{name}.jsonl"
    if not path.exists():
        pytest.skip(f"no recording for {name}")
    return HttpClient(base_url=base, transport=ReplayTransport(path), rate_per_sec=1000, burst=1000)


def _ledger(tmp_path):
    return SpendLedger(tmp_path / "l.sqlite", {s: 100 for s in ("gptzero", "zerogpt", "pangram", "its_ai")})


@pytest.mark.parametrize(
    "factory",
    [
        lambda led: GPTZero(_http("gptzero"), ledger=led, cost_fn=lambda _w: 0.0),
        lambda led: ZeroGPT(_http("zerogpt"), ledger=led, cost_fn=lambda _w: 0.0),
        lambda led: Pangram(
            _http("pangram", Pangram.BASE),
            ledger=led,
            cost_fn=lambda _w: 0.0,
            poll_interval=0,
            sleep=lambda _s: None,
        ),
        lambda led: ItsAI(_http("its_ai"), url="http://replay/predict", api_key="", ledger=led),
    ],
)
def test_real_responses_parse(tmp_path, factory):
    det = factory(_ledger(tmp_path))
    det.max_workers = 1
    out = det.score(_texts())
    assert all(s.ok for s in out), [s.error for s in out if not s.ok][:3]
    # the recorded texts are LLM-written: every detector should be confident they are AI
    assert sum(s.score for s in out) / len(out) > 0.8
