import math

import httpx
import pytest

from humanizer_eval.budget import SpendLedger
from humanizer_eval.detectors.cache import DetectorCache
from humanizer_eval.detectors.canary import run_canaries
from humanizer_eval.detectors.fake import (
    FakeDetector,
    RecordingTransport,
    ReplayTransport,
    fake_vendor_transport,
)
from humanizer_eval.detectors.http import HttpClient, RetryPolicy
from humanizer_eval.detectors.vendors import (
    GPTZero,
    ItsAI,
    Pangram,
    ZeroGPT,
    gptzero_cost,
    pangram_cost,
    zerogpt_cost,
)

TEXTS = ["The quick brown fox jumps over the lazy dog. " * 10, "Another sample text with several words " * 8]
NO_SLEEP = {"sleep": lambda _s: None}


def ledger(tmp_path, cap=10.0):
    return SpendLedger(tmp_path / "led.sqlite", {s: cap for s in ("gptzero", "zerogpt", "pangram", "its_ai")})


def http(vendor, base_url="", **kw):
    return HttpClient(
        base_url=base_url,
        transport=fake_vendor_transport(vendor, **kw),
        retry=RetryPolicy(max_attempts=4, base_delay=0),
        rate_per_sec=1000,
        burst=1000,
        **NO_SLEEP,
    )


def test_pricing_upper_bounds():
    assert pangram_cost(1) == pytest.approx(0.05)
    assert pangram_cost(230) == pytest.approx(0.15)  # 3 units, conservative ceil
    assert gptzero_cost(1_000_000) == pytest.approx(150)
    assert zerogpt_cost(1000) == pytest.approx(0.069)


@pytest.mark.parametrize("cls,vendor", [(GPTZero, "gptzero"), (ZeroGPT, "zerogpt")])
def test_sync_vendors_parse_and_meter(tmp_path, cls, vendor):
    led = ledger(tmp_path)
    cost = {"gptzero": gptzero_cost, "zerogpt": zerogpt_cost}[vendor]
    det = cls(http(vendor), ledger=led, cost_fn=cost)
    out = det.score(TEXTS)
    assert all(s.ok for s in out), [s.error for s in out]
    assert all(0 <= s.score <= 1 for s in out)
    assert led.spent(vendor) == pytest.approx(sum(cost(len(t.split())) for t in TEXTS))


def test_gptzero_primary_is_ai_plus_mixed(tmp_path):
    det = GPTZero(http("gptzero"), ledger=ledger(tmp_path), cost_fn=gptzero_cost)
    s = det.score(TEXTS[:1])[0]
    assert s.score == pytest.approx(s.candidates["p_ai"] + s.candidates["p_mixed"])
    assert s.version == "fake-gptzero-1"


def test_zerogpt_uses_fake_percentage_not_is_human(tmp_path):
    det = ZeroGPT(http("zerogpt"), ledger=ledger(tmp_path), cost_fn=zerogpt_cost)
    s = det.score(TEXTS[:1])[0]
    assert s.score == pytest.approx(s.candidates["fake_percentage"])
    assert s.candidates["is_human_raw"] == 0  # kept only as raw info; never used as the score


def test_pangram_async_task_polling(tmp_path):
    det = Pangram(
        http("pangram", base_url=Pangram.BASE),
        ledger=ledger(tmp_path),
        cost_fn=pangram_cost,
        poll_interval=0,
        sleep=lambda _s: None,
    )
    out = det.score(TEXTS)
    assert all(s.ok for s in out), [s.error for s in out]
    s = out[0]
    assert s.version == "4.1"
    assert s.score == pytest.approx(
        max(s.candidates["weighted_window_ai"], s.candidates["weighted_humanizer"])
    )
    assert det.config == "task-pangram-4.1"


def test_its_ai_batches_and_is_free(tmp_path):
    led = ledger(tmp_path)
    det = ItsAI(http("its_ai"), url="http://fake/predict", api_key="k", ledger=led)
    out = det.score(TEXTS * 20)
    assert len(out) == 40 and all(s.ok for s in out)
    assert led.spent("its_ai") == 0


def test_retries_on_429_then_succeeds(tmp_path):
    det = GPTZero(http("gptzero", fail_first=2), ledger=ledger(tmp_path), cost_fn=gptzero_cost)
    det.max_workers = 1
    assert det.score(TEXTS[:1])[0].ok


def test_persistent_5xx_reports_error_and_books_nothing(tmp_path):
    """The vendor answered every send with an error status: nothing was billed, the cap is untouched (an
    outage must not use up the cap)."""
    led = ledger(tmp_path)
    det = GPTZero(http("gptzero", fail_first=100, fail_status=503), ledger=led, cost_fn=gptzero_cost)
    det.max_workers = 1
    s = det.score(TEXTS[:1])[0]
    assert not s.ok and math.isnan(s.score)
    assert led.spent("gptzero") == 0 and led.outstanding("gptzero") == 0


def test_sends_delivered_with_an_unknown_outcome_are_each_booked(tmp_path):
    """A send that went out and timed out may have been billed: each one is booked at the estimate, and a
    later success adds its own (the cap counts every delivered request)."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise httpx.ReadTimeout("read timed out", request=request)
        return httpx.Response(200, json={"documents": [{"class_probabilities": {"ai": 1, "human": 0, "mixed": 0},
                                                         "completely_generated_prob": 1}]})  # fmt: skip

    led = ledger(tmp_path)
    client = HttpClient("", transport=httpx.MockTransport(handler),
                        retry=RetryPolicy(max_attempts=4), sleep=lambda _s: None)  # fmt: skip
    det = GPTZero(client, ledger=led, cost_fn=gptzero_cost)
    det.max_workers = 1
    s = det.score(TEXTS[:1])[0]
    one = gptzero_cost(len(TEXTS[0].split()))
    assert calls["n"] == 3 and led.spent("gptzero") == pytest.approx(3 * one), (s, led.spent("gptzero"))


def test_malformed_response_is_an_error(tmp_path):
    det = ZeroGPT(http("zerogpt", malformed=True), ledger=ledger(tmp_path), cost_fn=zerogpt_cost)
    s = det.score(TEXTS[:1])[0]
    assert not s.ok


def test_budget_cap_blocks_requests(tmp_path):
    led = ledger(tmp_path, cap=0.05)
    det = Pangram(
        http("pangram", base_url=Pangram.BASE),
        ledger=led,
        cost_fn=pangram_cost,
        poll_interval=0,
        sleep=lambda _s: None,
    )
    det.max_workers = 1
    out = det.score(TEXTS)  # each text costs >= $0.05 -> second one must be refused
    assert out[0].ok
    assert not out[1].ok and "budget" in out[1].error


def test_paid_detector_without_ledger_refuses():
    det = GPTZero(http("gptzero"), ledger=None, cost_fn=gptzero_cost)
    s = det.score(TEXTS[:1])[0]
    assert not s.ok and "ledger" in s.error


def test_cache_hit_skips_vendor_and_budget(tmp_path):
    led = ledger(tmp_path)
    cache = DetectorCache(tmp_path / "c.sqlite")
    det = ZeroGPT(http("zerogpt"), ledger=led, cache=cache, cost_fn=zerogpt_cost)
    first = det.score(TEXTS)
    spent = led.spent("zerogpt")
    second = det.score(TEXTS)
    assert [s.score for s in first] == [s.score for s in second]
    assert all(s.cached for s in second)
    assert led.spent("zerogpt") == pytest.approx(spent)


def test_record_and_replay_strips_secrets(tmp_path):
    rec_path = tmp_path / "rec.jsonl"
    rec = RecordingTransport(rec_path, inner=fake_vendor_transport("gptzero"))
    client = HttpClient(headers={"x-api-key": "SECRET123"}, transport=rec, rate_per_sec=1000, burst=1000)
    live = GPTZero(client, ledger=ledger(tmp_path), cost_fn=gptzero_cost).score(TEXTS)
    assert "SECRET123" not in rec_path.read_text()
    replay = HttpClient(transport=ReplayTransport(rec_path), rate_per_sec=1000, burst=1000)
    again = GPTZero(replay, ledger=ledger(tmp_path / "b"), cost_fn=gptzero_cost).score(TEXTS)
    assert [s.score for s in live] == [s.score for s in again]


def test_canaries_detect_version_change(tmp_path):
    base = tmp_path / "canary.json"
    det = FakeDetector("x", version="v1")
    assert not run_canaries(det, TEXTS, base).changed  # writes baseline
    assert not run_canaries(det, TEXTS, base).changed
    assert run_canaries(FakeDetector("x", version="v2"), TEXTS, base).changed
    assert run_canaries(FakeDetector("x", version="v1", lo=0.5, hi=0.6), TEXTS, base).changed


def test_fake_detector_rules():
    det = FakeDetector("f", rules=[(lambda t: "AI" in t, 0.99)])
    assert det.score(["AI text"])[0].score == 0.99
    assert 0 <= det.score(["human"])[0].score <= 1


def test_token_bucket_limits_rate():
    from humanizer_eval.detectors.http import TokenBucket

    t = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    b = TokenBucket(rate=2, burst=1, clock=lambda: t[0])
    for _ in range(5):
        b.acquire(sleep)
    assert t[0] == pytest.approx(2.0)  # 5 requests at 2/s with burst 1 -> 2 seconds


def test_http_client_has_no_secret_in_transport_error():
    def boom(_r):
        raise httpx.ConnectError("down")

    c = HttpClient(
        headers={"x-api-key": "SECRET"},
        transport=httpx.MockTransport(boom),
        retry=RetryPolicy(max_attempts=2, base_delay=0),
        rate_per_sec=1000,
        burst=10,
        **NO_SLEEP,
    )
    with pytest.raises(RuntimeError) as e:
        c.request("POST", "http://x")
    assert "SECRET" not in str(e.value)


def test_an_answer_without_probabilities_is_an_error_not_a_human_score(tmp_path):
    """A 200 without the probability fields used to score 0.0 (fully human) and be cached (audit 2, C2)."""
    from humanizer_eval.detectors.base import DetectorError
    from humanizer_eval.detectors.vendors import Pangram

    g = GPTZero(http("gptzero"), ledger=ledger(tmp_path), cost_fn=gptzero_cost)
    with pytest.raises(DetectorError):
        g._extract({"documents": [{"predicted_class": "ai"}]})
    p = Pangram.__new__(Pangram)
    with pytest.raises(DetectorError):
        p._extract({"windows": [{"label": "AI", "word_count": 10}]})
