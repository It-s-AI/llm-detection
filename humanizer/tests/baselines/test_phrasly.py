import json

import httpx
import pytest

from humanizer_eval.baselines.phrasly import Phrasly, phrasly_cost
from humanizer_eval.budget import BudgetExceededError, SpendLedger

TEXT = " ".join(["word"] * 100)


def _client(tmp_path, handler, cap=1.0):
    led = SpendLedger(tmp_path / "l.sqlite", {"phrasly": cap})
    return Phrasly.build("k", led, transport=httpx.MockTransport(handler)), led


def test_success_plain_text_and_ledger(tmp_path):
    seen = {}

    def handler(req):
        seen["key"] = req.headers["x-api-key"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text="  humanized text  ", headers={"content-type": "text/plain"})

    p, led = _client(tmp_path, handler)
    r = p.humanize(TEXT, ref="x")
    assert r.text == "humanized text" and r.error is None
    assert seen == {"key": "k", "body": {"text": TEXT}}
    assert led.spent("phrasly") == pytest.approx(phrasly_cost(100))
    assert phrasly_cost(100) >= 0.014  # never below the list price


def test_http_error_released_not_billed(tmp_path):
    p, led = _client(tmp_path, lambda req: httpx.Response(401, text="bad key"))
    r = p.humanize(TEXT)
    assert r.text is None and "401" in r.error
    assert led.spent("phrasly") == 0 and led.outstanding("phrasly") == 0


def test_length_limits_skip_request(tmp_path):
    calls = []
    p, _ = _client(tmp_path, lambda req: calls.append(1) or httpx.Response(200, text="x"))
    assert p.humanize("too short").error.startswith("length")
    assert calls == []


def test_cap_enforced(tmp_path):
    p, _ = _client(tmp_path, lambda req: httpx.Response(200, text="ok"), cap=0.001)
    with pytest.raises(BudgetExceededError):
        p.humanize(TEXT)


def test_ultra_options_sent_and_priced(tmp_path):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text="ok text")

    led = SpendLedger(tmp_path / "l.sqlite", {"phrasly": 1.0})
    p = Phrasly.build(
        "k", led, transport=httpx.MockTransport(handler), options={"mode": "max", "proofread": True}
    )
    p.humanize(TEXT)
    assert seen["body"] == {"text": TEXT, "mode": "max", "proofread": True}
    assert led.spent("phrasly") == pytest.approx(phrasly_cost(100, ultra=True, settings=("proofread",)))
    assert phrasly_cost(100, ultra=True) > phrasly_cost(100)
    with pytest.raises(ValueError):
        Phrasly.build("k", led, options={"mode": "aggressive"})
    with pytest.raises(ValueError):
        Phrasly.build("k", led, options={"strength": "max"})
