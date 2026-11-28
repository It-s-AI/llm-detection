import json

import httpx
import pytest

from humanizer_eval.budget import BudgetExceededError, SpendLedger
from humanizer_eval.llm.openrouter import HIDDEN, PUBLIC, ModelPricing, OpenRouter, pinned


class FakePricing(ModelPricing):
    def __init__(self):
        super().__init__()
        self._prices = {"m/cheap": (1e-6, 2e-6), "e/emb": (1e-8, 0.0)}


def make(tmp_path, handler, cap=1.0):
    led = SpendLedger(tmp_path / "l.sqlite", {"openrouter": cap})
    client = OpenRouter(
        "sk-test",
        led,
        pricing=FakePricing(),
        log_path=tmp_path / "log.jsonl",
        transport=httpx.MockTransport(handler),
    )
    client.http.sleep = lambda _s: None
    return client, led


def ok_chat(captured):
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "model": "m/cheap",
                "provider": "Anthropic",
                "choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.000123},
            },
        )

    return handler


def test_hidden_policy_sends_strict_provider_block(tmp_path):
    cap = []
    client, _ = make(tmp_path, ok_chat(cap))
    client.chat("m/cheap", [{"role": "user", "content": "hi"}], policy=HIDDEN, purpose="t")
    prov = cap[0]["provider"]
    assert prov["allow_fallbacks"] is False
    assert prov["data_collection"] == "deny"
    assert "chutes" not in prov["only"] and "anthropic" in prov["only"]
    assert "chutes" in prov["ignore"]


def test_public_policy_still_ignores_bittensor_hosts(tmp_path):
    cap = []
    client, _ = make(tmp_path, ok_chat(cap))
    client.chat("m/cheap", [{"role": "user", "content": "hi"}], policy=PUBLIC)
    assert "only" not in cap[0]["provider"] and "chutes" in cap[0]["provider"]["ignore"]


def test_actual_cost_committed_and_json_parsed(tmp_path):
    client, led = make(tmp_path, ok_chat([]))
    r = client.chat(
        "m/cheap", [{"role": "user", "content": "hi"}], json_schema={"title": "x", "type": "object"}
    )
    assert r.json == {"a": 1}
    assert led.spent("openrouter") == pytest.approx(0.000123)
    assert led.outstanding("openrouter") == 0


def test_hidden_calls_do_not_log_content(tmp_path):
    client, _ = make(tmp_path, ok_chat([]))
    r = client.chat("m/cheap", [{"role": "user", "content": "SECRET PERIOD TEXT"}], policy=HIDDEN)
    log = (tmp_path / "log.jsonl").read_text()
    assert "SECRET PERIOD TEXT" not in log and r.raw == {}
    assert json.loads(log)["provider"] == "Anthropic"


def test_error_response_releases_reservation(tmp_path):
    def handler(_r):
        return httpx.Response(400, json={"error": {"message": "No allowed providers are available"}})

    client, led = make(tmp_path, handler)
    with pytest.raises(RuntimeError, match="No allowed providers"):
        client.chat("m/cheap", [{"role": "user", "content": "hi"}], policy=HIDDEN)
    assert led.spent("openrouter") == 0 and led.outstanding("openrouter") == 0


def test_budget_refuses_expensive_call(tmp_path):
    client, _ = make(tmp_path, ok_chat([]), cap=0.0001)
    with pytest.raises(BudgetExceededError):
        client.chat("m/cheap", [{"role": "user", "content": "x" * 3000}], max_tokens=4000)


def test_unknown_model_refused(tmp_path):
    client, _ = make(tmp_path, ok_chat([]))
    with pytest.raises(RuntimeError, match="unknown OpenRouter model"):
        client.chat("nope/model", [{"role": "user", "content": "hi"}])


def test_pinned_policy_restricts_order(tmp_path):
    pol = pinned(HIDDEN, ("anthropic", "chutes"))
    assert pol.only == ("anthropic",) and pol.order == ("anthropic",) and not pol.allow_fallbacks
    with pytest.raises(ValueError):
        pinned(HIDDEN, ("chutes",))


def test_embeddings(tmp_path):
    def handler(request):
        n = len(json.loads(request.content)["input"])
        return httpx.Response(
            200,
            json={
                "model": "e/emb",
                "data": [{"index": i, "embedding": [i, 1.0]} for i in range(n)][::-1],
                "usage": {"cost": 1e-6},
            },
        )

    client, led = make(tmp_path, handler)
    r = client.embed("e/emb", ["a", "b", "c"])
    assert [v[0] for v in r.vectors] == [0, 1, 2]
    assert led.spent("openrouter") == pytest.approx(1e-6)


def test_parse_json_loose():
    from humanizer_eval.llm.openrouter import parse_json_loose

    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('Sure! {"a": 2} hope this helps') == {"a": 2}
    assert parse_json_loose('{"a": "truncated') is None
