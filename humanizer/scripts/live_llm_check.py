"""Live check of OpenRouter routing (HIDDEN allowlist per model family) + embeddings. Costs < $0.05."""

from __future__ import annotations

import json

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.llm.openrouter import HIDDEN, PUBLIC, ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir

MODELS = [
    "anthropic/claude-haiku-5.5",
    "openai/gpt-6-luna",
    "google/gemini-3.6-flash",
    "x-ai/grok-4.7",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.8-27b",
    "meta-llama/llama-4-maverick",
]


def main() -> int:
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    client = OpenRouter(
        s.openrouter_api_key,
        led,
        pricing=ModelPricing(var_dir() / "openrouter_models.json"),
        log_path=var_dir() / "llm_calls.jsonl",
    )
    out = {}
    for m in MODELS:
        try:
            r = client.chat(
                m,
                [{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=16,
                policy=HIDDEN,
                purpose="live-check",
            )
            out[m] = {"ok": True, "provider": r.provider, "cost": r.cost_usd, "text": r.text.strip()[:20]}
        except Exception as e:
            out[m] = {"ok": False, "error": str(e)[:200]}
        print(m, out[m])
    schema = {
        "title": "verdict",
        "type": "object",
        "additionalProperties": False,
        "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}},
        "required": ["answer", "confidence"],
    }
    try:
        r = client.chat(
            "anthropic/claude-haiku-5.5",
            [{"role": "user", "content": "Is water wet? Answer briefly."}],
            max_tokens=60,
            json_schema=schema,
            policy=HIDDEN,
            purpose="live-check-json",
        )
        out["json_schema"] = {"ok": r.json is not None, "json": r.json, "provider": r.provider}
    except Exception as e:
        out["json_schema"] = {"ok": False, "error": str(e)[:200]}
    print("json_schema", out["json_schema"])
    try:
        e = client.embed(
            "baai/bge-large-en-v1.5", ["hello world", "a second text"], policy=PUBLIC, purpose="live-check"
        )
        out["embeddings"] = {"ok": True, "dim": len(e.vectors[0]), "provider": e.provider, "cost": e.cost_usd}
    except Exception as ex:
        out["embeddings"] = {"ok": False, "error": str(ex)[:200]}
    print("embeddings", out["embeddings"])
    print("ledger openrouter:", led.summary().get("openrouter"))
    (var_dir() / "live_llm_check.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
