# 01b. LLM gateway (OpenRouter) — report

## What was done
- A single OpenRouter client for all LLM calls and embeddings (D-90): data generation, judges, pool embeddings.
- Two routing policies:
  - **HIDDEN** — for any requests containing hidden period data: only an allowlist of providers
    (OpenAI, Anthropic, Google Vertex/AI Studio, Azure, Amazon Bedrock, xAI, Mistral, Alibaba, DeepInfra,
    Together, Fireworks), `allow_fallbacks: false`, `data_collection: deny`;
  - **PUBLIC** — for public data; but even here Bittensor hosters (Chutes, Targon) are always excluded.
  - `pinned(...)` — for judges: a fixed provider order with no fallbacks (no provider → error, evaluation is paused).
- Budget: before a request, a reservation at the upper-bound estimate (model prices from the public OpenRouter API); afterwards, the
  **actual** cost from the response (`usage.cost`) is charged; failed requests are not charged.
- Call log `var/llm_calls.jsonl`: model, **which provider actually served the request**, tokens, price, purpose;
  for HIDDEN requests the content is not written.
- Robust parsing of JSON responses (code blocks, surrounding text); the `finish_reason` field (we catch truncated responses);
  the `reasoning` parameter for "thinking" models.

## Live check (HIDDEN, one short request each)
| Model | Result | Served by |
|---|---|---|
| anthropic/claude-haiku-5.5 | ✅ | Amazon Bedrock |
| openai/gpt-6-luna | ✅ | OpenAI |
| google/gemini-3.6-flash | ✅ (needs token headroom for reasoning) | Google |
| x-ai/grok-4.7 | ✅ | xAI |
| deepseek/deepseek-v4-pro | ✅ | DeepInfra |
| qwen/qwen3.8-27b | ✅ (needs token headroom for reasoning) | DeepInfra |
| meta-llama/llama-4-maverick | ⛔ rejected — no trusted provider (only DigitalOcean, Parasail) | — |
| embeddings baai/bge-large-en-v1.5 (PUBLIC) | ✅ dim=1024 | DeepInfra |

Conclusion: the allowlist works, and it noticeably narrows the choice of open models (for example, without the
allowlist Qwen 3.8 27B is also served by Chutes and by small hosters). Cost of the check — **~$0.001**.

## Findings
- "Thinking" models (Gemini 3.6 Flash, Qwen 3.8) spend part of `max_tokens` on reasoning: with a low limit
  the response is empty. For judges and generation we set `max_tokens` ≥ 512 and/or `reasoning: {"effort": "low"}`.
- A JSON response is truncated when `max_tokens` is small → `finish_reason="length"` is now visible.

## Tests
41 offline tests in the repository are green; for the gateway: strict `provider` block for HIDDEN, ignoring Bittensor hosters for
PUBLIC, charging the actual price, no HIDDEN request content in the log, releasing the reservation on
error, refusal when the budget is exceeded, unknown model, `pinned`, embeddings, JSON parsing.

## How to check
```bash
.venv/bin/pytest tests/llm
.venv/bin/python scripts/live_llm_check.py   # < $0.01
```

## For you to review
- The provider allowlist `TRUSTED_PROVIDERS` in `src/humanizer_eval/llm/openrouter.py` — whether you agree with its composition.
