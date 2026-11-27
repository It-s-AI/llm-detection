"""OpenRouter gateway for every LLM / embedding call (D-90).

* Routing policies: `PUBLIC` (anything goes) and `HIDDEN` (for prompts that contain
  hidden period data): only allow-listed providers, no fallbacks outside the list,
  `data_collection: deny`, ZDR endpoints where available.
* Budget: an upper-bound estimate is reserved in the spend ledger before the
  request; the actual cost reported by OpenRouter (`usage.cost`) is committed.
* Audit log: one JSON line per call (model, served provider, tokens, cost,
  purpose, request hash). For HIDDEN calls no prompt/response content is logged.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from humanizer_eval.budget import SpendLedger
from humanizer_eval.detectors.http import HttpClient, RetryPolicy

API = "https://openrouter.ai/api/v1"

# Providers trusted with hidden period data: hyperscalers, first-party labs and a
# few large commercial hosts. Deliberately excludes decentralized / Bittensor-run
# hosts (e.g. Chutes) and small unknown hosts (D-90).
TRUSTED_PROVIDERS: tuple[str, ...] = (
    "openai",
    "anthropic",
    "google-vertex",
    "google-ai-studio",
    "azure",
    "amazon-bedrock",
    "xai",
    "mistral",
    "alibaba",
    "deepinfra",
    "together",
    "fireworks",
)
# Never used even for public data (consistency + ecosystem-participant hosts).
ALWAYS_IGNORED: tuple[str, ...] = ("chutes", "targon")


@dataclass(frozen=True)
class RoutingPolicy:
    name: str
    only: tuple[str, ...] | None = None
    allow_fallbacks: bool = True
    data_collection: str = "allow"
    zdr: bool | None = None
    order: tuple[str, ...] | None = None

    def provider_block(self) -> dict[str, Any]:
        block: dict[str, Any] = {
            "allow_fallbacks": self.allow_fallbacks,
            "data_collection": self.data_collection,
            "ignore": list(ALWAYS_IGNORED),
        }
        if self.only is not None:
            block["only"] = list(self.only)
        if self.order is not None:
            block["order"] = list(self.order)
        if self.zdr is not None:
            block["zdr"] = self.zdr
        return block


PUBLIC = RoutingPolicy("public")
HIDDEN = RoutingPolicy(
    "hidden", only=TRUSTED_PROVIDERS, allow_fallbacks=False, data_collection="deny", zdr=None
)


def pinned(policy: RoutingPolicy, providers: tuple[str, ...]) -> RoutingPolicy:
    """Judges: fixed provider order, no fallbacks outside it (pause instead, D-90)."""
    allowed = tuple(p for p in providers if policy.only is None or p in policy.only)
    if not allowed:
        raise ValueError("pinned providers are not allowed by the base policy")
    return RoutingPolicy(
        # the providers are part of the name, so the judge cache never reuses a verdict from another one
        f"{policy.name}+pinned:{','.join(allowed)}",
        only=allowed,
        allow_fallbacks=False,
        data_collection=policy.data_collection,
        zdr=policy.zdr,
        order=allowed,
    )


@dataclass
class ChatResult:
    text: str
    model: str
    provider: str | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    id: str | None
    json: Any = None
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    model: str
    provider: str | None
    cost_usd: float


class ModelPricing:
    """Per-token prices from OpenRouter's public model list, cached on disk for a day."""

    def __init__(self, cache_path: Path | None = None, transport: httpx.BaseTransport | None = None):
        self.cache_path = cache_path
        self.transport = transport
        self._prices: dict[str, tuple[float, float]] | None = None

    def _load(self) -> dict[str, tuple[float, float]]:
        if self._prices is not None:
            return self._prices
        data = None
        if (
            self.cache_path
            and self.cache_path.exists()
            and time.time() - self.cache_path.stat().st_mtime < 86400
        ):
            data = json.loads(self.cache_path.read_text())
        if data is None:
            with httpx.Client(transport=self.transport, timeout=60) as c:
                chat = c.get(f"{API}/models").json()["data"]
                try:
                    emb = c.get(f"{API}/embeddings/models").json().get("data", [])
                except Exception:
                    emb = []
            data = {
                m["id"]: [float(m["pricing"].get("prompt", 0)), float(m["pricing"].get("completion", 0))]
                for m in chat + emb
            }
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(data))
        self._prices = {k: (v[0], v[1]) for k, v in data.items()}
        return self._prices

    def per_token(self, model: str) -> tuple[float, float]:
        prices = self._load()
        if model not in prices:
            raise KeyError(f"unknown OpenRouter model {model!r}")
        return prices[model]

    def estimate(self, model: str, prompt_chars: int, max_tokens: int) -> float:
        p_in, p_out = self.per_token(model)
        prompt_tokens = prompt_chars / 3.0  # conservative (real ratio ~4 chars/token for English)
        return 1.2 * (prompt_tokens * p_in + max_tokens * p_out) + 1e-6


def parse_json_loose(text: str) -> Any:
    """Parse JSON from a model reply, tolerating ```json fences and surrounding prose."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    try:
        return json.loads(s)
    except ValueError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except ValueError:
            return None
    return None


def _req_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class OpenRouter:
    service = "openrouter"

    def __init__(
        self,
        api_key: str,
        ledger: SpendLedger,
        *,
        pricing: ModelPricing | None = None,
        log_path: Path | str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 180.0,
    ):
        if ledger is None:
            raise ValueError("OpenRouter requires a spend ledger")
        self.ledger = ledger
        self.pricing = pricing or ModelPricing()
        self.log_path = Path(log_path) if log_path else None
        self.http = HttpClient(
            base_url=API,
            headers={"Authorization": f"Bearer {api_key}"},
            rate_per_sec=5,
            burst=10,
            timeout=timeout,
            transport=transport,
            retry=RetryPolicy(max_attempts=4),
        )

    def _log(self, record: dict[str, Any]) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int | None = None,
        json_schema: dict[str, Any] | None = None,
        policy: RoutingPolicy = PUBLIC,
        purpose: str = "",
        reasoning: dict[str, Any] | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "provider": policy.provider_block(),
            "usage": {"include": True},
        }
        if seed is not None:
            payload["seed"] = seed
        if reasoning is not None:
            payload["reasoning"] = (
                reasoning  # e.g. {"effort": "low"}; thinking tokens count toward max_tokens
            )
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "output"),
                    "strict": True,
                    "schema": json_schema,
                },
            }
        est = self.pricing.estimate(model, sum(len(m.get("content", "")) for m in messages), max_tokens)
        rhash = _req_hash(payload)
        res = self.ledger.reserve(self.service, est, ref=f"chat:{purpose}:{rhash}")
        t0 = time.time()
        counting = hasattr(self.http, "begin_paid")
        if counting:
            self.http.begin_paid()

        def extra() -> int:  # earlier sends of this call delivered with an unknown outcome (billed)
            return max(0, self.http.paid_sends() - 1) if counting else 0

        try:
            resp = self.http.request("POST", "/chat/completions", json=payload)
        except RuntimeError:
            n = self.http.paid_sends() if counting else 0
            if n:  # delivered and timed out: OpenRouter generated (and billed) the reply
                res.commit(est * n, {"error": "transport", "sends": n})
            else:
                res.release({"error": "transport"})  # never reached OpenRouter -> not billed
            raise
        except Exception as e:
            res.commit(est, {"error": type(e).__name__})
            raise
        try:
            body = resp.json()
        except ValueError:
            body = {}
        if resp.status_code >= 400 or "error" in body or not body:
            n = self.http.paid_sends() if counting else 0
            if n:
                res.commit(est * n, {"status": resp.status_code, "sends": n})
            else:
                res.release({"status": resp.status_code})  # failed requests are not billed
            err = body.get("error") if isinstance(body, dict) else None
            msg = err.get("message") if isinstance(err, dict) else err
            raise RuntimeError(f"openrouter HTTP {resp.status_code}: {str(msg)[:300]}")
        usage = body.get("usage") or {}
        cost = float(usage.get("cost", est))
        res.commit(cost + est * extra(), {"model": model, "provider": body.get("provider")})
        choice = (body.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        parsed = None
        if json_schema is not None:
            parsed = parse_json_loose(text)
        out = ChatResult(
            text=text,
            model=body.get("model", model),
            provider=body.get("provider"),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            cost_usd=cost,
            id=body.get("id"),
            json=parsed,
            finish_reason=choice.get("finish_reason"),
            raw={} if policy.name.startswith("hidden") else body,
        )
        self._log(
            {
                "ts": t0,
                "sec": round(time.time() - t0, 2),
                "kind": "chat",
                "purpose": purpose,
                "policy": policy.name,
                "model": out.model,
                "provider": out.provider,
                "req": rhash,
                "prompt_tokens": out.prompt_tokens,
                "completion_tokens": out.completion_tokens,
                "cost": cost,
                "finish": choice.get("finish_reason"),
            }
        )
        return out

    def embed(
        self, model: str, inputs: list[str], *, policy: RoutingPolicy = PUBLIC, purpose: str = ""
    ) -> EmbedResult:
        payload = {"model": model, "input": inputs, "provider": policy.provider_block()}
        p_in, _ = self.pricing.per_token(model)
        est = 1.2 * p_in * sum(len(t) for t in inputs) / 3.0 + 1e-6
        res = self.ledger.reserve(self.service, est, ref=f"embed:{purpose}")
        resp = self.http.request("POST", "/embeddings", json=payload)
        body = resp.json()
        if resp.status_code >= 400 or "error" in body:
            res.release({"status": resp.status_code})
            raise RuntimeError(f"openrouter embeddings HTTP {resp.status_code}: {str(body)[:300]}")
        cost = float((body.get("usage") or {}).get("cost", est))
        res.commit(cost, {"model": model})
        vectors = [d["embedding"] for d in sorted(body["data"], key=lambda d: d.get("index", 0))]
        self._log(
            {
                "ts": time.time(),
                "kind": "embed",
                "purpose": purpose,
                "policy": policy.name,
                "model": model,
                "provider": body.get("provider"),
                "n": len(inputs),
                "cost": cost,
            }
        )
        return EmbedResult(vectors, body.get("model", model), body.get("provider"), cost)
