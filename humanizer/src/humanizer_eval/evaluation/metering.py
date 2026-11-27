"""Per-submission metering of judge tokens (D-80 statement: tokens per judge model).

Wrap the (cached) LLM once per submission. Cache hits report zero tokens, so an output that was
already judged costs nothing, like a detector cache hit.
"""

from __future__ import annotations

import threading
from typing import Any

from humanizer_eval.llm.openrouter import ChatResult


class MeteringLLM:
    def __init__(self, llm: Any):
        self.llm = llm
        self._lock = threading.Lock()
        self._tokens: dict[str, list[int]] = {}
        self._by_out: dict[int, dict[str, list[int]]] = {}  # output index -> model -> [in, out]
        self.calls = 0
        self.cost_usd = 0.0

    supports_refresh = True

    def for_output(self, i: int) -> _OutputView:
        """A view whose calls are also counted for output ``i`` (each output's judgement is charged once
        across attempts, whatever a deploy did to the cache)."""
        return _OutputView(self, i)

    def judge_tokens_by_output(self) -> dict[int, dict[str, tuple[int, int]]]:
        with self._lock:
            return {i: {m: (a, b) for m, (a, b) in d.items()} for i, d in self._by_out.items()}

    def chat(
        self, model: str, messages: list[dict[str, str]], _out: int | None = None, **kw: Any
    ) -> ChatResult:
        if not getattr(self.llm, "supports_refresh", False):
            kw.pop("cache_refresh", None)  # no cache below: nothing to skip
        r = self.llm.chat(model, messages, **kw)
        if kw.get("json_schema") is not None and not isinstance(getattr(r, "json", None), dict):
            # an unusable verdict (no JSON) is asked again: the owner pays failed judge calls, the miner
            # pays only the judgements that were delivered (it is never cached, so a resume asks again)
            with self._lock:
                self.calls += 1
            return r
        with self._lock:
            t = self._tokens.setdefault(model, [0, 0])
            t[0] += int(r.prompt_tokens or 0)
            t[1] += int(r.completion_tokens or 0)
            if _out is not None:
                o = self._by_out.setdefault(_out, {}).setdefault(model, [0, 0])
                o[0] += int(r.prompt_tokens or 0)
                o[1] += int(r.completion_tokens or 0)
            self.calls += 1
            self.cost_usd += float(r.cost_usd or 0.0)
        return r

    def judge_tokens(self) -> dict[str, tuple[int, int]]:
        """``{model: (tokens_in, tokens_out)}`` for models with any metered tokens."""
        with self._lock:
            return {m: (a, b) for m, (a, b) in self._tokens.items() if a or b}


class _OutputView:
    supports_refresh = True

    def __init__(self, meter: MeteringLLM, i: int):
        self.meter, self.i = meter, i
        self.llm = meter.llm

    def chat(self, model: str, messages: list[dict[str, str]], **kw: Any) -> ChatResult:
        return self.meter.chat(model, messages, _out=self.i, **kw)
