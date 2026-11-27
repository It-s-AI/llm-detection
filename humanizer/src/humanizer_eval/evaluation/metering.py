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
        self.calls = 0
        self.cost_usd = 0.0

    def chat(self, model: str, messages: list[dict[str, str]], **kw: Any) -> ChatResult:
        r = self.llm.chat(model, messages, **kw)
        with self._lock:
            t = self._tokens.setdefault(model, [0, 0])
            t[0] += int(r.prompt_tokens or 0)
            t[1] += int(r.completion_tokens or 0)
            self.calls += 1
            self.cost_usd += float(r.cost_usd or 0.0)
        return r

    def judge_tokens(self) -> dict[str, tuple[int, int]]:
        """``{model: (tokens_in, tokens_out)}`` for models with any metered tokens."""
        with self._lock:
            return {m: (a, b) for m, (a, b) in self._tokens.items() if a or b}
