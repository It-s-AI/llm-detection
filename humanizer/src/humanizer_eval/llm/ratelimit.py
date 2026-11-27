"""Per-model request rate limit for OpenRouter (measured 2026-11-27 on the production key).

OpenRouter limits a new account per MODEL ("new-account-rpm/<model>"): at most 20 requests in each calendar
minute (the 429's x-ratelimit-reset is the next whole minute), independently for every model, and only on
some models (GPT, Gemini, Claude Sonnet; Qwen, Mistral and Claude Haiku had none). Requests to different
models do not compete, so a call waits only for its own model's budget; other models run at full concurrency.

``OPENROUTER_MODEL_RPM`` (default 18, a margin under 20): the budget per model per minute; 0 turns the limit
off (once OpenRouter lifts the new-account limit, or with our own provider keys: BYOK).
``OPENROUTER_RPM_EXEMPT``: comma-separated model prefixes without the limit (default: the unlimited ones).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any


class ModelRateLimiter:
    """At most ``rpm`` calls per model in each calendar minute (OpenRouter's window: a 429 carries
    ``x-ratelimit-reset`` = the next whole minute), counted on the wall clock; a full model waits for the next
    minute (plus ``skew`` seconds for the clocks' difference), other models are not affected."""

    def __init__(
        self,
        rpm: float,
        *,
        exempt: tuple[str, ...] = (),
        skew: float = 1.0,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rpm = int(rpm)
        self.exempt = tuple(exempt)
        self.skew = skew
        self.now, self.sleep = now, sleep
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[int, int]] = {}  # model -> (minute, calls in it)

    def wait(self, model: str) -> float:
        """Block until ``model`` may be called; returns the seconds waited."""
        if self.rpm <= 0 or model.startswith(self.exempt):
            return 0.0
        waited = 0.0
        while True:
            with self._lock:
                t = self.now()
                minute = int((t - self.skew) // 60)  # a call just after :00 still counts in the old minute
                cur, n = self._windows.get(model, (minute, 0))
                if cur != minute:
                    cur, n = minute, 0
                if n < self.rpm:
                    self._windows[model] = (cur, n + 1)
                    return waited
                pause = (minute + 1) * 60 + self.skew - t + 0.05
            self.sleep(pause)
            waited += pause


class RateLimitedLLM:
    """Wraps an LLM client (``chat(model, messages, **kw)``): every call first waits for its model's token.
    Everything else is the wrapped client's (its ledger, pricing, etc.)."""

    def __init__(self, llm: Any, limiter: ModelRateLimiter):
        self._llm, self._limiter = llm, limiter

    def chat(self, model: str, messages: Any, **kw: Any) -> Any:
        self._limiter.wait(model)
        return self._llm.chat(model, messages, **kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


EXEMPT_DEFAULT = "anthropic/claude-haiku,qwen/,mistralai/"


def limited(llm: Any) -> Any:
    """``llm`` behind the per-model limit configured in the environment (unchanged if it is off)."""
    rpm = int(os.environ.get("OPENROUTER_MODEL_RPM", "18") or 0)
    if rpm <= 0:
        return llm
    exempt = tuple(
        x.strip() for x in os.environ.get("OPENROUTER_RPM_EXEMPT", EXEMPT_DEFAULT).split(",") if x.strip()
    )
    return RateLimitedLLM(llm, ModelRateLimiter(rpm, exempt=exempt))
