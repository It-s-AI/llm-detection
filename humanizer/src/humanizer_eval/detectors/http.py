"""HTTP plumbing shared by vendor adapters: retries, rate limiting, injectable transport."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class TokenBucket:
    """Thread-safe token bucket: at most `rate` requests per second, bursts up to `burst`."""

    def __init__(self, rate: float, burst: int = 1, clock: Callable[[], float] = time.monotonic):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = max(1, burst)
        self.tokens = float(self.capacity)
        self.clock = clock
        self.updated = clock()
        self.lock = threading.Lock()

    def acquire(self, sleep: Callable[[float], None] = time.sleep) -> None:
        while True:
            with self.lock:
                now = self.clock()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            sleep(wait)


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0

    def delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(self.max_delay, max(0.0, float(retry_after)))
            except ValueError:
                pass
        d = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return d * (0.5 + random.random() / 2)  # noqa: S311 - jitter, not crypto


class HttpClient:
    """Thin wrapper over httpx.Client with retry + rate limit.

    Secrets live only in `headers`; they are never included in exceptions or logs
    produced here.
    """

    def __init__(
        self,
        base_url: str = "",
        headers: dict[str, str] | None = None,
        *,
        rate_per_sec: float = 5.0,
        burst: int = 5,
        timeout: float = 60.0,
        retry: RetryPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._client = httpx.Client(
            base_url=base_url, headers=headers or {}, timeout=timeout, transport=transport
        )
        self.bucket = TokenBucket(rate_per_sec, burst)
        self.retry = retry or RetryPolicy()
        self.sleep = sleep
        self._tl = threading.local()  # per thread: what the paid sends of the current call did

    def begin_paid(self) -> None:
        """Start counting the sends of one paid call (``paid_sends`` after it)."""
        self._tl.accepted = 0
        self._tl.unknown = 0

    def paid_sends(self) -> int:
        """Sends since ``begin_paid`` the vendor may have billed: accepted (2xx) plus delivered with an
        unknown outcome (a read timeout or a dropped connection after the request went out). A send
        answered with an error status or one that never connected is not counted."""
        return int(getattr(self._tl, "accepted", 0)) + int(getattr(self._tl, "unknown", 0))

    def _count(self, method: str, resp: httpx.Response | None, exc: Exception | None) -> None:
        if method.upper() == "GET" or not hasattr(self._tl, "accepted"):
            return
        if exc is not None:
            if not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
                self._tl.unknown += 1
        elif resp is not None and resp.status_code < 300:
            self._tl.accepted += 1

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self.retry.max_attempts + 1):
            self.bucket.acquire(self.sleep)
            try:
                resp = self._client.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                self._count(method, None, e)
                last_exc = e
                if attempt == self.retry.max_attempts:
                    break
                self.sleep(self.retry.delay(attempt, None))
                continue
            self._count(method, resp, None)
            if resp.status_code in RETRY_STATUS and attempt < self.retry.max_attempts:
                self.sleep(self.retry.delay(attempt, resp.headers.get("retry-after")))
                continue
            return resp
        raise RuntimeError(f"{method} {url}: transport failure after retries: {type(last_exc).__name__}")

    def close(self) -> None:
        self._client.close()
