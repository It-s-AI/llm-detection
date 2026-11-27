"""Alerts to the owner's Telegram through @alarmer_bot (alarmerbot.ru): GET ?key=...&message=....

The key lives in the environment (ALARMER_KEY), never in the code. Alerts never raise: a failed alert is
logged and the caller carries on. Repeated alerts with the same ``dedupe`` key are sent at most once per
``min_interval_s`` (state in a small JSON file, so separate runs of the watchdog do not repeat themselves).

alarmerbot.ru serves an expired TLS certificate (2026-11-20), so the request goes over HTTPS without
certificate verification: still encrypted, the key is not critical (the owner's call).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)


class _RedactKey(logging.Filter):
    """httpx logs every request URL at INFO; the alarm key travels in the query (?key=...): mask it."""

    _rx = re.compile(r"([?&]key=)[^&\s\"']+")

    def filter(self, record: logging.LogRecord) -> bool:
        if "key=" in str(record.msg) or any("key=" in str(a) for a in (record.args or ())):
            record.msg = self._rx.sub(r"\1***", record.getMessage())
            record.args = None
        return True


logging.getLogger("httpx").addFilter(_RedactKey())

URL = "https://alarmerbot.ru/"
MAX_CHARS = 3500


class Alerter:
    def __init__(
        self,
        key: str,
        *,
        prefix: str = "SN32 humanizer",
        state_path: Path | str | None = None,
        min_interval_s: float = 3600.0,
        url: str = URL,
        http: Any = None,
        now=time.time,
    ):
        self.key, self.prefix, self.url, self.now = key, prefix, url, now
        self.min_interval_s = min_interval_s
        self.state_path = Path(state_path) if state_path else None
        # the service's certificate expired; the key is not critical (module docstring)
        self.http = http or httpx.Client(timeout=20, verify=False)  # noqa: S501
        self._lock = threading.Lock()
        self.sent: list[str] = []  # what went out in this process (tests, the watchdog's summary)

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def _state(self) -> dict[str, float]:
        if self.state_path and self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except ValueError:
                return {}
        return {}

    def send(self, text: str, *, dedupe: str | None = None, min_interval_s: float | None = None) -> bool:
        """Send ``text``; with ``dedupe``, skip it if the same key was sent less than the interval ago."""
        if not self.enabled:
            log.info("alert (no ALARMER_KEY): %s", text)
            return False
        with self._lock:
            state = self._state()
            gap = self.min_interval_s if min_interval_s is None else min_interval_s
            if dedupe and self.now() - state.get(dedupe, 0.0) < gap:
                return False
            message = f"[{self.prefix}] {text}"[:MAX_CHARS]
            try:
                r = self.http.get(self.url, params={"key": self.key, "message": message})
                ok = r.status_code == 200
            except httpx.HTTPError as e:
                log.warning("alert not sent (%s): %s", type(e).__name__, text)
                return False
            if not ok:
                log.warning("alert not sent (HTTP %s): %s", r.status_code, text)
                return False
            self.sent.append(message)
            if dedupe:
                state[dedupe] = self.now()
                if self.state_path:  # several Alerters and processes share it: a file lock, a private tmp
                    try:
                        self._save_key(dedupe, state[dedupe])
                    except Exception as e:
                        log.warning("alert state not saved: %s", e)
            return True

    def _save_key(self, key: str, ts: float) -> None:
        import fcntl
        import os
        import threading

        p = Path(self.state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p.with_name(p.name + ".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                cur = self._state()  # re-read under the lock: other writers' keys are kept
                cur[key] = ts
                tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(cur))
                tmp.replace(p)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def from_settings(state_path: Path | str | None = None, **kw: Any) -> Alerter:
    from humanizer_eval.config import get_settings

    return Alerter(get_settings().alarmer_key, state_path=state_path, **kw)
