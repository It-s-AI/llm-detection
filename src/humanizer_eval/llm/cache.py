"""On-disk cache of LLM replies (judge verdicts), so a repeated judgement costs nothing.

Judges run at temperature 0 with a JSON schema; the same request (model, messages, schema, reasoning,
token limit, routing policy) gives a reply we can reuse: re-running a gate on the same output, a
crashed run, re-scoring the king, or re-analysing a calibration with other thresholds. Only replies that
parsed as JSON are stored. Values may hold fragments of hidden texts, so they are encrypted at rest
when a key is given (NaCl SecretBox, same key as the period data).

``cache_all=True`` also caches non-zero-temperature calls (calibration variants, so a resumed run
sees the same variants); production gates keep the default.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from nacl.secret import SecretBox

from humanizer_eval.llm.openrouter import ChatResult


def _request_key(model: str, messages: list[dict[str, str]], kw: dict[str, Any]) -> str:
    policy = kw.get("policy")
    payload = {
        "model": model,
        "messages": messages,
        "json_schema": kw.get("json_schema"),
        "reasoning": kw.get("reasoning"),
        "max_tokens": kw.get("max_tokens"),
        "temperature": kw.get("temperature", 0.0),
        "seed": kw.get("seed"),
        "policy": getattr(policy, "name", None),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class CachedLLM:
    def __init__(self, llm: Any, path: Path | str, key: bytes | None = None, cache_all: bool = False):
        self.llm = llm
        self.box = SecretBox(key) if key else None
        self.cache_all = cache_all
        self.hits = self.misses = 0
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(p, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS replies (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
        self._db.commit()
        self._lock = threading.Lock()

    def _get(self, k: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM replies WHERE k = ?", (k,)).fetchone()
        if row is None:
            return None
        blob = self.box.decrypt(row[0]) if self.box else row[0]
        return json.loads(blob)

    def _put(self, k: str, value: dict[str, Any]) -> None:
        blob = json.dumps(value).encode()
        if self.box:
            blob = self.box.encrypt(blob)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO replies (k, v) VALUES (?, ?)", (k, blob))
            self._db.commit()

    def chat(self, model: str, messages: list[dict[str, str]], **kw: Any) -> ChatResult:
        cacheable = self.cache_all or (
            kw.get("temperature", 0.0) == 0.0 and kw.get("json_schema") is not None
        )
        if not cacheable:
            return self.llm.chat(model, messages, **kw)
        k = _request_key(model, messages, kw)
        hit = self._get(k)
        if hit is not None:
            self.hits += 1
            return ChatResult(
                text=hit["text"],
                model=hit["model"],
                provider=hit.get("provider"),
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                id=None,
                json=hit.get("json"),
                finish_reason="cached",
            )
        self.misses += 1
        r = self.llm.chat(model, messages, **kw)
        if kw.get("json_schema") is None or r.json is not None:
            self._put(k, {"text": r.text, "model": r.model, "provider": r.provider, "json": r.json})
        return r
