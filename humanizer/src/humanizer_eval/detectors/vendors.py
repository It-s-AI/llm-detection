"""Vendor adapters: GPTZero, ZeroGPT, Pangram, It's AI.

Each adapter turns a vendor response into a primary continuous AI score in [0, 1]
plus alternative `candidates`. All paid calls are budgeted through the spend
ledger *before* the request (upper-bound estimate) and committed afterwards.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx

from humanizer_eval.budget import BudgetExceededError, SpendLedger
from humanizer_eval.detectors.base import DetectorError, DetectorScore, text_hash, word_count
from humanizer_eval.detectors.cache import DetectorCache
from humanizer_eval.detectors.http import HttpClient, RetryPolicy

# --- pricing (upper-bound estimates; vendors do not return per-call cost) -----


def pangram_cost(words: int) -> float:
    """Official: $0.05 per 100 words; billed in 100-word units (conservative ceil)."""
    return 0.05 * max(1, math.ceil(words / 100))


def gptzero_cost(words: int) -> float:
    """Third-party figure: $150 per 1M words overage (plan-based; conservative)."""
    return 150.0 * words / 1_000_000


def zerogpt_cost(words: int) -> float:
    """Third-party figure: up to $0.069 per 1k words (upper bound of reported range)."""
    return 0.069 * words / 1000


@dataclass
class Extracted:
    score: float
    candidates: dict[str, float]
    version: str | None
    raw: dict[str, Any]


class VendorDetector:
    """Base class: cache → budget → HTTP → parse → cache."""

    name: str = "vendor"
    service: str = "vendor"  # spend-ledger service name
    config: str = "v1"  # part of the cache key; bump when request semantics change
    max_workers: int = 4

    def __init__(
        self,
        http: HttpClient,
        *,
        ledger: SpendLedger | None,
        cache: DetectorCache | None = None,
        cost_fn: Callable[[int], float] | None = None,
        primary: str | None = None,
    ):
        self.http = http
        self.ledger = ledger
        self.cache = cache
        self.cost_fn = cost_fn or (lambda _w: 0.0)
        if primary:
            self.primary = primary

    primary: str = ""

    # subclasses implement
    def _call(self, text: str) -> dict[str, Any]:
        raise NotImplementedError

    def _extract(self, payload: dict[str, Any]) -> Extracted:
        raise NotImplementedError

    def _one(self, text: str) -> DetectorScore:
        sha = text_hash(text)
        if self.cache is not None:
            hit = self.cache.get(self.name, self.config, sha)
            if hit is not None:
                return hit
        est = self.cost_fn(word_count(text))
        res = None
        if self.ledger is not None:
            try:
                res = self.ledger.reserve(self.service, est, ref=f"{self.name}:{sha[:12]}")
            except BudgetExceededError as e:
                return DetectorScore(self.name, sha, float("nan"), error=f"budget: {e}")
        elif est > 0:
            return DetectorScore(self.name, sha, float("nan"), error="paid detector requires a spend ledger")
        http = getattr(self, "http", None)
        counting = http is not None and hasattr(http, "begin_paid")
        if counting:
            http.begin_paid()
        try:
            payload = self._call(text)
        except Exception as e:
            # booked: every send the vendor may have billed (accepted, or delivered and timed out); a call
            # the vendor refused with an error status costs nothing (an outage must not eat the cap)
            n = http.paid_sends() if counting else 1
            if res is not None:
                if n:
                    res.commit(est * n, {"error": type(e).__name__, "sends": n})
                else:
                    res.release({"error": type(e).__name__})
            return DetectorScore(self.name, sha, float("nan"), cost_usd=est * n,
                                 error=f"{type(e).__name__}: {e}")  # fmt: skip
        n = max(1, http.paid_sends()) if counting else 1
        if res is not None:
            res.commit(est * n, {"sends": n} if n > 1 else None)
        try:
            ex = self._extract(payload)
        except Exception as e:
            return DetectorScore(
                self.name,
                sha,
                float("nan"),
                raw=payload,
                cost_usd=est,
                error=f"parse: {type(e).__name__}: {e}",
            )
        out = DetectorScore(
            detector=self.name,
            text_sha256=sha,
            score=ex.score,
            candidates=ex.candidates,
            version=ex.version,
            raw=ex.raw,
            cost_usd=est,
        )
        if self.cache is not None:
            self.cache.put(self.config, out)
        return out

    def score(self, texts: list[str]) -> list[DetectorScore]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._one, texts))


def _check(resp: httpx.Response, vendor: str) -> dict[str, Any]:
    if resp.status_code >= 400:
        # never echo request headers; body is the vendor's error message
        raise DetectorError(f"{vendor} HTTP {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()
    except ValueError as e:
        raise DetectorError(f"{vendor}: non-JSON response") from e


def _clip01(x: float) -> float:
    v = float(x)
    if v != v:  # NaN: never a "human" 0.0
        raise DetectorError("the detector returned NaN")
    return min(1.0, max(0.0, v))


# --- GPTZero -------------------------------------------------------------------


class GPTZero(VendorDetector):
    name = service = "gptzero"
    config = "v2-predict-text"
    primary = "p_ai_plus_mixed"
    URL = "https://api.gptzero.me/v2/predict/text"

    @classmethod
    def build(cls, api_key: str, **kw) -> GPTZero:
        transport = kw.pop("transport", None)
        http = HttpClient(headers={"x-api-key": api_key}, rate_per_sec=4, burst=4, transport=transport)
        return cls(http, cost_fn=gptzero_cost, **kw)

    def _call(self, text: str) -> dict[str, Any]:
        return _check(self.http.request("POST", self.URL, json={"document": text}), "gptzero")

    def _extract(self, p: dict[str, Any]) -> Extracted:
        doc = p["documents"][0]
        cp = doc.get("class_probabilities") or {}
        if not isinstance(cp.get("ai"), int | float):  # never a "human" score from an incomplete answer
            raise DetectorError("gptzero: no class_probabilities.ai in the response")
        cands = {
            "p_ai": _clip01(cp.get("ai", 0.0)),
            "p_mixed": _clip01(cp.get("mixed", 0.0)),
            "p_ai_plus_mixed": _clip01(cp.get("ai", 0.0) + cp.get("mixed", 0.0)),
            "completely_generated_prob": _clip01(doc.get("completely_generated_prob", 0.0)),
            "average_generated_prob": _clip01(doc.get("average_generated_prob", 0.0)),
        }
        slim = {
            k: doc.get(k) for k in ("class_probabilities", "completely_generated_prob", "predicted_class")
        }
        return Extracted(cands[self.primary], cands, p.get("version"), {"version": p.get("version"), **slim})


# --- ZeroGPT ---------------------------------------------------------------------


class ZeroGPT(VendorDetector):
    name = service = "zerogpt"
    config = "detectText-v1"
    primary = "fake_percentage"
    URL = "https://api.zerogpt.com/api/detect/detectText"

    @classmethod
    def build(cls, api_key: str, **kw) -> ZeroGPT:
        transport = kw.pop("transport", None)
        http = HttpClient(headers={"ApiKey": api_key}, rate_per_sec=4, burst=4, transport=transport)
        return cls(http, cost_fn=zerogpt_cost, **kw)

    def _call(self, text: str) -> dict[str, Any]:
        return _check(self.http.request("POST", self.URL, json={"input_text": text}), "zerogpt")

    def _extract(self, p: dict[str, Any]) -> Extracted:
        if p.get("code") != 200 or not isinstance(p.get("data"), dict):
            raise DetectorError(f"zerogpt: code={p.get('code')} message={p.get('message')!r}")
        d = p["data"]
        if d.get("fakePercentage") is None:
            raise DetectorError("zerogpt: missing fakePercentage")
        cands = {
            "fake_percentage": _clip01(float(d["fakePercentage"]) / 100.0),
            "is_human_raw": float(d.get("isHuman") or 0.0),  # observed 0 or 100, not a probability
        }
        slim = {k: d.get(k) for k in ("fakePercentage", "isHuman", "textWords", "aiWords")}
        return Extracted(cands[self.primary], cands, None, slim)


# --- Pangram -------------------------------------------------------------------


class Pangram(VendorDetector):
    """Async task API (v2 endpoints were sunset on 2026-04-01)."""

    name = service = "pangram"
    primary = "max_ai_humanizer"
    BASE = "https://text.external-api.pangram.com"

    def __init__(
        self,
        http: HttpClient,
        *,
        model: str = "pangram-4.1",
        poll_interval: float = 1.5,
        poll_timeout: float = 180.0,
        sleep: Callable[[float], None] = time.sleep,
        **kw,
    ):
        super().__init__(http, **kw)
        self.model = model
        self.config = f"task-{model}"
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self._sleep = sleep

    @classmethod
    def build(cls, api_key: str, **kw) -> Pangram:
        transport = kw.pop("transport", None)
        http = HttpClient(
            base_url=cls.BASE,
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            rate_per_sec=4,
            burst=4,
            transport=transport,
            retry=RetryPolicy(max_attempts=5),
        )
        return cls(http, cost_fn=pangram_cost, **kw)

    def _call(self, text: str) -> dict[str, Any]:
        created = _check(
            self.http.request("POST", "/task", json={"text": text, "model": self.model}), "pangram"
        )
        task_id = created.get("task_id")
        if not task_id:
            raise DetectorError("pangram: no task_id in response")
        deadline = time.monotonic() + self.poll_timeout
        while True:
            body = _check(self.http.request("GET", f"/task/{task_id}"), "pangram")
            stage = body.get("stage")
            if stage == "STAGE_SUCCESS":
                return body
            if stage == "STAGE_FAILED":
                raise DetectorError(f"pangram: task failed: {body.get('error') or body.get('message')}")
            if time.monotonic() > deadline:
                raise DetectorError("pangram: polling timeout")
            self._sleep(self.poll_interval)

    def _extract(self, p: dict[str, Any]) -> Extracted:
        windows = p.get("windows") or []
        if not windows:
            raise DetectorError("pangram: no windows in response")
        total = ai = hum = 0.0
        for w in windows:
            weight = float(w.get("token_length") or w.get("word_count") or 1)
            cp = w.get("class_probabilities")
            if cp and len(cp) == 3:  # (Human, AI-Assisted, AI-Generated)
                p_ai = float(cp[1]) + float(cp[2])
            elif isinstance(w.get("ai_assistance_score"), int | float):
                p_ai = float(w["ai_assistance_score"])
            else:  # never a "human" score from an incomplete answer
                raise DetectorError("pangram: a window without class_probabilities or ai_assistance_score")
            ai += weight * p_ai
            hum += weight * float(w.get("humanizer_score") or 0.0)
            total += weight
        w_ai, w_hum = ai / total, hum / total
        cands = {
            "fraction_ai": _clip01(p.get("fraction_ai", 0.0)),
            "fraction_ai_or_assisted": _clip01(
                p.get("fraction_ai", 0.0) + p.get("fraction_ai_assisted", 0.0)
            ),
            "weighted_window_ai": _clip01(w_ai),
            "weighted_humanizer": _clip01(w_hum),
            "max_ai_humanizer": _clip01(max(w_ai, w_hum)),
        }
        slim = {
            "version": p.get("version"),
            "headline": p.get("headline"),
            "fraction_ai": p.get("fraction_ai"),
            "fraction_ai_assisted": p.get("fraction_ai_assisted"),
            "windows": [
                {
                    k: w.get(k)
                    for k in (
                        "label",
                        "class_probabilities",
                        "humanizer_score",
                        "is_humanized",
                        "token_length",
                        "confidence",
                    )
                }
                for w in windows
            ],
        }
        return Extracted(cands[self.primary], cands, p.get("version"), slim)


# --- It's AI (proxy for simulations; never used in crown scoring, D-33) ----------


class ItsAI(VendorDetector):
    name = service = "its_ai"
    config = "special-predict-v1"
    primary = "mean_word_prob"
    batch_size = 32

    def __init__(self, http: HttpClient, *, url: str, api_key: str, **kw):
        super().__init__(http, **kw)
        self.url = url
        self._api_key = api_key

    @classmethod
    def build(cls, url: str, api_key: str, **kw) -> ItsAI:
        transport = kw.pop("transport", None)
        http = HttpClient(rate_per_sec=2, burst=2, timeout=120, transport=transport)
        return cls(http, url=url, api_key=api_key, **kw)

    def _call(self, text: str) -> dict[str, Any]:  # pragma: no cover - batching path is used
        return self._call_batch([text])[0]

    def _call_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        body = _check(
            self.http.request("POST", self.url, json={"texts": texts, "api_key": self._api_key}), "its_ai"
        )
        if body.get("error"):
            raise DetectorError(f"its_ai: {body['error']}")
        preds = body.get("preds")
        if not isinstance(preds, list) or len(preds) != len(texts):
            raise DetectorError("its_ai: malformed preds")
        return [{"word_probs": p} for p in preds]

    def _extract(self, p: dict[str, Any]) -> Extracted:
        probs = [float(x) for x in p["word_probs"]]
        if not probs:
            raise DetectorError("its_ai: empty predictions")
        mean = sum(probs) / len(probs)
        frac = sum(1 for x in probs if x >= 0.5) / len(probs)
        cands = {"mean_word_prob": _clip01(mean), "fraction_words_ai": _clip01(frac)}
        return Extracted(cands[self.primary], cands, None, {"n_words": len(probs)})

    def score(self, texts: list[str]) -> list[DetectorScore]:
        out: list[DetectorScore | None] = [None] * len(texts)
        todo: list[int] = []
        for i, t in enumerate(texts):
            hit = self.cache.get(self.name, self.config, text_hash(t)) if self.cache else None
            if hit is not None:
                out[i] = hit
            else:
                todo.append(i)
        for start in range(0, len(todo), self.batch_size):
            idx = todo[start : start + self.batch_size]
            batch = [texts[i] for i in idx]
            try:
                payloads = self._call_batch(batch)
            except Exception as e:
                for i in idx:
                    out[i] = DetectorScore(self.name, text_hash(texts[i]), float("nan"), error=str(e))
                continue
            for i, payload in zip(idx, payloads, strict=True):
                sha = text_hash(texts[i])
                try:
                    ex = self._extract(payload)
                    s = DetectorScore(self.name, sha, ex.score, ex.candidates, ex.version, ex.raw)
                    if self.cache:
                        self.cache.put(self.config, s)
                except Exception as e:
                    s = DetectorScore(self.name, sha, float("nan"), error=f"parse: {e}")
                out[i] = s
        return [s for s in out if s is not None]
