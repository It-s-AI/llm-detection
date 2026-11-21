"""Phrasly business API as an external reference humanizer (owner pilots only, never crowned).

``POST https://business.phrasly.ai/api/v1/humanize`` with ``x-api-key`` and ``{"text": ...}``;
the reply body is the humanized text as ``text/plain``. 20-5,000 words per request, 5 requests/s per
key, billed on input words (vendor docs, 2026-11):

* standard model: $0.14 per 1,000 words; ignores ``mode`` and the text settings;
* Phrasly Ultra (enabled per account on request): $0.30 per 1,000 words; ``mode`` from fewest
  changes to strongest rewrite ``gentle | light | balanced (default) | strong | max``; text settings
  ``more_commas``, ``sentence_length`` (``longer|shorter``), ``cleaner_grammar`` (+$0.01 each per
  1,000 words), ``proofread`` (+$0.04). ``model="latest"`` forces the standard model on an Ultra account.

The API reports no cost, so the ledger is charged a conservative estimate on the input length.

Inputs sent here leave our trusted-provider set (D-90): use it only on pilot periods that will
never be used for scoring.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx

from humanizer_eval.budget import SpendLedger
from humanizer_eval.detectors.http import HttpClient, RetryPolicy

BASE = "https://business.phrasly.ai"
PRICE_PER_1K_WORDS = 0.14
ULTRA_PRICE_PER_1K_WORDS = 0.30
ULTRA_SETTING_PRICE = {
    "more_commas": 0.01,
    "sentence_length": 0.01,
    "cleaner_grammar": 0.01,
    "proofread": 0.04,
}
ULTRA_MODES = ("gentle", "light", "balanced", "strong", "max")
MIN_WORDS, MAX_WORDS = 20, 5000
#: margin over the list price in case billing counts words differently from us (or output words)
COST_MARGIN = 1.25


def phrasly_cost(words: int, ultra: bool = False, settings: tuple[str, ...] = ()) -> float:
    """Conservative ledger estimate for one request of ``words`` input words."""
    rate = PRICE_PER_1K_WORDS
    if ultra:
        rate = ULTRA_PRICE_PER_1K_WORDS + sum(ULTRA_SETTING_PRICE[k] for k in settings)
    return round(math.ceil(max(words, MIN_WORDS) * COST_MARGIN) * rate / 1000, 6)


@dataclass(frozen=True)
class PhraslyResult:
    text: str | None  # None on failure
    error: str | None
    cost_usd: float


class Phrasly:
    service = "phrasly"

    def __init__(self, http: HttpClient, ledger: SpendLedger, options: dict | None = None):
        if ledger is None:
            raise ValueError("Phrasly requires a spend ledger")
        self.http = http
        self.ledger = ledger
        self.options = dict(options or {})  # extra body fields: model, mode, Ultra text settings
        mode = self.options.get("mode")
        if mode is not None and mode not in ULTRA_MODES:
            raise ValueError(f"unknown Phrasly mode {mode!r}; one of {ULTRA_MODES}")
        unknown = set(self.options) - {"model", "mode", "stream", *ULTRA_SETTING_PRICE}
        if unknown:
            raise ValueError(f"unknown Phrasly options {sorted(unknown)}")

    @classmethod
    def build(
        cls,
        api_key: str,
        ledger: SpendLedger,
        *,
        transport: httpx.BaseTransport | None = None,
        options: dict | None = None,
    ) -> Phrasly:
        if not api_key:
            raise ValueError("PHRASLY_API_KEY is not set")
        http = HttpClient(
            base_url=BASE,
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            rate_per_sec=4,
            burst=4,
            timeout=120.0,
            transport=transport,
            retry=RetryPolicy(max_attempts=3),
        )
        return cls(http, ledger, options)

    def humanize(self, text: str, ref: str = "") -> PhraslyResult:
        words = len(text.split())
        if not MIN_WORDS <= words <= MAX_WORDS:
            return PhraslyResult(None, f"length {words} words outside {MIN_WORDS}-{MAX_WORDS}", 0.0)
        settings = tuple(k for k in ULTRA_SETTING_PRICE if self.options.get(k))
        # unless the standard model is forced, an Ultra account may serve the request at the Ultra rate
        ultra = self.options.get("model") != "latest" and bool(self.options)
        est = phrasly_cost(words, ultra=ultra, settings=settings)
        res = self.ledger.reserve(self.service, est, ref=f"humanize:{ref}")
        try:
            resp = self.http.request("POST", "/api/v1/humanize", json={"text": text, **self.options})
        except RuntimeError:
            # a timed-out request may still have been processed and billed: keep the estimate
            res.commit(est, {"error": "transport"})
            return PhraslyResult(None, "transport failure", est)
        if resp.status_code >= 400:
            res.release({"status": resp.status_code})
            return PhraslyResult(None, f"HTTP {resp.status_code}: {resp.text[:200]}", 0.0)
        out = resp.text.strip()
        res.commit(est, {"words_in": words, "words_out": len(out.split())})
        if not out:
            return PhraslyResult(None, "empty response", est)
        return PhraslyResult(out, None, est)

    def humanize_all(
        self, texts: list[str], refs: list[str] | None = None, workers: int = 4
    ) -> list[PhraslyResult]:
        refs = refs or [str(i) for i in range(len(texts))]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(self.humanize, texts, refs))
