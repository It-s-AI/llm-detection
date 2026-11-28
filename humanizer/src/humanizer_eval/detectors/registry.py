"""Build configured detectors from settings."""

from __future__ import annotations

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import Settings, get_settings
from humanizer_eval.detectors.cache import DetectorCache
from humanizer_eval.detectors.vendors import (
    GPTZero,
    ItsAI,
    Originality,
    Pangram,
    VendorDetector,
    Winston,
    ZeroGPT,
)
from humanizer_eval.paths import ledger_path, var_dir

# Detectors that take part in crown scoring (D-30), each with its model version pinned in the request.
# ZeroGPT left on 2026-11-27: its API takes no version and reports none, and its model moved 4 times in 2.5
# days; Winston AI took its place on 2026-11-28 (version pinned in the request and checked in the answer,
# deterministic). Originality.ai is supported but not in the crown (it pins a model's name, not a version).
# It's AI is a simulation proxy only (D-33).
CROWN_DETECTORS = ("gptzero", "pangram", "winston")


def build_detector(
    name: str,
    *,
    settings: Settings | None = None,
    ledger: SpendLedger | None = None,
    use_cache: bool = True,
    transport=None,
) -> VendorDetector:
    s = settings or get_settings()
    led = ledger or SpendLedger(ledger_path(), s.caps())
    cache = DetectorCache(var_dir() / "detector_cache.sqlite") if use_cache else None
    kw = {"ledger": led, "cache": cache, "transport": transport}
    if name == "gptzero":
        return GPTZero.build(s.gptzero_api_key, **kw)
    if name == "winston":
        return Winston.build(s.winston_api_key, **kw)
    if name == "originality":
        return Originality.build(s.originality_api_key, **kw)
    if name == "zerogpt":
        return ZeroGPT.build(s.zerogpt_api_key, **kw)
    if name == "pangram":
        return Pangram.build(s.pangram_api_key, **kw)
    if name == "its_ai":
        return ItsAI.build(s.its_ai_special_url, s.its_ai_api_key, **kw)
    raise ValueError(f"unknown detector {name!r}")
